#!/usr/bin/env python3
"""Real trust-rejection experiment: threshold sweep (#1) and preventive
corruption-detection test (#4), built on the existing real ManiSkillCLExperiment
infrastructure (same classes used for Table 6's CL-baselines comparison).

Unlike the previous "Trust Threshold Sensitivity" result (which simulated
fake trust scores from a Beta distribution and a hardcoded reward formula,
never touching the real model), this subclasses the real WorldModelTrustCL
and adds a genuine accept/reject branch: batches with mean trust below the
threshold skip the optimizer step entirely (theta unchanged), instead of
just being down-weighted.
"""
from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continual_learning import WorldModelTrustCL
from rssm_world_model import WorldModel
from maniskill_benchmark import SimpleMLP, ManiSkillBenchmark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TASKS = 4
NUM_SEEDS = 5
EPOCHS_PER_TASK = 5
# Calibrated to the real TrustScorer's observed range (0.244-0.268 across
# seeds/tasks; std=0.004) -- the paper's original theta_safe=0.5 is entirely
# outside this range and would reject 100% of updates unconditionally.
THRESHOLDS = [0.245, 0.250, 0.253, 0.256, 0.260]
CORRUPTION_PROB = 0.2
RESULTS_DIR = Path(__file__).parent / "results_trust_rejection"
RESULTS_DIR.mkdir(exist_ok=True)


class TrustRejectionCL(WorldModelTrustCL):
    """Real accept/reject: batches with mean trust below threshold skip the
    optimizer step entirely (parameters unchanged), rather than being
    down-weighted as in the base class's consolidation-only mechanism.
    """

    def __init__(self, *args, corruption_prob: float = 0.0, trust_threshold: float = 0.5, **kwargs):
        super().__init__(*args, trust_threshold=trust_threshold, **kwargs)
        self.trust_threshold = trust_threshold
        self.corruption_prob = corruption_prob
        self.n_batches = 0
        self.n_rejected = 0
        self.n_corrupted = 0
        self.n_corrupted_rejected = 0
        self.n_clean_rejected = 0

    def observe(self, batch: dict) -> dict:
        is_corrupted = False
        if self.corruption_prob > 0 and np.random.rand() < self.corruption_prob:
            is_corrupted = True
            batch = dict(batch)
            batch["obs"] = batch["obs"] + torch.randn_like(batch["obs"]) * 2.0

        self.model.train()
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        targets = batch["targets"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        task_id = batch.get("task_id", 0)

        with torch.no_grad():
            pred_errors = self.world_model.compute_prediction_error(obs, actions, next_obs)
            trust_confidences = self.world_model.compute_trust(obs, actions)
        trust_scores = self.trust_scorer.compute_trust(pred_errors, trust_confidences, task_id)
        mean_trust = trust_scores.mean().item()

        self.n_batches += 1
        rejected = mean_trust < self.trust_threshold
        if is_corrupted:
            self.n_corrupted += 1
        if rejected:
            self.n_rejected += 1
            if is_corrupted:
                self.n_corrupted_rejected += 1
            else:
                self.n_clean_rejected += 1
            if task_id not in self.task_trust_scores:
                self.task_trust_scores[task_id] = []
            self.task_trust_scores[task_id].extend(trust_scores.tolist())
            return {"loss": float("nan"), "accuracy": float("nan"), "rejected": True}

        self.optimizer.zero_grad()
        logits = self.model(obs)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        weighted_loss = (trust_scores.to(self.device) * ce_loss).mean()
        if self.previous_models:
            kd_loss = torch.tensor(0.0, device=self.device)
            for prev_task_id, prev_model in self.previous_models.items():
                prev_model.eval()
                with torch.no_grad():
                    prev_logits = prev_model(obs)
                kl = F.kl_div(F.log_softmax(logits / 2, dim=-1), F.softmax(prev_logits / 2, dim=-1), reduction="none").sum(dim=-1)
                prev_trust = np.mean(self.task_trust_scores.get(prev_task_id, [0.5]))
                kd_loss = kd_loss + prev_trust * self.kd_weight * kl.mean()
            weighted_loss = weighted_loss + kd_loss
        ewc_penalty = self.consolidation.compute_penalty()
        total_loss = weighted_loss + ewc_penalty
        total_loss.backward()
        self.optimizer.step()

        if task_id not in self.task_trust_scores:
            self.task_trust_scores[task_id] = []
        self.task_trust_scores[task_id].extend(trust_scores.tolist())
        self._task_samples.append({"obs": obs.detach().cpu(), "targets": targets.detach().cpu()})
        return {
            "loss": total_loss.item(),
            "accuracy": (logits.argmax(-1) == targets).float().mean().item(),
            "rejected": False,
        }


def evaluate(model, dataset, device):
    model.eval()
    with torch.no_grad():
        logits = model(dataset["obs"].to(device))
        acc = (logits.argmax(-1) == dataset["targets"].to(device)).float().mean().item()
    return acc


def run_seed(seed: int, threshold: float, corruption_prob: float = 0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    bench = ManiSkillBenchmark(num_tasks=NUM_TASKS, episodes_per_task=50, max_steps=100, obs_dim=64, action_dim=10)
    task_datasets = []
    for i, env_name in enumerate(bench.selected_envs):
        data = bench.collect_task_data(env_name, bench.episodes_per_task)
        task_datasets.append(bench.create_classification_task(data, i))
    num_classes = int(max(d["targets"].max().item() for d in task_datasets) + 1)
    obs_dim = task_datasets[0]["obs"].shape[1]

    model = SimpleMLP(obs_dim, hidden_dim=256, num_classes=num_classes).to(DEVICE)
    wm = WorldModel(obs_dim=obs_dim, action_dim=10, hidden_dim=256, stochastic_dim=16, stochastic_classes=16, deterministic_dim=256).to(DEVICE)
    learner = TrustRejectionCL(model, wm, device=DEVICE, ewc_lambda=5000.0, trust_threshold=threshold, corruption_prob=corruption_prob)

    n = len(task_datasets)
    final_accs = []
    for task_id in range(n):
        dataset = task_datasets[task_id]
        batch_size = min(64, len(dataset["obs"]))
        for _ in range(EPOCHS_PER_TASK):
            indices = torch.randperm(len(dataset["obs"]))
            for start in range(0, len(indices), batch_size):
                idx = indices[start:start + batch_size]
                batch = {
                    "obs": dataset["obs"][idx],
                    "actions": torch.randn(len(idx), 10),
                    "targets": dataset["targets"][idx],
                    "next_obs": dataset["obs"][idx] + torch.randn_like(dataset["obs"][idx]) * 0.01,
                    "task_id": task_id,
                }
                learner.observe(batch)
        if hasattr(learner, "consolidate"):
            avg_trust = np.mean(learner.task_trust_scores.get(task_id, [0.5]))
            learner.consolidate(task_id, avg_trust)

    for j in range(n):
        final_accs.append(evaluate(learner.model, task_datasets[j], DEVICE))

    result = {
        "avg_accuracy": float(np.mean(final_accs)),
        "rejection_rate": learner.n_rejected / max(1, learner.n_batches),
    }
    if corruption_prob > 0:
        result["corrupted_rejection_rate"] = learner.n_corrupted_rejected / max(1, learner.n_corrupted)
        result["clean_rejection_rate"] = learner.n_clean_rejected / max(1, (learner.n_batches - learner.n_corrupted))
    return result


def main():
    # Experiment 1: threshold sweep
    sweep_results = {}
    for threshold in THRESHOLDS:
        accs, rej_rates = [], []
        for seed in range(NUM_SEEDS):
            r = run_seed(seed, threshold)
            accs.append(r["avg_accuracy"])
            rej_rates.append(r["rejection_rate"])
            print(f"[sweep] threshold={threshold} seed={seed} acc={r['avg_accuracy']:.4f} rej={r['rejection_rate']:.4f}", flush=True)
        sweep_results[threshold] = {
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "rej_mean": float(np.mean(rej_rates)), "rej_std": float(np.std(rej_rates)),
            "accs": accs, "rej_rates": rej_rates,
        }

    # Experiment 2: preventive corruption detection (median threshold)
    CORRUPTION_THRESHOLD = 0.253
    corr_results = {"corrupted_rejection_rate": [], "clean_rejection_rate": []}
    for seed in range(NUM_SEEDS):
        r = run_seed(seed, CORRUPTION_THRESHOLD, corruption_prob=CORRUPTION_PROB)
        corr_results["corrupted_rejection_rate"].append(r["corrupted_rejection_rate"])
        corr_results["clean_rejection_rate"].append(r["clean_rejection_rate"])
        print(f"[corruption] seed={seed} corrupted_rej={r['corrupted_rejection_rate']:.4f} clean_rej={r['clean_rejection_rate']:.4f}", flush=True)

    output = {"threshold_sweep": sweep_results, "corruption_detection": corr_results}
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
