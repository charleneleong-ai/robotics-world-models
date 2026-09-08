#!/usr/bin/env python3
"""Does trust-weighting's per-sample differentiation matter for Table 6's
ContinualWAM (WorldModelTrustCL) result, given the real trust scorer's output
occupies an extremely narrow range (0.244-0.268, std~0.004)?

Ablation: ConstantTrustCL is identical to WorldModelTrustCL except every
per-sample trust score is replaced by the batch's own mean trust before
weighting the loss/KD/EWC penalty. This removes within-batch differentiation
between "reliable" and "unreliable" samples while preserving the same
overall magnitude (no confound from just scaling everything up or down).
If ContinualWAM's real backward-transfer advantage comes from trust
differentiation, this ablation should perform worse; if it comes from the
EWC-style penalty structure regardless of trust, performance should be
similar to the real ContinualWAM row in Table 6 (0.721+-0.020 accuracy,
-0.029+-0.011 backward transfer).
"""
from __future__ import annotations

import json
import sys
import os

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


class ConstantTrustCL(WorldModelTrustCL):
    """Real trust computed as usual, but broadcast to the batch's own mean
    before use -- isolates whether within-batch differentiation matters,
    holding the overall per-batch magnitude fixed.
    """

    def observe(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        targets = batch["targets"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        task_id = batch.get("task_id", 0)

        with torch.no_grad():
            pred_errors = self.world_model.compute_prediction_error(obs, actions, next_obs)
            trust_confidences = self.world_model.compute_trust(obs, actions)
        trust_scores = self.trust_scorer.compute_trust(pred_errors, trust_confidences, task_id)
        # Ablation: replace with the batch's own mean (same overall magnitude,
        # zero within-batch differentiation)
        trust_scores = torch.full_like(trust_scores, trust_scores.mean().item())

        if task_id not in self.task_trust_scores:
            self.task_trust_scores[task_id] = []
        self.task_trust_scores[task_id].extend(trust_scores.tolist())

        logits = self.model(obs)
        sample_weights = trust_scores.to(self.device)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        weighted_loss = (sample_weights * ce_loss).mean()

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

        self._task_samples.append({"obs": obs.detach().cpu(), "targets": targets.detach().cpu()})
        return {
            "loss": total_loss.item(),
            "accuracy": (logits.argmax(-1) == targets).float().mean().item(),
        }


def evaluate(model, dataset, device):
    model.eval()
    with torch.no_grad():
        logits = model(dataset["obs"].to(device))
        acc = (logits.argmax(-1) == dataset["targets"].to(device)).float().mean().item()
    return acc


def run_seed(seed: int, learner_cls):
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
    learner = learner_cls(model, wm, device=DEVICE, ewc_lambda=5000.0, trust_threshold=0.5)

    n = len(task_datasets)
    acc_matrix = [[0.0] * n for _ in range(n)]
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
        for j in range(task_id + 1):
            acc_matrix[task_id][j] = evaluate(learner.model, task_datasets[j], DEVICE)

    final_accs = [acc_matrix[n - 1][j] for j in range(n)]
    avg_accuracy = float(np.mean(final_accs))
    # Backward transfer: avg over tasks 0..n-2 of (final_acc - acc_right_after_training)
    bwt_terms = []
    for j in range(n - 1):
        acc_right_after = acc_matrix[j][j]
        acc_final = acc_matrix[n - 1][j]
        bwt_terms.append(acc_final - acc_right_after)
    bwt = float(np.mean(bwt_terms))
    return {"avg_accuracy": avg_accuracy, "backward_transfer": bwt}


def main():
    results = {"real_trust": [], "constant_trust": []}
    for seed in range(NUM_SEEDS):
        r_real = run_seed(seed, WorldModelTrustCL)
        r_const = run_seed(seed, ConstantTrustCL)
        results["real_trust"].append(r_real)
        results["constant_trust"].append(r_const)
        print(f"seed={seed} real_trust={r_real} constant_trust={r_const}", flush=True)

    from scipy import stats
    real_acc = [r["avg_accuracy"] for r in results["real_trust"]]
    const_acc = [r["avg_accuracy"] for r in results["constant_trust"]]
    real_bwt = [r["backward_transfer"] for r in results["real_trust"]]
    const_bwt = [r["backward_transfer"] for r in results["constant_trust"]]

    t_acc, p_acc = stats.ttest_rel(real_acc, const_acc)
    t_bwt, p_bwt = stats.ttest_rel(real_bwt, const_bwt)

    summary = {
        "real_trust": {"acc_mean": float(np.mean(real_acc)), "acc_std": float(np.std(real_acc)),
                       "bwt_mean": float(np.mean(real_bwt)), "bwt_std": float(np.std(real_bwt))},
        "constant_trust": {"acc_mean": float(np.mean(const_acc)), "acc_std": float(np.std(const_acc)),
                           "bwt_mean": float(np.mean(const_bwt)), "bwt_std": float(np.std(const_bwt))},
        "paired_t_acc": float(t_acc), "p_acc": float(p_acc),
        "paired_t_bwt": float(t_bwt), "p_bwt": float(p_bwt),
        "raw": results,
    }
    print(json.dumps(summary, indent=2))
    with open("trust_weighting_ablation_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
