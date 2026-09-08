#!/usr/bin/env python3
"""Trust-guided buffer curation for GDumb: does using prediction-reliability
to decide *which* buffer sample to evict (instead of GDumb's random choice
within the largest class) improve over plain GDumb?

Trust-prioritised *sampling* during training already failed (Result 5).
This tests a different lever: under a fixed buffer capacity, when a class is
full and a new sample must displace one, evict the *most reliably predicted*
(highest-trust, i.e. most "already known"/redundant) existing member of the
largest class, keeping the harder, less-reliable ones -- the same underlying
signal, applied to a genuinely different decision.
"""
from __future__ import annotations

import json
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rssm_world_model import WorldModel
from trust_scoring import TrustScorer
from maniskill_benchmark import SimpleMLP, ManiSkillBenchmark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TASKS = 4
NUM_SEEDS = 5
BUFFER_SIZE = 500
RETRAIN_EPOCHS = 20


class GDumbBase:
    """Plain GDumb: greedy class-balanced buffer, random eviction within the
    largest class. Retrains from scratch on the buffer after each task.
    """

    def __init__(self, model_cls, lr=1e-3, device=DEVICE, buffer_size=BUFFER_SIZE):
        self.model_cls = model_cls
        self.model = model_cls().to(device)
        self.lr = lr
        self.device = device
        self.buffer_size = buffer_size
        self.buffer_obs, self.buffer_targets = [], []
        self.buffer_actions, self.buffer_next_obs = [], []

    def _pick_evict(self, candidates: list[int]) -> int:
        return candidates[np.random.randint(len(candidates))]

    def observe(self, batch: dict) -> None:
        obs, targets = batch["obs"], batch["targets"]
        actions, next_obs = batch["actions"], batch["next_obs"]
        for o, t, a, no in zip(obs, targets, actions, next_obs):
            if len(self.buffer_obs) < self.buffer_size:
                self.buffer_obs.append(o.clone())
                self.buffer_targets.append(t.clone())
                self.buffer_actions.append(a.clone())
                self.buffer_next_obs.append(no.clone())
            else:
                counts = torch.bincount(torch.stack(self.buffer_targets))
                largest_class = counts.argmax().item()
                candidates = [i for i, bt in enumerate(self.buffer_targets) if bt.item() == largest_class]
                evict = self._pick_evict(candidates)
                self.buffer_obs[evict] = o.clone()
                self.buffer_targets[evict] = t.clone()
                self.buffer_actions[evict] = a.clone()
                self.buffer_next_obs[evict] = no.clone()

    def consolidate(self) -> None:
        self.model = self.model_cls().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        obs = torch.stack(self.buffer_obs).to(self.device)
        targets = torch.stack(self.buffer_targets).to(self.device)
        self.model.train()
        for _ in range(RETRAIN_EPOCHS):
            perm = torch.randperm(len(obs))
            for start in range(0, len(obs), 64):
                idx = perm[start:start + 64]
                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(obs[idx]), targets[idx])
                loss.backward()
                optimizer.step()


class TrustGuidedGDumb(GDumbBase):
    """Same buffer/retraining logic, but evicts the *highest-trust* (most
    reliably predicted by an online world model) member of the largest
    class, instead of a random one.
    """

    def __init__(self, model_cls, obs_dim, act_dim, lr=1e-3, device=DEVICE, buffer_size=BUFFER_SIZE):
        super().__init__(model_cls, lr, device, buffer_size)
        self.world_model = WorldModel(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=256,
                                       stochastic_dim=16, stochastic_classes=16, deterministic_dim=256).to(device)
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr)
        self.trust_scorer = TrustScorer()

    def observe(self, batch: dict) -> None:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        # Online world-model update (purely for scoring buffer candidates)
        self.wm_optimizer.zero_grad()
        pred_errors = self.world_model.compute_prediction_error(obs, actions, next_obs)
        wm_loss = pred_errors.mean()
        wm_loss.backward()
        self.wm_optimizer.step()
        super().observe(batch)

    def _pick_evict(self, candidates: list[int]) -> int:
        cand_obs = torch.stack([self.buffer_obs[i] for i in candidates]).to(self.device)
        cand_act = torch.stack([self.buffer_actions[i] for i in candidates]).to(self.device)
        cand_next = torch.stack([self.buffer_next_obs[i] for i in candidates]).to(self.device)
        with torch.no_grad():
            pred_errors = self.world_model.compute_prediction_error(cand_obs, cand_act, cand_next)
        # Highest trust = lowest prediction error = most reliably predicted = evict
        best_idx = int(torch.argmin(pred_errors).item())
        return candidates[best_idx]


def evaluate(model, dataset, device):
    model.eval()
    with torch.no_grad():
        logits = model(dataset["obs"].to(device))
        return (logits.argmax(-1) == dataset["targets"].to(device)).float().mean().item()


def run_seed(seed: int, use_trust: bool):
    torch.manual_seed(seed)
    np.random.seed(seed)
    bench = ManiSkillBenchmark(num_tasks=NUM_TASKS, episodes_per_task=50, max_steps=100, obs_dim=64, action_dim=10)
    task_datasets = []
    for i, env_name in enumerate(bench.selected_envs):
        data = bench.collect_task_data(env_name, bench.episodes_per_task)
        task_datasets.append(bench.create_classification_task(data, i))
    num_classes = int(max(d["targets"].max().item() for d in task_datasets) + 1)
    obs_dim = task_datasets[0]["obs"].shape[1]
    model_cls = lambda: SimpleMLP(obs_dim, hidden_dim=256, num_classes=num_classes)

    if use_trust:
        learner = TrustGuidedGDumb(model_cls, obs_dim=obs_dim, act_dim=10, device=DEVICE)
    else:
        learner = GDumbBase(model_cls, device=DEVICE)

    n = len(task_datasets)
    acc_matrix = [[0.0] * n for _ in range(n)]
    for task_id in range(n):
        dataset = task_datasets[task_id]
        batch_size = min(64, len(dataset["obs"]))
        indices = torch.randperm(len(dataset["obs"]))
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            batch = {
                "obs": dataset["obs"][idx],
                "actions": torch.randn(len(idx), 10),
                "targets": dataset["targets"][idx],
                "next_obs": dataset["obs"][idx] + torch.randn_like(dataset["obs"][idx]) * 0.01,
            }
            learner.observe(batch)
        learner.consolidate()
        for j in range(task_id + 1):
            acc_matrix[task_id][j] = evaluate(learner.model, task_datasets[j], DEVICE)

    final_accs = [acc_matrix[n - 1][j] for j in range(n)]
    avg_accuracy = float(np.mean(final_accs))
    bwt_terms = [acc_matrix[n - 1][j] - acc_matrix[j][j] for j in range(n - 1)]
    bwt = float(np.mean(bwt_terms))
    return {"avg_accuracy": avg_accuracy, "backward_transfer": bwt}


def main():
    results = {"gdumb_plain": [], "gdumb_trust_curated": []}
    for seed in range(NUM_SEEDS):
        r_plain = run_seed(seed, use_trust=False)
        r_trust = run_seed(seed, use_trust=True)
        results["gdumb_plain"].append(r_plain)
        results["gdumb_trust_curated"].append(r_trust)
        print(f"seed={seed} plain={r_plain} trust_curated={r_trust}", flush=True)

    from scipy import stats
    plain_acc = [r["avg_accuracy"] for r in results["gdumb_plain"]]
    trust_acc = [r["avg_accuracy"] for r in results["gdumb_trust_curated"]]
    plain_bwt = [r["backward_transfer"] for r in results["gdumb_plain"]]
    trust_bwt = [r["backward_transfer"] for r in results["gdumb_trust_curated"]]

    t_acc, p_acc = stats.ttest_rel(plain_acc, trust_acc)
    t_bwt, p_bwt = stats.ttest_rel(plain_bwt, trust_bwt)

    summary = {
        "gdumb_plain": {"acc_mean": float(np.mean(plain_acc)), "acc_std": float(np.std(plain_acc)),
                        "bwt_mean": float(np.mean(plain_bwt)), "bwt_std": float(np.std(plain_bwt))},
        "gdumb_trust_curated": {"acc_mean": float(np.mean(trust_acc)), "acc_std": float(np.std(trust_acc)),
                                "bwt_mean": float(np.mean(trust_bwt)), "bwt_std": float(np.std(trust_bwt))},
        "paired_t_acc": float(t_acc), "p_acc": float(p_acc),
        "paired_t_bwt": float(t_bwt), "p_bwt": float(p_bwt),
        "raw": results,
    }
    print(json.dumps(summary, indent=2))
    with open("trust_guided_gdumb_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
