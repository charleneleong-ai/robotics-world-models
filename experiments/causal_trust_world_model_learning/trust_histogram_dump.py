#!/usr/bin/env python3
"""Dump per-sample trust scores from the absolute (existing) and relative
(z-scored) scorers over the same ManiSkill 4-task stream, and plot the two
distributions side by side -> fig_trust_histogram.pdf.

This is the paper's central diagnosis in one panel: the absolute scorer's
output is pinned near 0.25 with almost no spread, so every downstream use of
it is starved of signal.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continual_learning import WorldModelTrustCL
from rssm_world_model import WorldModel
from maniskill_benchmark import SimpleMLP, ManiSkillBenchmark
from relative_trust import RelativeTrustCL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1]
NUM_TASKS = 4


def collect(learner_cls, seed: int) -> list[float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    bench = ManiSkillBenchmark(num_tasks=NUM_TASKS, episodes_per_task=50, max_steps=100, obs_dim=64, action_dim=10)
    datasets = []
    for i, env in enumerate(bench.selected_envs):
        d = bench.collect_task_data(env, bench.episodes_per_task)
        datasets.append(bench.create_classification_task(d, i))
    num_classes = int(max(d["targets"].max().item() for d in datasets) + 1)
    obs_dim = datasets[0]["obs"].shape[1]
    model = SimpleMLP(obs_dim, hidden_dim=256, num_classes=num_classes).to(DEVICE)
    wm = WorldModel(obs_dim=obs_dim, action_dim=10, hidden_dim=256, stochastic_dim=16,
                    stochastic_classes=16, deterministic_dim=256).to(DEVICE)
    learner = learner_cls(model, wm, device=DEVICE, ewc_lambda=5000.0, trust_threshold=0.5)
    for task_id, ds in enumerate(datasets):
        idx_all = torch.randperm(len(ds["obs"]))
        for start in range(0, len(idx_all), 64):
            idx = idx_all[start:start + 64]
            batch = {"obs": ds["obs"][idx], "actions": torch.randn(len(idx), 10), "targets": ds["targets"][idx],
                     "next_obs": ds["obs"][idx] + torch.randn_like(ds["obs"][idx]) * 0.01, "task_id": task_id}
            learner.observe(batch)
    vals = []
    for v in learner.task_trust_scores.values():
        vals.extend(v)
    return vals


def main() -> None:
    absolute, relative = [], []
    for s in SEEDS:
        absolute += collect(WorldModelTrustCL, s)
        relative += collect(RelativeTrustCL, s)
        print(f"seed={s} collected", flush=True)
    stats_out = {
        "absolute": {"n": len(absolute), "min": float(np.min(absolute)), "max": float(np.max(absolute)),
                     "mean": float(np.mean(absolute)), "std": float(np.std(absolute))},
        "relative": {"n": len(relative), "min": float(np.min(relative)), "max": float(np.max(relative)),
                     "mean": float(np.mean(relative)), "std": float(np.std(relative))},
    }
    print(json.dumps(stats_out, indent=2))
    with open("trust_histogram_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    axes[0].hist(absolute, bins=60, color="#4a86e8")
    axes[0].set_xlim(0, 1)
    axes[0].set_title(f"Absolute scorer (std={stats_out['absolute']['std']:.3f})", fontsize=9)
    axes[0].set_xlabel("trust", fontsize=8)
    axes[1].hist(relative, bins=60, color="#e06666")
    axes[1].set_xlim(0, 1)
    axes[1].set_title(f"Relative (z-scored) scorer (std={stats_out['relative']['std']:.3f})", fontsize=9)
    axes[1].set_xlabel("trust", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("fig_trust_histogram.pdf", bbox_inches="tight")
    print("wrote fig_trust_histogram.pdf", flush=True)


if __name__ == "__main__":
    main()
