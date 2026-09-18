#!/usr/bin/env python3
"""Held-out validation of ContinualWAM's trust_threshold.

Protocol: try each candidate threshold on validation seeds {0,1,2}, pick the
threshold with the best mean validation accuracy, then report that
threshold's performance on held-out seeds {3,4} -- which were not used for
selection -- against the current default (0.5) on the same held-out seeds.
"""
from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cl_baselines_full_rerun import ManiSkillBenchmark, ManiSkillCLExperiment
from continual_learning import WorldModelTrustCL
from maniskill_benchmark import SimpleMLP
from rssm_world_model import WorldModel

RESULTS_DIR = Path(__file__).parent / "results_threshold_tuning"
RESULTS_DIR.mkdir(exist_ok=True)

CANDIDATES = [0.3, 0.5, 0.7]
VAL_SEEDS = [0, 1, 2]
TEST_SEEDS = [3, 4]
NUM_TASKS = 4


def run_one(seed: int, threshold: float) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    bench = ManiSkillBenchmark(num_tasks=NUM_TASKS, episodes_per_task=50, max_steps=100,
                                obs_dim=64, action_dim=10)
    exp = ManiSkillCLExperiment(bench)
    exp.collect()
    obs_dim = exp.task_datasets[0]["obs"].shape[1]
    model_cls = lambda: SimpleMLP(obs_dim, hidden_dim=256, num_classes=10)
    world_model = WorldModel(obs_dim=obs_dim, action_dim=10, hidden_dim=256,
                              stochastic_dim=16, stochastic_classes=16, deterministic_dim=256)
    learner = WorldModelTrustCL(model_cls(), world_model, trust_threshold=threshold)
    return exp.run_method("world_model_trust_cl", learner)


def main() -> None:
    t0 = time.time()
    val_results = {th: [] for th in CANDIDATES}
    for th in CANDIDATES:
        for seed in VAL_SEEDS:
            m = run_one(seed, th)
            val_results[th].append(m["avg_accuracy"])
            print(f"[val] threshold={th} seed={seed} avg_acc={m['avg_accuracy']:.4f} [{time.time()-t0:.0f}s]",
                  flush=True)

    val_means = {th: float(np.mean(accs)) for th, accs in val_results.items()}
    best_threshold = max(val_means, key=val_means.get)
    print(f"\nValidation means: {val_means}")
    print(f"Best threshold (by validation): {best_threshold}\n")

    test_results = {"default_0.5": [], f"tuned_{best_threshold}": []}
    for seed in TEST_SEEDS:
        m_default = run_one(seed, 0.5)
        test_results["default_0.5"].append(m_default["avg_accuracy"])
        print(f"[test] default=0.5 seed={seed} avg_acc={m_default['avg_accuracy']:.4f} [{time.time()-t0:.0f}s]",
              flush=True)
        if best_threshold != 0.5:
            m_tuned = run_one(seed, best_threshold)
            test_results[f"tuned_{best_threshold}"].append(m_tuned["avg_accuracy"])
            print(f"[test] tuned={best_threshold} seed={seed} avg_acc={m_tuned['avg_accuracy']:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        else:
            test_results[f"tuned_{best_threshold}"].append(m_default["avg_accuracy"])

    summary = {
        "validation_means": val_means,
        "best_threshold": best_threshold,
        "held_out_test": {
            "default_0.5_mean": float(np.mean(test_results["default_0.5"])),
            "default_0.5_std": float(np.std(test_results["default_0.5"])),
            "tuned_mean": float(np.mean(test_results[f"tuned_{best_threshold}"])),
            "tuned_std": float(np.std(test_results[f"tuned_{best_threshold}"])),
            "seeds": test_results,
        },
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
