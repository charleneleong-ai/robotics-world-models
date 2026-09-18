#!/usr/bin/env python3
"""Cross-suite LIBERO sequence with real distribution shift.

Every null in this paper is explained with "simple tasks, minimal shift".
This tests that directly using Result 2's own protocol (train_sequential from
task_order_sensitivity, EMA trust vs none, same forgetting metric), but on a
task sequence that crosses suite boundaries: 3 tasks from Object, then 3
from Spatial, then 3 from Goal. The shift is the order, so the order is
fixed and seeds supply the variance (5 seeds x {mlp, rssm} x {none, ema}).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continualwam import get_backbone, load_demos, make_trust
from task_order_sensitivity import SUITE_DIRS, train_sequential, compute_forgetting

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = 5
TASKS_PER_SUITE = 3
SUITE_ORDER = ["object", "spatial", "goal"]
BACKBONES = ["mlp", "rssm"]
RESULTS_DIR = Path(__file__).parent / "results_cross_suite_shift"
RESULTS_DIR.mkdir(exist_ok=True)


def build_demos() -> list:
    demos = []
    for suite in SUITE_ORDER:
        demos += load_demos(SUITE_DIRS[suite], TASKS_PER_SUITE, 5)
    return demos


def run_one(backbone: str, trust: str, seed: int, demos: list) -> list[float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = demos[0][0]["obs"].shape[-1]
    act_dim = demos[0][0]["acts"].shape[-1]
    wm = get_backbone(backbone, obs_dim, act_dim).to(DEVICE)
    policy = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, act_dim),
    ).to(DEVICE)
    scorer = make_trust(trust, obs_dim, act_dim) if trust != "none" else None
    return train_sequential(wm, policy, demos, list(range(len(demos))), trust_scorer=scorer, device=DEVICE)


def main() -> None:
    demos = build_demos()
    print(f"loaded {len(demos)} tasks across {SUITE_ORDER}", flush=True)
    out = {}
    for backbone in BACKBONES:
        errs = {"none": [], "ema": []}
        for trust in ["none", "ema"]:
            for seed in range(SEEDS):
                e = run_one(backbone, trust, seed, demos)
                errs[trust].append(e)
                print(f"{backbone} {trust} seed={seed} final={e[-1]:.4f}", flush=True)
        m_none = compute_forgetting(errs["none"])
        m_ema = compute_forgetting(errs["ema"])
        f_none = m_none["per_ordering_forgetting"]
        f_ema = m_ema["per_ordering_forgetting"]
        t, p = stats.ttest_ind(f_none, f_ema, equal_var=False)
        out[backbone] = {
            "none": {"forgetting_mean": float(np.mean(f_none)), "forgetting_std": float(np.std(f_none)),
                     "final_error_mean": float(m_none["mean_final_error"])},
            "ema": {"forgetting_mean": float(np.mean(f_ema)), "forgetting_std": float(np.std(f_ema)),
                    "final_error_mean": float(m_ema["mean_final_error"])},
            "welch_t": float(t), "p_value": float(p),
            "reduction_pct": float(100 * (np.mean(f_none) - np.mean(f_ema)) / max(np.mean(f_none), 1e-9)),
            "raw_errors": errs,
        }
        print(f"== {backbone}: none={np.mean(f_none):.3f} ema={np.mean(f_ema):.3f} p={p:.4f}", flush=True)
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({b: {k: v for k, v in d.items() if k != "raw_errors"} for b, d in out.items()}, indent=2))


if __name__ == "__main__":
    main()
