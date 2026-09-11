#!/usr/bin/env python3
"""Why FLOW fails across suites: dispersion of the previous policy's loss on the next task.

For each transition type, train a BC policy on three tasks (plain BC, 3 seeds), compute its
per-sample loss on the next task and the FLOW weights w = exp(-l / median l), and report
how dispersed the losses are and how many samples the weighting actually down-weights.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "temp"]  # flow_robustness reads an experiment name at import
import flow_robustness as fr  # noqa: E402
from continualwam import load_demos, train_bc  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_per_param_consolidation", "flow_weight_diagnostic.json")


def make_policy(od: int, ad: int) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(od, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, ad)).to(fr.fw.DEVICE)


def weight_stats(train_tasks, next_task, seed: int) -> dict[str, float]:
    torch.manual_seed(seed); np.random.seed(seed)
    od, ad = train_tasks[0][0]["obs"].shape[-1], train_tasks[0][0]["acts"].shape[-1]
    pol = make_policy(od, ad)
    for t in train_tasks:
        train_bc(pol, t, fr.m.BC_EPOCHS, device=fr.fw.DEVICE)
    pol.eval()
    obs, act, _ = fr.fw.transitions(next_task)
    with torch.no_grad():
        loss = F.mse_loss(pol(obs), act, reduction="none").mean(dim=-1)
    w = torch.exp(-loss / loss.median())
    lo, wn = loss.cpu().numpy(), w.cpu().numpy()
    return {"loss_median": float(np.median(lo)), "loss_cv": float(lo.std() / lo.mean()),
            "loss_iqr_ratio": float(np.percentile(lo, 75) / np.percentile(lo, 25)),
            "w_mean": float(wn.mean()), "w_std": float(wn.std()),
            "frac_w_below_0.2": float((wn < 0.2).mean()), "frac_w_above_0.8": float((wn > 0.8).mean())}


def main() -> None:
    s = fr.SUITE_DIRS
    spa, obj, goal = load_demos(s["spatial"], 10, 5), load_demos(s["object"], 3, 5), load_demos(s["goal"], 3, 5)
    cases = {"within_suite_spatial0-2_to_spatial3": (spa[:3], spa[3]),
             "cross_suite_object0-2_to_spatial0": (obj, spa[0]),
             "cross_suite_spatial0-2_to_goal0": (spa[:3], goal[0])}
    out = {}
    for name, (train, nxt) in cases.items():
        runs = [weight_stats(train, nxt, seed) for seed in range(3)]
        out[name] = {"per_seed": runs, "mean": {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}}
        print(name, json.dumps({k: round(v, 3) for k, v in out[name]["mean"].items()}), flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
