#!/usr/bin/env python3
"""Experiment B: prediction error as a task-boundary change-point detector.

Instead of gating each batch by a trust threshold (structurally degenerate,
Appendix), use the world model's *pre-update* prediction error as a change-point
statistic. Stream LIBERO-Spatial's 10 tasks in a fixed order (boundaries known),
train an online world model one 16-step chunk at a time, record each chunk's
prediction error BEFORE updating on it, and run a CUSUM against reference
statistics taken from a lagged trailing window (so the current spike cannot
contaminate its own baseline). Baseline: a raw z-score threshold.

Metrics per detector setting: recall (boundaries with an alarm within D steps),
precision (alarms that fall inside some boundary window), F1, mean detection delay.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continualwam import get_backbone, load_demos
from task_order_sensitivity import SUITE_DIRS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUITE = "spatial"
N_TASKS = 10
MAX_DEMOS = 20
CHUNK = 16
BACKBONES = ["mlp", "rssm", "jepa"]
SEEDS = [0, 1, 2, 3, 4]
WINDOW, LAG, REFRACTORY, DETECT_WITHIN = 30, 5, 10, 15
CUSUM_K = 0.5
CUSUM_H = [2.0, 4.0, 6.0, 8.0, 12.0]
Z_THRESH = [2.0, 3.0, 4.0]
RESULTS_DIR = Path(__file__).parent / "results_boundary_cusum"
RESULTS_DIR.mkdir(exist_ok=True)


def stream(backbone: str, seed: int, demos) -> tuple[list[float], list[int]]:
    torch.manual_seed(seed); np.random.seed(seed)
    obs_dim = demos[0][0]["obs"].shape[-1]; act_dim = demos[0][0]["acts"].shape[-1]
    wm = get_backbone(backbone, obs_dim, act_dim).to(DEVICE)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    errors, boundaries, step = [], [], 0
    for task_demos in demos:
        boundaries.append(step)
        chunks = []
        for d in task_demos:
            o, a = d["obs"], d["acts"]
            for i in range(0, len(o) - CHUNK - 1, CHUNK):
                chunks.append((o[i:i + CHUNK + 1], a[i:i + CHUNK + 1]))
        np.random.shuffle(chunks)
        for o, a in chunks:
            ot = torch.tensor(o, dtype=torch.float32, device=DEVICE)
            at = torch.tensor(a, dtype=torch.float32, device=DEVICE)
            wm.eval()
            with torch.no_grad():
                e = wm.predict_error(ot[:-1], at[:-1], ot[1:]).mean().item()
            errors.append(e)
            wm.train()
            opt.zero_grad()
            wm.train_loss(ot.unsqueeze(0), at.unsqueeze(0)).backward()
            opt.step()
            step += 1
    return errors, boundaries[1:]  # first "boundary" at step 0 is not a change


def reference(errors: list[float], t: int) -> tuple[float, float]:
    lo, hi = max(0, t - LAG - WINDOW), max(0, t - LAG)
    win = errors[lo:hi]
    if len(win) < 5:
        return float("nan"), float("nan")
    return float(np.mean(win)), float(np.std(win) + 1e-8)


def alarms_cusum(errors, h, k=CUSUM_K) -> list[int]:
    s, out, cool = 0.0, [], 0
    for t, e in enumerate(errors):
        mu, sd = reference(errors, t)
        if np.isnan(mu):
            continue
        s = max(0.0, s + (e - mu) / sd - k)
        if cool > 0:
            cool -= 1
            continue
        if s > h:
            out.append(t); s = 0.0; cool = REFRACTORY
    return out


def alarms_zthresh(errors, z) -> list[int]:
    out, cool = [], 0
    for t, e in enumerate(errors):
        mu, sd = reference(errors, t)
        if np.isnan(mu):
            continue
        if cool > 0:
            cool -= 1
            continue
        if (e - mu) / sd > z:
            out.append(t); cool = REFRACTORY
    return out


def score(alarms: list[int], boundaries: list[int]) -> dict:
    hits, delays, used = 0, [], set()
    for b in boundaries:
        cands = [a for a in alarms if b <= a < b + DETECT_WITHIN and a not in used]
        if cands:
            hits += 1; delays.append(cands[0] - b); used.add(cands[0])
    tp = len(used)
    fp = len(alarms) - tp
    recall = hits / len(boundaries)
    precision = tp / len(alarms) if alarms else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"recall": recall, "precision": precision, "f1": f1,
            "mean_delay": float(np.mean(delays)) if delays else float("nan"),
            "n_alarms": len(alarms), "false_alarms": fp}


def main() -> None:
    demos = load_demos(SUITE_DIRS[SUITE], N_TASKS, MAX_DEMOS)
    results, traces = {}, {}
    for backbone in BACKBONES:
        per_setting: dict[str, list[dict]] = {}
        for seed in SEEDS:
            errors, boundaries = stream(backbone, seed, demos)
            if seed == 0:
                traces[backbone] = {"errors": errors, "boundaries": boundaries}
            for h in CUSUM_H:
                per_setting.setdefault(f"cusum_h{h}", []).append(score(alarms_cusum(errors, h), boundaries))
            for z in Z_THRESH:
                per_setting.setdefault(f"zthresh_{z}", []).append(score(alarms_zthresh(errors, z), boundaries))
            print(f"{backbone} seed={seed} steps={len(errors)} boundaries={len(boundaries)}", flush=True)
        agg = {}
        for k, runs in per_setting.items():
            agg[k] = {m: (float(np.nanmean([r[m] for r in runs])), float(np.nanstd([r[m] for r in runs])))
                      for m in ["recall", "precision", "f1", "mean_delay", "false_alarms"]}
        results[backbone] = agg
        best = max(agg.items(), key=lambda kv: kv[1]["f1"][0])
        print(f"== {backbone}: best {best[0]} F1={best[1]['f1'][0]:.3f} recall={best[1]['recall'][0]:.3f} "
              f"precision={best[1]['precision'][0]:.3f} delay={best[1]['mean_delay'][0]:.1f}", flush=True)
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump({"results": results, "traces_seed0": traces}, f, indent=2)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
