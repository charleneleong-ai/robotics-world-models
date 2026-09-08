#!/usr/bin/env python3
"""Relative (z-scored) trust: fixes the absolute scorer's collapsed range.

The existing TrustScorer computes trust = (1 - clip(err/ema_err, 0, 2)/2) * conf.
Because each sample's error is close to the running EMA of errors, the first
factor is pinned near 0.5, and conf ~ 0.5, so trust is structurally ~0.25 with
almost no per-sample spread (observed: 0.244-0.268, std ~0.004). Every
downstream consumer of the signal (loss weighting, update rejection, replay
priority, buffer eviction, threshold gating) is therefore starved of signal.

RelativeTrustScorer instead z-scores each sample's error against a running
per-task mean/variance and maps trust = sigmoid(-z), which spans (0, 1) with
real within-batch spread. Same interface, drop-in for WorldModelTrustCL.

Reuses the existing matched harnesses unchanged:
  Part A: trust_weighting_ablation.run_seed  -> Table-6-style acc/BWT, real vs relative
  Part B: trust_rejection_real.run_seed      -> threshold sweep + corruption test with relative trust
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

from trust_scoring import TrustScorer
from continual_learning import WorldModelTrustCL
import trust_weighting_ablation as abl
import trust_rejection_real as rej

NUM_SEEDS = 5
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]  # meaningful now that trust spans (0,1)
CORRUPTION_THRESHOLD = 0.5
RESULTS_DIR = Path(__file__).parent / "results_relative_trust"
RESULTS_DIR.mkdir(exist_ok=True)


class RelativeTrustScorer(TrustScorer):
    """z-score per-sample prediction error against a running per-task
    mean/variance (EMA), trust = sigmoid(-z)."""

    def __init__(self, ema_alpha: float = 0.95, trust_threshold: float = 0.5):
        super().__init__(ema_alpha=ema_alpha, trust_threshold=trust_threshold)
        self.task_ema_sq: dict[int, float] = {}

    def compute_trust(self, prediction_errors: torch.Tensor, confidences: torch.Tensor, task_id: int) -> torch.Tensor:
        errors = prediction_errors.detach().cpu().numpy().astype(np.float64)
        m = float(errors.mean())
        s2 = float((errors ** 2).mean())
        if task_id not in self.task_ema_error:
            self.task_ema_error[task_id] = m
            self.task_ema_sq[task_id] = s2
        else:
            a = self.ema_alpha
            self.task_ema_error[task_id] = a * self.task_ema_error[task_id] + (1 - a) * m
            self.task_ema_sq[task_id] = a * self.task_ema_sq[task_id] + (1 - a) * s2
        mu = self.task_ema_error[task_id]
        var = max(self.task_ema_sq[task_id] - mu * mu, 1e-12)
        z = (errors - mu) / np.sqrt(var)
        trust = 1.0 / (1.0 + np.exp(np.clip(z, -20, 20)))  # sigmoid(-z): low error -> high trust
        return torch.tensor(trust, dtype=torch.float32)


class RelativeTrustCL(WorldModelTrustCL):
    def __init__(self, *args, trust_threshold: float = 0.5, **kwargs):
        super().__init__(*args, trust_threshold=trust_threshold, **kwargs)
        self.trust_scorer = RelativeTrustScorer(trust_threshold=trust_threshold)


class RelativeTrustRejectionCL(rej.TrustRejectionCL):
    def __init__(self, *args, trust_threshold: float = 0.5, **kwargs):
        super().__init__(*args, trust_threshold=trust_threshold, **kwargs)
        self.trust_scorer = RelativeTrustScorer(trust_threshold=trust_threshold)


def part_a_weighting() -> dict:
    real, rel = [], []
    for seed in range(NUM_SEEDS):
        r = abl.run_seed(seed, WorldModelTrustCL)
        q = abl.run_seed(seed, RelativeTrustCL)
        real.append(r)
        rel.append(q)
        print(f"[A] seed={seed} absolute={r} relative={q}", flush=True)
    ra = [x["avg_accuracy"] for x in real]
    qa = [x["avg_accuracy"] for x in rel]
    rb = [x["backward_transfer"] for x in real]
    qb = [x["backward_transfer"] for x in rel]
    t_a, p_a = stats.ttest_rel(ra, qa)
    t_b, p_b = stats.ttest_rel(rb, qb)
    return {
        "absolute": {"acc_mean": float(np.mean(ra)), "acc_std": float(np.std(ra)),
                     "bwt_mean": float(np.mean(rb)), "bwt_std": float(np.std(rb))},
        "relative": {"acc_mean": float(np.mean(qa)), "acc_std": float(np.std(qa)),
                     "bwt_mean": float(np.mean(qb)), "bwt_std": float(np.std(qb))},
        "p_acc": float(p_a), "p_bwt": float(p_b), "raw": {"absolute": real, "relative": rel},
    }


def part_b_rejection() -> dict:
    rej.TrustRejectionCL = RelativeTrustRejectionCL  # run_seed builds this name
    sweep = {}
    for thr in THRESHOLDS:
        accs, rejs = [], []
        for seed in range(NUM_SEEDS):
            r = rej.run_seed(seed, thr)
            accs.append(r["avg_accuracy"])
            rejs.append(r["rejection_rate"])
            print(f"[B-sweep] thr={thr} seed={seed} acc={r['avg_accuracy']:.4f} rej={r['rejection_rate']:.4f}", flush=True)
        sweep[thr] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                      "rej_mean": float(np.mean(rejs)), "rej_std": float(np.std(rejs)),
                      "accs": accs, "rej_rates": rejs}
    corr = {"corrupted_rejection_rate": [], "clean_rejection_rate": []}
    for seed in range(NUM_SEEDS):
        r = rej.run_seed(seed, CORRUPTION_THRESHOLD, corruption_prob=rej.CORRUPTION_PROB)
        corr["corrupted_rejection_rate"].append(r["corrupted_rejection_rate"])
        corr["clean_rejection_rate"].append(r["clean_rejection_rate"])
        print(f"[B-corr] seed={seed} corrupted_rej={r['corrupted_rejection_rate']:.4f} clean_rej={r['clean_rejection_rate']:.4f}", flush=True)
    t, p = stats.ttest_rel(corr["corrupted_rejection_rate"], corr["clean_rejection_rate"])
    corr["p_value"] = float(p)
    return {"threshold_sweep": sweep, "corruption_detection": corr}


def main() -> None:
    out = {"part_a_weighting": part_a_weighting(), "part_b_rejection": part_b_rejection()}
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
