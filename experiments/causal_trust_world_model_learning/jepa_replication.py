#!/usr/bin/env python3
"""Independent replication of Result 2's per-sample trust-weighted protocol on
JEPA, on fresh random orderings (Random(1): distinct from experiment A's Random(0)
and from the original unseeded run). Arms: none vs per_sample, paired on
identical orderings and seeds; tested at the ordering level (seeds averaged).
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import per_param_consolidation as m
from continualwam import load_demos
from task_order_sensitivity import SUITE_DIRS, compute_forgetting

N_ORD, N_SEED = 6, 3

demos = load_demos(SUITE_DIRS["spatial"], 10, 5)
rng = random.Random(1)
orderings = []
for _ in range(N_ORD):
    o = list(range(10)); rng.shuffle(o); orderings.append(o)

per_arm = {"none": [], "per_sample": []}
for oi, order in enumerate(orderings):
    for seed in range(N_SEED):
        for arm in per_arm:
            errs = m.run_A("jepa", arm, demos, order, seed)
            per_arm[arm].append(errs)
            print(f"[R] jepa ord={oi} seed={seed} arm={arm:10s} final={errs[-1]:.4f}", flush=True)

def by_ordering(vals):
    return np.array(vals).reshape(N_ORD, N_SEED).mean(1)

fn = by_ordering(compute_forgetting(per_arm["none"])["per_ordering_forgetting"])
fp = by_ordering(compute_forgetting(per_arm["per_sample"])["per_ordering_forgetting"])
en = by_ordering([e[-1] for e in per_arm["none"]])
ep = by_ordering([e[-1] for e in per_arm["per_sample"]])
out = {
    "forgetting_none": float(fn.mean()), "forgetting_per_sample": float(fp.mean()),
    "reduction_pct": float(100 * (fn.mean() - fp.mean()) / fn.mean()),
    "p_forgetting_paired_by_ordering": float(stats.ttest_rel(fn, fp)[1]),
    "final_none": float(en.mean()), "final_per_sample": float(ep.mean()),
    "p_final_paired_by_ordering": float(stats.ttest_rel(en, ep)[1]),
    "orderings": orderings, "raw": per_arm,
}
print("== [R] jepa: " + json.dumps({k: round(v, 4) for k, v in out.items() if isinstance(v, float)}), flush=True)
os.makedirs("results_per_param_consolidation", exist_ok=True)
json.dump(out, open("results_per_param_consolidation/jepa_replication.json", "w"), indent=2)
