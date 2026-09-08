import json
import numpy as np
from scipy import stats

with open("results_maniskill_backbone_sweep/aggregated.json") as f:
    agg = json.load(f)

BACKBONES = ["mlp", "rssm", "jepa", "dreamerv3", "diffusion", "transformer"]
TRUST = ["none", "ema", "multi_step", "ensemble"]

n_sig = 0
n_total = 0
for b in BACKBONES:
    methods = agg[b]
    none_seeds = np.array(methods["none"]["seeds"])
    best_name = min(TRUST, key=lambda t: methods[t]["mean"])
    best_seeds = np.array(methods[best_name]["seeds"])
    none_mean, none_std = methods["none"]["mean"], methods["none"]["std"]
    best_mean, best_std = methods[best_name]["mean"], methods[best_name]["std"]
    if best_name == "none":
        p_val = float("nan")
        line = f"{b:12s} & {none_mean:.4f}$\\pm${none_std:.4f} & --- (None) & --- \\\\"
    else:
        t, p_val = stats.ttest_rel(none_seeds, best_seeds)
        n_total += 1
        if p_val < 0.05:
            n_sig += 1
        marker = "$^*$" if p_val < 0.05 else ""
        line = f"{b:12s} & {none_mean:.4f}$\\pm${none_std:.4f} & {best_mean:.4f}$\\pm${best_std:.4f} ({best_name}){marker} & {p_val:.3f} \\\\"
    print(line)

print(f"\n% significant (uncorrected): {n_sig}/{n_total}, bonferroni alpha={0.05/n_total:.4f}")
