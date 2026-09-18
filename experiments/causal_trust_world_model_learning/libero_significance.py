import json
from itertools import combinations

import numpy as np
from scipy import stats

with open("sweep_5seeds_all.json") as f:
    data = json.load(f)

BACKBONES = ["mlp", "rssm", "jepa", "dreamerv3", "diffusion", "transformer"]
TRUST = ["none", "ema", "multi_step", "ensemble"]

print(f"{'suite':10s} {'backbone':12s} {'best':10s} {'none':>8s} {'best_val':>9s} {'gain%':>7s} {'paired_t':>9s} {'p':>7s} {'sig?':>5s}")
n_sig = 0
n_total = 0
for suite, backbones in data.items():
    for b in BACKBONES:
        methods = backbones[b]
        none_seeds = np.array(methods["none"]["seeds"])
        best_name = min(TRUST, key=lambda t: methods[t]["mean"])
        best_seeds = np.array(methods[best_name]["seeds"])
        none_mean = methods["none"]["mean"]
        best_mean = methods[best_name]["mean"]
        gain = 100 * (none_mean - best_mean) / none_mean
        if best_name == "none":
            t_stat, p_val = 0.0, 1.0
        else:
            t_stat, p_val = stats.ttest_rel(none_seeds, best_seeds)
        sig = "YES" if (p_val < 0.05 and best_name != "none") else "no"
        n_total += 1
        if sig == "YES":
            n_sig += 1
        print(f"{suite:10s} {b:12s} {best_name:10s} {none_mean:8.4f} {best_mean:9.4f} {gain:6.1f}% {t_stat:9.3f} {p_val:7.4f} {sig:>5s}")
    print()

print(f"Significant (paired t-test, p<0.05, excluding best=none): {n_sig}/{n_total}")
