import json
import numpy as np
from scipy import stats

with open("sweep_5seeds_all.json") as f:
    data = json.load(f)

BACKBONES = ["mlp", "rssm", "jepa", "dreamerv3", "diffusion", "transformer"]
TRUST = ["none", "ema", "multi_step", "ensemble"]
SUITES = ["spatial", "object", "goal"]
SUITE_LABEL = {"spatial": "Spatial", "object": "Object", "goal": "Goal"}

n_sig_uncorrected = 0
n_total = 0
rows = {s: [] for s in SUITES}
for suite in SUITES:
    backbones = data[suite]
    for b in BACKBONES:
        methods = backbones[b]
        none_mean = methods["none"]["mean"]
        none_std = methods["none"]["std"]
        best_name = min(TRUST, key=lambda t: methods[t]["mean"])
        best_mean = methods[best_name]["mean"]
        best_std = methods[best_name]["std"]
        if best_name == "none":
            p_val = float("nan")
        else:
            none_seeds = np.array(methods["none"]["seeds"])
            best_seeds = np.array(methods[best_name]["seeds"])
            _, p_val = stats.ttest_rel(none_seeds, best_seeds)
            n_total += 1
            if p_val < 0.05:
                n_sig_uncorrected += 1
        rows[suite].append((b, none_mean, none_std, best_mean, best_std, p_val))

bonferroni_alpha = 0.05 / n_total
print(f"n_total tests={n_total}, uncorrected sig={n_sig_uncorrected}, bonferroni alpha={bonferroni_alpha:.5f}")

for suite in SUITES:
    print(f"\n\\textbf{{{SUITE_LABEL[suite]}}} & & & & & & \\\\")
    none_line = "\\; None"
    best_line = "\\; Best"
    p_line = "\\; $p$"
    for b, nm, ns, bm, bs, p in rows[suite]:
        none_line += f" & {nm:.3f}$\\pm${ns:.3f}"
        if np.isnan(p):
            best_line += f" & {bm:.3f}$\\pm${bs:.3f}"
            p_line += " & ---"
        else:
            marker = "$^*$" if p < 0.05 else ""
            best_line += f" & {bm:.3f}$\\pm${bs:.3f}{marker}"
            p_line += f" & {p:.3f}"
    print(none_line + " \\\\")
    print(best_line + " \\\\")
    print(p_line + " \\\\")
