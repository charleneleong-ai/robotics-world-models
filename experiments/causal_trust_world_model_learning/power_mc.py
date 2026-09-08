import json
import numpy as np
from scipy import stats
from scipy.optimize import brentq

rng = np.random.default_rng(0)

def simulate_power(d, n, alpha, n_sims=20000):
    # Paired t-test: simulate n paired differences ~ N(d, 1) (in units of std),
    # test against 0 with a two-sided one-sample t-test.
    rejections = 0
    for _ in range(n_sims):
        sample = rng.normal(loc=d, scale=1.0, size=n)
        t_stat, p = stats.ttest_1samp(sample, 0.0)
        if p < alpha:
            rejections += 1
    return rejections / n_sims

def mde_d(n, alpha, target_power=0.8):
    f = lambda d: simulate_power(d, n, alpha, n_sims=8000) - target_power
    return brentq(f, 0.05, 6.0, xtol=0.02)

n = 5
d_unc = mde_d(n, 0.05)
print(f"Minimum detectable Cohen's d (of the paired difference) at n={n}, alpha=0.05, power=0.8: {d_unc:.3f}")

alpha_bonf_lib = 0.05 / 13
d_bonf_lib = mde_d(n, alpha_bonf_lib)
print(f"Minimum detectable Cohen's d at n={n}, alpha={alpha_bonf_lib:.4f} (LIBERO Bonferroni), power=0.8: {d_bonf_lib:.3f}")

alpha_bonf_ms = 0.05 / 5
d_bonf_ms = mde_d(n, alpha_bonf_ms)
print(f"Minimum detectable Cohen's d at n={n}, alpha={alpha_bonf_ms:.4f} (ManiSkill Bonferroni), power=0.8: {d_bonf_ms:.3f}")

with open("sweep_5seeds_all.json") as f:
    libero = json.load(f)
with open("results_maniskill_backbone_sweep/aggregated.json") as f:
    maniskill = json.load(f)

print("\n=== LIBERO: minimum detectable relative effect (%) at 80% power, n=5 ===")
print("(using each cell's own none-condition std as the effect-size unit)")
for suite in ["spatial", "object", "goal"]:
    for b in ["mlp", "rssm", "jepa", "dreamerv3", "diffusion", "transformer"]:
        std = libero[suite][b]["none"]["std"]
        mean = libero[suite][b]["none"]["mean"]
        mde_unc = d_unc * std
        mde_bonf = d_bonf_lib * std
        print(f"{suite:8s} {b:12s} mean={mean:.4f} std={std:.4f} -> MDE_uncorr={100*mde_unc/mean:5.1f}%  MDE_bonf={100*mde_bonf/mean:5.1f}%")

print("\n=== ManiSkill: minimum detectable relative effect (%) at 80% power, n=5 ===")
for b in ["mlp", "rssm", "jepa", "dreamerv3", "transformer"]:
    std = maniskill[b]["none"]["std"]
    mean = maniskill[b]["none"]["mean"]
    mde_unc = d_unc * std
    mde_bonf = d_bonf_ms * std
    print(f"{b:12s} mean={mean:.4f} std={std:.4f} -> MDE_uncorr={100*mde_unc/mean:5.1f}%  MDE_bonf={100*mde_bonf/mean:5.1f}%")
