"""Render the PegInsertionSide world-model-vs-floor learning curves from curves.json.

TD-MPC2 (3 seeds) success_once vs env steps, against the PPO floor's 0.000 at 50M.
Log-x so the ~50x sample-efficiency gap (TD-MPC2 solves at ~1M, PPO null at 50M) is
visible on one axis. Data recovered from W&B (runs killed by the 12h cap at ~1M steps).
Run: `python curve.py`.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = json.loads((HERE / "curves.json").read_text())
td = d["tdmpc2"]

fig, ax = plt.subplots(figsize=(7.6, 4.8))
for seed, color in [("s1", "#c44"), ("s2", "#28a"), ("s3", "#2a2")]:
    xs = [p[0] * 1000 for p in td[seed]]
    ys = [p[1] for p in td[seed]]
    ax.plot(xs, ys, marker="o", ms=3, lw=1.8, color=color, alpha=0.9,
            label="TD-MPC2 seed %s (→ %.2f)" % (seed[-1], td["final_success_once"][seed]))

ax.scatter([d["ppo"]["steps_k"] * 1000], [d["ppo"]["mean"]], s=120, marker="X",
           color="black", zorder=5, label="PPO floor: 0.00 @ 50M")
ax.annotate("PPO: 0.00 success\nat 50M steps (50x more data)",
            xy=(50_000_000, 0.0), xytext=(6_000_000, 0.18), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"))

ax.set_xscale("log")
ax.set_xlabel("Environment steps")
ax.set_ylabel("success_once")
ax.set_ylim(-0.03, 1.0)
ax.set_title("PegInsertionSide-v1 — world model vs model-free floor\nTD-MPC2 solves the contact-rich task at ~50x less data")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
out = HERE / "learning_curve.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
