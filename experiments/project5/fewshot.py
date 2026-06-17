"""Render the Project #5 M1 few-shot curve from fewshot.json.

PatchCore image-AUROC on MVTec-AD `cable` vs the number of normal training images
(k-shot). The point: how little data buys most of the full-data accuracy -- the
"new factory line, few examples" story. Run: `python fewshot.py`.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
rows = json.loads((HERE / "fewshot.json").read_text())
xs = [r["n_normal"] for r in rows]
ys = [r["auroc"] for r in rows]
full = next(r["auroc"] for r in rows if r["k"] == "full")

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(xs, ys, marker="o", color="#2a6", linewidth=2, zorder=3)
ax.axhline(full, ls="--", color="gray", alpha=0.7, label=f"full-data ({full:.3f})")
for r in rows:
    ax.annotate(f'{r["auroc"]:.3f}', (r["n_normal"], r["auroc"]),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.set_xscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([str(r["k"]) for r in rows])
ax.set_xlabel("Normal training images (k-shot)")
ax.set_ylabel("Image AUROC (MVTec-AD cable)")
ax.set_title("Project #5 M1 — PatchCore few-shot curve\n0.925 AUROC from 8 images")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
out = HERE / "m1_fewshot.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
