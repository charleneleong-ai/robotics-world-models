"""Render the Project #5 M1 few-shot curves from fewshot.json + anomalydino.json.

Two detectors vs the number of normal training images (k-shot) on MVTec-AD `cable`:
  - PatchCore (WideResNet-50 memory bank)
  - AnomalyDINO (training-free DINOv2-S/14 patch nearest-neighbour)
The point: foundation-model patch features win the extreme-low-data regime (1-4 shot);
PatchCore catches up by 8-shot. Run: `python fewshot.py`.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
patchcore = json.loads((HERE / "fewshot.json").read_text())
dino = json.loads((HERE / "anomalydino.json").read_text())

pc = [(r["n_normal"], r["auroc"]) for r in patchcore if r["k"] != "full"]
pc_full = next(r["auroc"] for r in patchcore if r["k"] == "full")
dn = [(r["k"], r["auroc"]) for r in dino]

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.plot([x for x, _ in dn], [y for _, y in dn], marker="o", lw=2,
        color="#c44", label="AnomalyDINO (DINOv2-S, training-free)", zorder=3)
ax.plot([x for x, _ in pc], [y for _, y in pc], marker="s", lw=2,
        color="#28a", label="PatchCore (WideResNet-50)", zorder=3)
ax.axhline(pc_full, ls="--", color="gray", alpha=0.6, label=f"PatchCore full-data ({pc_full:.3f})")
ax.annotate("+0.10 @ 1-shot", xy=(1, 0.81), fontsize=9, color="#c44")

ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8])
ax.set_xticklabels(["1", "2", "4", "8"])
ax.set_xlabel("Normal training images (k-shot)")
ax.set_ylabel("Image AUROC (MVTec-AD cable)")
ax.set_title("Project #5 M1 — few-shot: foundation-model vs memory-bank\nDINOv2 patch-NN wins the low-data regime")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
out = HERE / "m1_fewshot.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
