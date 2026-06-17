"""Render the Project #5 M1 accuracy<->latency<->compute Pareto from results.jsonl.

Each point is one anomaly-detection model exported to OpenVINO INT8 and benchmarked
on CPU (MVTec-AD `cable`). Marker size encodes INT8 model size on disk (MB) -- the
third Pareto axis (compute/footprint). Run: `python pareto.py`.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
rows = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines() if l.strip()]

fig, ax = plt.subplots(figsize=(7.5, 5.2))
for r in rows:
    size = r["int8_mb"]
    ax.scatter(r["int8_ms"], r["auroc"], s=size * 12, alpha=0.55,
               edgecolors="black", linewidths=1.2, zorder=3)
    ax.annotate(f'{r["model"]}\n{size:.0f} MB',
                (r["int8_ms"], r["auroc"]),
                textcoords="offset points", xytext=(10, 8), fontsize=9)

ax.set_xlabel("CPU latency (ms/img, OpenVINO INT8)")
ax.set_ylabel("Image AUROC (MVTec-AD cable)")
ax.set_title("Project #5 M1 — edge inspection Pareto\n(marker area = INT8 model size on disk)")
ax.grid(True, alpha=0.3)
ax.set_ylim(0.74, 1.01)
fig.tight_layout()
out = HERE / "m1_pareto.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
