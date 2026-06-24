"""Plot classical vs world-model robustness under pose/obs noise.

Reads classical_results.json + wm_results.json (each a list of
{sigma_mm, n, success_rate}) and plots success_rate vs sigma_mm for both
methods, saving divergence.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path) -> tuple[list[float], list[float]]:
    rows = sorted(json.loads(path.read_text()), key=lambda r: r["sigma_mm"])
    return [r["sigma_mm"] for r in rows], [r["success_rate"] for r in rows]


def main(
    classical: Path = typer.Option(Path("classical_results.json")),
    wm: Path = typer.Option(Path("wm_results.json")),
    out: Path = typer.Option(Path("divergence.png")),
) -> None:
    cx, cy = _load(classical)
    wx, wy = _load(wm)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cx, cy, marker="o", linewidth=2, label="Classical (RRTConnect + screw)")
    ax.plot(wx, wy, marker="s", linewidth=2, label="World model (TD-MPC2 / MPC)")
    ax.set_xlabel("Pose / observation noise  σ (mm)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Robustness under pose/obs noise: classical vs world model")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    typer.run(main)
