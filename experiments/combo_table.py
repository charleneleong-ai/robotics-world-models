#!/usr/bin/env python3
"""Does EWC add anything on top of replay?

Reads the two-axis sweep -- memory held fixed, EWC strength swept -- and tests each cell
against the same memory with no EWC. Orderings are shared across runs because the probe
draws them from a fixed seed, so the comparison is paired positionally; that holds only
while every run used the same `n_orderings` and `n_tasks`, which the saved JSON does not
record, so the length check below is the most this can verify.

All three metrics are losses: lower is better and a negative percentage is an improvement,
`plasticity` included despite its name. Percentages are ratios of means, descriptive only --
the p-value belongs to the paired differences. With this many cells and six orderings the
p-values are uncorrected and a handful below 0.05 are expected under the null.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from scipy import stats

METRICS = ("final", "abs_forgetting", "plasticity")
Cell = dict[str, dict[str, list[float]]]


def read_cell(root: Path, memory: int, strength: int) -> Cell:
    # Strength is the plain integer the sweep script wrote, never 1e6. The neighbouring
    # continual_er_<M>.json and continual_ewc_<L>.json are *different* arms -- replay with
    # no EWC code path, and EWC with no replay -- so the name has to be built exactly.
    return json.loads((root / f"continual_er{memory}_ewc_{strength}.json").read_text())["per_ordering"]


def load(root: Path, memories: list[int], strengths: list[int]) -> dict[int, dict[int, Cell]]:
    """Read every cell up front, so a missing file fails before any of the table is printed."""
    return {m: {s: read_cell(root, m, s) for s in [0, *strengths]} for m in memories}


def format_cell(base: list[float], arm: list[float]) -> str:
    x, y = np.asarray(base), np.asarray(arm)
    if len(x) != len(y):
        raise ValueError(f"cannot pair {len(x)} orderings against {len(y)}")
    p = stats.ttest_rel(x, y).pvalue
    shown = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    return f"{y.mean():.4f} ({100 * (y.mean() / x.mean() - 1):+5.1f}%, {shown}, {(y < x).sum()}/{len(x)} better)"


def main(root: Path = Path("/home/ubuntu/wan_latents"), memories: str = "100,2000",
         strengths: str = "10000,100000,1000000") -> None:
    grid = [int(s) for s in strengths.split(",")]
    cells = load(root, [int(m) for m in memories.split(",")], grid)
    typer.echo("all three metrics are losses: lower is better, a negative % is an improvement")
    for memory, arms in cells.items():
        base = arms[0]
        for metric in METRICS:
            label = "final_error" if metric == "final" else metric
            typer.echo(f"\n### memory {memory}, {label}: EWC on top of replay, vs the same memory alone")
            for rep in base:
                row = "  ".join(format_cell(base[rep][metric], arms[s][rep][metric]) for s in grid)
                typer.echo(f"{rep:>16}  alone {np.mean(base[rep][metric]):.4f}  |  {row}")


if __name__ == "__main__":
    typer.run(main)
