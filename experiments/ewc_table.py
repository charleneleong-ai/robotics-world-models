#!/usr/bin/env python3
"""Reads the per-lambda continual_probe JSONs and asks whether EWC closes the gap to state.

Rows are paired *across runs*: run N's ordering i is compared with run M's ordering i. That is
only legitimate because ContinualExperiment seeds its orderings from a fixed random.Random(0)
and every run in the sweep used the same --n-orderings, so ordering i is the same task sequence
everywhere. `load` asserts the ordering counts agree rather than letting a mismatch pair silently.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import typer
from scipy import stats

REF = "state"
BEST = "openvla_ctx8"  # best backbone representation from the single-task probe
Runs = dict[float, dict[str, dict[str, list[float]]]]


def load(root: Path, pattern: str) -> Runs:
    runs: Runs = {}
    for path in sorted(root.glob(pattern)):
        match = re.search(r"_ewc_(.+)\.json$", path.name)
        if not match:
            raise ValueError(f"cannot read a lambda out of {path.name}")
        runs[float(match.group(1))] = json.loads(path.read_text())["per_ordering"]
    if not runs:
        raise FileNotFoundError(f"no results matching {pattern} under {root}")
    widths = {len(next(iter(r.values()))["final"]) for r in runs.values()}
    assert len(widths) == 1, f"runs disagree on ordering count {widths}; cross-lambda pairing invalid"
    return dict(sorted(runs.items()))


def paired(a: list[float], b: list[float]) -> tuple[float, float, int, int]:
    """Mean gap b-a, its paired p-value, how many orderings favour b, and how many there were."""
    x, y = np.array(a), np.array(b)
    if np.array_equal(x, y):
        return 0.0, float("nan"), 0, len(x)
    return float(y.mean() - x.mean()), float(stats.ttest_rel(x, y).pvalue), int((y < x).sum()), len(x)


def mean_of(runs: Runs, lam: float, rep: str, metric: str) -> float:
    return float(np.mean(runs[lam][rep][metric]))


def curves(runs: Runs, reps: list[str]) -> None:
    for metric in ("final", "abs_forgetting", "plasticity"):
        typer.echo(f"\n### {metric}" + ("   (a loss: lower is more plastic)" if metric == "plasticity" else ""))
        for lam in runs:
            typer.echo(f"{lam:>10g} | " + " | ".join(f"{mean_of(runs, lam, r, metric):14.4f}" for r in reps))


def gap_by_lambda(runs: Runs) -> None:
    typer.echo(f"\n### {BEST} minus {REF}, paired across orderings")
    typer.echo(f"{'lambda':>10} | {'final gap':>10} {'p':>8} {'wins':>7} | {'forget gap':>10} {'p':>8} {'wins':>7}")
    for lam, data in runs.items():
        gf, pf, wf, n = paired(data[REF]["final"], data[BEST]["final"])
        gg, pg, wg, _ = paired(data[REF]["abs_forgetting"], data[BEST]["abs_forgetting"])
        typer.echo(f"{lam:>10g} | {gf:+10.4f} {pf:8.4f} {wf:4d}/{n} | {gg:+10.4f} {pg:8.4f} {wg:4d}/{n}")


def tuned(runs: Runs, reps: list[str]) -> dict[str, float]:
    """Lambda minimising mean final error, per representation.

    Selection and the significance tests below share the same orderings, so those p-values are
    optimistically biased and uncorrected for the choice over the lambda grid. They say the tuned
    arms differ, not that the tuning was justified.
    """
    return {rep: min(runs, key=lambda lam: mean_of(runs, lam, rep, "final")) for rep in reps}


def report_tuned(runs: Runs, reps: list[str], best: dict[str, float]) -> None:
    baseline = min(runs)
    typer.echo("\n### best lambda per representation (lowest final error; selection is post hoc)")
    for rep in reps:
        base, got = mean_of(runs, baseline, rep, "final"), mean_of(runs, best[rep], rep, "final")
        _, p, w, n = paired(runs[baseline][rep]["final"], runs[best[rep]][rep]["final"])
        verdict = "unregularised is best" if best[rep] == baseline else f"p={p:.4f}, {w}/{n}"
        typer.echo(f"{rep:>16}  lambda={best[rep]:>10g}  {base:.4f} -> {got:.4f} "
                   f"({100 * (got / base - 1):+.1f}%, {verdict})")

    typer.echo(f"\n### each representation at its own best lambda, vs {REF} at its best")
    ref_vals = runs[best[REF]][REF]["final"]
    for rep in reps:
        vals = runs[best[rep]][rep]["final"]
        gap, p, w, n = paired(ref_vals, vals)
        pct = 100 * (float(np.mean(vals)) / float(np.mean(ref_vals)) - 1)
        verdict = "reference" if rep == REF else f"p={p:.4f} {w}/{n}"
        typer.echo(f"{rep:>16}  {np.mean(vals):.4f}  gap {gap:+.4f} ({pct:+.1f}%) {verdict}")


def main(root: Path = Path("/home/ubuntu/wan_latents"), pattern: str = "continual_ewc_*.json") -> None:
    runs = load(root, pattern)
    reps = list(next(iter(runs.values())))
    typer.echo(f"{'lambda':>10} | " + " | ".join(f"{r:>14}" for r in reps))
    curves(runs, reps)
    gap_by_lambda(runs)
    report_tuned(runs, reps, tuned(runs, reps))


if __name__ == "__main__":
    typer.run(main)
