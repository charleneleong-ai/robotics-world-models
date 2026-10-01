#!/usr/bin/env python3
"""Reads a sweep of continual_probe JSONs and asks whether the method closes the gap to state.

Serves any single-axis sweep the probe can run -- EWC strength, replay memory size -- by
globbing the files that sweep wrote and reading the swept value out of each filename.

Rows are paired *across runs*: run N's ordering i is compared with run M's ordering i. That is
only legitimate because ContinualExperiment seeds its orderings from a fixed random.Random(0)
and every run in the sweep used the same --n-orderings, so ordering i is the same task sequence
everywhere. `load` asserts the ordering counts agree rather than letting a mismatch pair silently.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from scipy import stats

REF = "state"
Runs = dict[float, dict[str, dict[str, list[float]]]]


def load(root: Path, pattern: str) -> Runs:
    """Read each result file, taking the swept value from whatever the glob's `*` matched."""
    prefix, star, suffix = pattern.partition("*")
    if not star:
        raise ValueError(f"pattern {pattern} needs a * where the swept value goes")
    runs: Runs = {}
    for path in sorted(root.glob(pattern)):
        token = path.name[len(prefix) : len(path.name) - len(suffix)]
        try:
            value = float(token)
        except ValueError as err:
            raise ValueError(f"cannot read a swept value out of {path.name}") from err
        if value in runs:
            raise ValueError(f"{path.name} maps to {value:g}, which another file already claimed")
        runs[value] = json.loads(path.read_text())["per_ordering"]
    if not runs:
        raise FileNotFoundError(f"no results matching {pattern} under {root}")
    arms = {tuple(sorted(r)) for r in runs.values()}
    assert len(arms) == 1, "runs disagree on which representations they hold; pairing invalid"
    widths = {len(next(iter(r.values()))["final"]) for r in runs.values()}
    assert len(widths) == 1, f"runs disagree on ordering count {widths}; cross-setting pairing invalid"
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
    """Each metric's mean by swept value, one column per representation."""
    for metric in ("final", "abs_forgetting", "plasticity"):
        typer.echo(f"\n### {metric}" + ("   (a loss: lower is more plastic)" if metric == "plasticity" else ""))
        for setting in runs:
            typer.echo(f"{setting:>10g} | "
                       + " | ".join(f"{mean_of(runs, setting, r, metric):14.4f}" for r in reps))


def gap_by_setting(runs: Runs, best: str) -> None:
    typer.echo(f"\n### {best} minus {REF}, paired across orderings")
    typer.echo(f"{'setting':>10} | {'final gap':>10} {'p':>8} {'wins':>7} | {'forget gap':>10} {'p':>8} {'wins':>7}")
    for setting, data in runs.items():
        gf, pf, wf, n = paired(data[REF]["final"], data[best]["final"])
        gg, pg, wg, _ = paired(data[REF]["abs_forgetting"], data[best]["abs_forgetting"])
        typer.echo(f"{setting:>10g} | {gf:+10.4f} {pf:8.4f} {wf:4d}/{n} | {gg:+10.4f} {pg:8.4f} {wg:4d}/{n}")


def tuned(runs: Runs, reps: list[str]) -> dict[str, float]:
    """The swept value minimising mean final error, per representation.

    Ties break toward the smallest setting, which matters because the post-hoc selection below
    should never prefer a stronger intervention that bought nothing.

    Selection and the significance tests share the same orderings, so those p-values are
    optimistically biased and uncorrected for the choice over the grid. They say the tuned arms
    differ, not that the tuning was justified.
    """
    return {rep: min(runs, key=lambda s: mean_of(runs, s, rep, "final")) for rep in reps}


def report_tuned(runs: Runs, reps: list[str], best: dict[str, float], axis: str) -> None:
    baseline = min(runs)
    typer.echo(f"\n### best {axis} per representation (lowest final error; selection is post hoc)")
    for rep in reps:
        base, got = mean_of(runs, baseline, rep, "final"), mean_of(runs, best[rep], rep, "final")
        _, p, w, n = paired(runs[baseline][rep]["final"], runs[best[rep]][rep]["final"])
        verdict = f"lowest {axis} is best" if best[rep] == baseline else f"p={p:.4f}, {w}/{n}"
        typer.echo(f"{rep:>16}  {axis}={best[rep]:>10g}  {base:.4f} -> {got:.4f} "
                   f"({100 * (got / base - 1):+.1f}%, {verdict})")

    typer.echo(f"\n### each representation at its own best {axis}, vs {REF} at its best")
    ref_vals = runs[best[REF]][REF]["final"]
    for rep in reps:
        vals = runs[best[rep]][rep]["final"]
        gap, p, w, n = paired(ref_vals, vals)
        pct = 100 * (float(np.mean(vals)) / float(np.mean(ref_vals)) - 1)
        verdict = "reference" if rep == REF else f"p={p:.4f} {w}/{n}"
        typer.echo(f"{rep:>16}  {np.mean(vals):.4f}  gap {gap:+.4f} ({pct:+.1f}%) {verdict}")


def main(root: Path = Path("/home/ubuntu/wan_latents"), pattern: str = "continual_ewc_*.json",
         axis: str = "lambda", best: str = "openvla_ctx8") -> None:
    runs = load(root, pattern)
    reps = list(next(iter(runs.values())))
    typer.echo(f"{axis:>10} | " + " | ".join(f"{r:>14}" for r in reps))
    curves(runs, reps)
    gap_by_setting(runs, best)
    report_tuned(runs, reps, tuned(runs, reps), axis)


if __name__ == "__main__":
    typer.run(main)
