#!/usr/bin/env python3
"""Does a pretrained video representation beat hand-crafted state vectors for LIBERO BC?

Per task, trains an identical MLP action head on each representation and scores it on
held-out demonstrations of that task (2 of 12 demos, split by demonstration, never by
frame, so no frames from an evaluation trajectory appear in training).

Representations
  state       21-dim proprioceptive vector (ee pose/state, gripper, joints) -- the
              representation every earlier experiment in this line used
  wan         3072-dim frozen Wan2.2 video-VAE latent of the agentview frame
  wan_pca21   the same latent projected to 21 dims by PCA fitted on the training split,
              matching `state` dimensionality so the comparison is not merely about width
  wan_state   concatenation of the two

Comparisons are paired across tasks (same tasks, same seeds) and tested with a paired
t-test at the task level.

Usage: wan_probe.py [suite] [n_seeds]
"""
from __future__ import annotations

import numpy as np
import typer
import wandb
from scipy import stats

from probe_common import CACHE_ROOT, TaskArrays, fit_probe, held_out_mask, load_task

REPS = ["state", "wan", "wan_pca21", "wan_state"]
N_EVAL_DEMOS, PCA_DIM = 2, 21


def features(task: TaskArrays, rep: str, tr: np.ndarray, ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat, st = task.features["wan"], task.state
    if rep == "state":
        return st[tr], st[ev]
    if rep == "wan":
        return lat[tr], lat[ev]
    if rep == "wan_state":
        return np.concatenate([lat[tr], st[tr]], 1), np.concatenate([lat[ev], st[ev]], 1)
    if rep == "wan_pca21":                      # PCA fitted on the training split only
        mu = lat[tr].mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(lat[tr] - mu, full_matrices=False)
        w = vt[:PCA_DIM].T
        return (lat[tr] - mu) @ w, (lat[ev] - mu) @ w
    raise ValueError(rep)


def report(per_task: dict[str, np.ndarray]) -> wandb.Table:
    table = wandb.Table(columns=["representation", "dims", "heldout_mse", "vs_state_%", "p_paired"])
    dims = {"state": 21, "wan": 3072, "wan_pca21": PCA_DIM, "wan_state": 3093}
    for r in REPS:
        delta = 100 * (per_task[r].mean() / per_task["state"].mean() - 1)
        p = float(stats.ttest_rel(per_task["state"], per_task[r])[1]) if r != "state" else float("nan")
        print(f"== {r:10s} dims={dims[r]:5d} held-out MSE {per_task[r].mean():.4f}  vs state {delta:+6.1f}%  p={p:.4f}")
        wandb.summary[f"mse/{r}"] = per_task[r].mean()
        wandb.summary[f"delta_vs_state/{r}"] = delta
        wandb.summary[f"p_vs_state/{r}"] = p
        table.add_data(r, dims[r], round(float(per_task[r].mean()), 5), round(float(delta), 2), round(p, 5))
    return table


def main(suite: str = typer.Argument("spatial"), n_seeds: int = typer.Argument(3)) -> None:
    cache = CACHE_ROOT / f"{suite}_agentview_rgb"
    n_tasks = len(list(cache.glob("task*.npz")))
    run = wandb.init(project="video-wam", job_type="probe", name=f"probe-{suite}",
                     config=dict(suite=suite, n_tasks=n_tasks, n_seeds=n_seeds, reps=REPS,
                                 backbone="Wan2.2-TI2V-5B-VAE", eval_demos_per_task=N_EVAL_DEMOS,
                                 epochs=30, pca_dim=PCA_DIM))
    err = {r: np.zeros((n_tasks, n_seeds)) for r in REPS}
    for ti in range(n_tasks):
        task = load_task({"wan": cache}, ti)
        ev = held_out_mask(task.demo, N_EVAL_DEMOS)
        tr = ~ev
        for rep in REPS:
            xtr, xev = features(task, rep, tr, ev)
            for s in range(n_seeds):
                err[rep][ti, s] = fit_probe(xtr, task.action[tr], xev, task.action[ev], s)
            print(f"task {ti} {rep:10s} held-out MSE {err[rep][ti].mean():.4f}", flush=True)
            wandb.log({"task": ti, f"heldout_mse/{rep}": err[rep][ti].mean()})
    print()
    wandb.log({"results": report({r: err[r].mean(1) for r in REPS})})   # pair at the task level
    np.savez(cache / "probe_results.npz", **err)
    run.finish()


if __name__ == "__main__":
    typer.run(main)
