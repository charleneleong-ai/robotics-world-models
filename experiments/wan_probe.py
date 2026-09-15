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

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from scipy import stats

SUITE = sys.argv[1] if len(sys.argv) > 1 else "spatial"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
CACHE = f"/home/ubuntu/wan_latents/{SUITE}_agentview_rgb"
REPS = ["state", "wan", "wan_pca21", "wan_state"]
N_EVAL_DEMOS, EPOCHS, BATCH, LR, PCA_DIM = 2, 30, 256, 1e-3, 21
DEVICE = "cuda"


def load_task(ti: int) -> dict[str, np.ndarray]:
    z = np.load(os.path.join(CACHE, f"task{ti:02d}.npz"))
    return {k: z[k] for k in ("latent", "action", "state", "demo")}


def split(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the last N_EVAL_DEMOS demonstrations of the task, whole trajectories."""
    demos = np.unique(d["demo"])
    held = set(demos[-N_EVAL_DEMOS:].tolist())
    is_eval = np.array([x in held for x in d["demo"]])
    return ~is_eval, is_eval


def features(d: dict[str, np.ndarray], rep: str, tr: np.ndarray, ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat, st = d["latent"].astype(np.float32), d["state"]
    if rep == "state":
        return st[tr], st[ev]
    if rep == "wan":
        return lat[tr], lat[ev]
    if rep == "wan_state":
        return np.concatenate([lat[tr], st[tr]], 1), np.concatenate([lat[ev], st[ev]], 1)
    if rep == "wan_pca21":                      # PCA fitted on the training split only
        x = lat[tr]; mu = x.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
        w = vt[:PCA_DIM].T
        return (lat[tr] - mu) @ w, (lat[ev] - mu) @ w
    raise ValueError(rep)


def run_probe(xtr, ytr, xev, yev, seed: int) -> float:
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    xtr, xev = (xtr - mu) / sd, (xev - mu) / sd
    xtr_t = torch.tensor(xtr, device=DEVICE); ytr_t = torch.tensor(ytr, device=DEVICE)
    xev_t = torch.tensor(xev, device=DEVICE); yev_t = torch.tensor(yev, device=DEVICE)
    net = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, ytr.shape[1])).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xtr_t), device=DEVICE)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.mse_loss(net(xtr_t[idx]), ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return float(F.mse_loss(net(xev_t), yev_t))


def main() -> None:
    n_tasks = len([f for f in os.listdir(CACHE) if f.endswith(".npz")])
    run = wandb.init(project="video-wam", job_type="probe", name=f"probe-{SUITE}",
                     config=dict(suite=SUITE, n_tasks=n_tasks, n_seeds=N_SEEDS, reps=REPS,
                                 backbone="Wan2.2-TI2V-5B-VAE", eval_demos_per_task=N_EVAL_DEMOS,
                                 epochs=EPOCHS, pca_dim=PCA_DIM))
    err = {r: np.zeros((n_tasks, N_SEEDS)) for r in REPS}
    for ti in range(n_tasks):
        d = load_task(ti); tr, ev = split(d)
        for rep in REPS:
            xtr, xev = features(d, rep, tr, ev)
            for s in range(N_SEEDS):
                err[rep][ti, s] = run_probe(xtr, d["action"][tr], xev, d["action"][ev], s)
            print(f"task {ti} {rep:10s} held-out MSE {err[rep][ti].mean():.4f}", flush=True)
            wandb.log({"task": ti, f"heldout_mse/{rep}": err[rep][ti].mean()})

    per_task = {r: err[r].mean(1) for r in REPS}          # pair at the task level
    table = wandb.Table(columns=["representation", "dims", "heldout_mse", "vs_state_%", "p_paired"])
    dims = {"state": 21, "wan": 3072, "wan_pca21": PCA_DIM, "wan_state": 3093}
    print()
    for r in REPS:
        delta = 100 * (per_task[r].mean() / per_task["state"].mean() - 1)
        p = float(stats.ttest_rel(per_task["state"], per_task[r])[1]) if r != "state" else float("nan")
        print(f"== {r:10s} dims={dims[r]:5d} held-out MSE {per_task[r].mean():.4f}  vs state {delta:+6.1f}%  p={p:.4f}")
        wandb.summary[f"mse/{r}"] = per_task[r].mean()
        wandb.summary[f"delta_vs_state/{r}"] = delta
        wandb.summary[f"p_vs_state/{r}"] = p
        table.add_data(r, dims[r], round(float(per_task[r].mean()), 5), round(float(delta), 2), round(p, 5))
    wandb.log({"results": table})
    np.savez(os.path.join(CACHE, "probe_results.npz"), **{r: err[r] for r in REPS})
    run.finish()


if __name__ == "__main__":
    main()
