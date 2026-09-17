#!/usr/bin/env python3
"""Which representation is best for LIBERO behaviour cloning?

Probes identical MLP action heads on every cached representation, paired across the same
tasks, seeds and held-out split (2 of 12 demonstrations per task, split by whole
trajectory). Reports each against three references so the comparisons that matter are
explicit: proprioception (is video worth anything?), the per-frame DiT (does temporal
context buy anything, i.e. is the dynamics prior doing work?), and the VAE (does the
world model beat its own tokenizer?).

  state          21-dim proprioception, the representation the earlier papers used
  vae            frozen Wan VAE latent, per frame
  dit_ctx1       Wan DiT mid-block features, one frame -- temporal attention inert
  dit_ctx8       Wan DiT mid-block features, 8-frame window
  dit_ctx8_state dit_ctx8 concatenated with proprioception
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from scipy import stats

ROOT = "/home/ubuntu/wan_latents"
CTXS = [1, 4, 8, 16]
SOURCES = {"vae": f"{ROOT}/spatial_agentview_rgb",
           **{f"dit_ctx{c}": f"{ROOT}/spatial_dit_ctx{c}" for c in CTXS},
           "siglip2": f"{ROOT}/spatial_siglip2", "siglip2_ctx8": f"{ROOT}/spatial_siglip2_ctx8",
           "openvla": f"{ROOT}/spatial_openvla", "openvla_ctx8": f"{ROOT}/spatial_openvla_ctx8"}
# 1-frame group and 2-frame group are each internally matched on temporal context
REPS = (["state", "vae", "dit_ctx1", "siglip2", "openvla"]
        + ["dit_ctx4", "dit_ctx8", "dit_ctx16", "siglip2_ctx8", "openvla_ctx8"]
        + ["openvla_ctx8_state"])
N_TASKS, N_SEEDS, N_EVAL_DEMOS, EPOCHS, BATCH, LR = 10, 3, 2, 30, 256, 1e-3
SOURCES = {k: v for k, v in SOURCES.items() if os.path.isdir(v) and os.path.exists(f"{v}/task00.npz")}
REPS = [r for r in REPS if r == "state" or r.rstrip("_state") in SOURCES or r in SOURCES]
REPS = [r for r in REPS if r == "state" or (r[:-6] if r.endswith("_state") else r) in SOURCES]
DEVICE = "cuda"


def load(ti: int) -> dict[str, np.ndarray]:
    out = {}
    for name, path in SOURCES.items():
        z = np.load(os.path.join(path, f"task{ti:02d}.npz"))
        out[name] = z["latent"].astype(np.float32)
        if "action" not in out:
            out["action"], out["state"], out["demo"] = z["action"], z["state"], z["demo"]
        else:  # every cache must describe the same frames in the same order
            assert np.array_equal(out["demo"], z["demo"]) and out["action"].shape == z["action"].shape
    return out


def features(d: dict[str, np.ndarray], rep: str) -> np.ndarray:
    if rep == "state":
        return d["state"]
    if rep.endswith("_state"):
        return np.concatenate([d[rep[:-6]], d["state"]], 1)
    return d[rep]


def probe(xtr, ytr, xev, yev, seed: int) -> float:
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((xtr - mu) / sd, device=DEVICE); yt = torch.tensor(ytr, device=DEVICE)
    xe = torch.tensor((xev - mu) / sd, device=DEVICE); ye = torch.tensor(yev, device=DEVICE)
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, yt.shape[1])).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xt), device=DEVICE)
        for i in range(0, len(perm), BATCH):
            j = perm[i:i + BATCH]
            loss = F.mse_loss(net(xt[j]), yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return float(F.mse_loss(net(xe), ye))


def summarise(pt: dict[str, np.ndarray], refs: list[str]) -> wandb.Table:
    dims = {r: (load(0)[r[:-6]].shape[1] + 21 if r.endswith("_state")
                else 21 if r == "state" else load(0)[r].shape[1]) for r in REPS}
    tbl = wandb.Table(columns=["rep", "dims", "mse"] + [c for r in refs for c in (f"vs_{r}_%", f"p_{r}")])
    for r in REPS:
        row = [r, dims[r], round(float(pt[r].mean()), 5)]
        for ref in refs:
            row += ([None, None] if r == ref else
                    [round(100 * (pt[r].mean() / pt[ref].mean() - 1), 2),
                     round(float(stats.ttest_rel(pt[ref], pt[r])[1]), 5)])
        tbl.add_data(*row)
        wandb.summary[f"mse/{r}"] = float(pt[r].mean())
        print("== %-19s dims=%5d  MSE %.4f   vs state %+7.1f%%" % (
            r, dims[r], pt[r].mean(), 100 * (pt[r].mean() / pt["state"].mean() - 1)))
    return tbl


def report_pairs(pt: dict[str, np.ndarray]) -> None:
    candidates = [(r, "state") for r in REPS if r != "state"]
    candidates += [("siglip2", "vae"), ("openvla", "vae"), ("openvla", "dit_ctx1"), ("siglip2", "dit_ctx1"),
                   ("openvla", "siglip2"), ("openvla_ctx8", "siglip2_ctx8"),
                   ("openvla_ctx8", "dit_ctx8"), ("siglip2_ctx8", "dit_ctx8"),
                   ("openvla_ctx8", "openvla"), ("siglip2_ctx8", "siglip2")]
    for a, b in [(a, b) for a, b in candidates if a in REPS and b in REPS]:
        p_val = float(stats.ttest_rel(pt[b], pt[a])[1])
        wandb.summary[f"p/{a}_vs_{b}"] = p_val
        print("   %-19s vs %-12s %+7.1f%%  p=%.4f  %d/%d tasks" % (
            a, b, 100 * (pt[a].mean() / pt[b].mean() - 1), p_val,
            int((pt[a] < pt[b]).sum()), N_TASKS))


def main() -> None:
    run = wandb.init(project="video-wam", job_type="compare", name="compare-backbone-audit",
                     config=dict(suite="spatial", n_tasks=N_TASKS, n_seeds=N_SEEDS, reps=REPS,
                                 eval_demos_per_task=N_EVAL_DEMOS, epochs=EPOCHS, renders_frames=False))
    err = {r: np.zeros((N_TASKS, N_SEEDS)) for r in REPS}
    for ti in range(N_TASKS):
        d = load(ti)
        held = set(np.unique(d["demo"])[-N_EVAL_DEMOS:].tolist())
        ev = np.array([x in held for x in d["demo"]]); tr = ~ev
        for rep in REPS:
            x = features(d, rep)
            for s in range(N_SEEDS):
                err[rep][ti, s] = probe(x[tr], d["action"][tr], x[ev], d["action"][ev], s)
            wandb.log({"task": ti, f"heldout_mse/{rep}": err[rep][ti].mean()})
        print("task %d  " % ti + "  ".join("%s=%.4f" % (r, err[r][ti].mean()) for r in REPS), flush=True)

    pt = {r: err[r].mean(1) for r in REPS}
    print()
    tbl = summarise(pt, [x for x in ("state", "dit_ctx1", "vae") if x in REPS])
    report_pairs(pt)
    wandb.log({"results": tbl})
    np.savez(f"{ROOT}/compare_results.npz", **err)
    run.finish()


if __name__ == "__main__":
    main()
