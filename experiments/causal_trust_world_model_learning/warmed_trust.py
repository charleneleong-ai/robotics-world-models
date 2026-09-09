#!/usr/bin/env python3
"""Per-sample trust weighting with corrected and warmed-state prediction error.

`train_bc_trust` scores each transition with `predict_error(obs, act, obs)`, i.e.
against the *current* observation, from a zero recurrent state. This re-runs the
task-order protocol (experiment A's orderings/seeds) with three trust variants,
all paired against no trust:

  none      plain BC
  ps_asis   train_bc_trust as published (next_obs := obs, zero state)
  ps_true   true next observation, zero recurrent state
  ps_warm   true next observation, recurrent state carried over the demo
            (identical to ps_true for the feedforward MLP, so skipped there)

Diagnostics on the first task of ordering 0 / seed 0 record how the three error
signals relate (spread, correlation) and what trust spread each induces.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import per_param_consolidation as m
from continualwam import eval_bc, get_backbone, load_demos, make_trust, train_bc, train_bc_trust
from continualwam.training import train_wm
from task_order_sensitivity import SUITE_DIRS, compute_forgetting

DEVICE = m.DEVICE
N_ORD, N_SEED = 6, 3
ALL_BACKBONES = {"mlp": ["none", "ps_asis", "ps_true"],
                 "jepa": ["none", "ps_asis", "ps_true"],
                 "rssm": ["none", "ps_asis", "ps_true", "ps_warm"],
                 "dreamerv3": ["none", "ps_asis", "ps_true", "ps_warm"]}
SELECTED = sys.argv[1:] or list(ALL_BACKBONES)
BACKBONES = {b: ALL_BACKBONES[b] for b in SELECTED}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_per_param_consolidation",
                   f"warmed_trust_{'_'.join(SELECTED)}.json")


def to_t(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def seq_errors(wm: torch.nn.Module, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """Per-step next-obs error over one demo with the recurrent state carried (mirrors train_loss)."""
    wm.eval()
    T = obs.shape[0]
    h = torch.zeros(1, wm.state_dim, device=DEVICE)
    dreamer = hasattr(wm, "stoch_encoder")
    errs = []
    for t in range(T - 1):
        o_enc = torch.relu(wm.obs_encoder(obs[t:t + 1]))
        h = torch.relu(wm.rnn(torch.cat([o_enc, act[t:t + 1]], dim=-1), h))
        if dreamer:
            s = torch.relu(wm.stoch_encoder(h))
            pred = wm.pred_head(torch.cat([h, s], dim=-1))
        else:
            pred = wm.pred_head(h)
        errs.append(F.mse_loss(pred, obs[t + 1:t + 2], reduction="none").mean(dim=-1))
    return torch.cat(errs)


@torch.no_grad()
def transition_errors(wm: torch.nn.Module, demos: list[dict], mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = to_t(np.concatenate([d["obs"][:-1] for d in demos]))
    act = to_t(np.concatenate([d["acts"][:-1] for d in demos]))
    nxt = to_t(np.concatenate([d["obs"][1:] for d in demos]))
    wm.eval()
    if mode == "warm":
        pe = torch.cat([seq_errors(wm, to_t(d["obs"]), to_t(d["acts"])) for d in demos])
    elif mode == "true":
        pe = wm.predict_error(obs, act, nxt)
    else:  # as published
        pe = wm.predict_error(obs, act, obs)
    return obs, act, pe


def train_bc_variant(policy, wm, scorer, demos, mode: str, epochs: int = m.BC_EPOCHS, batch_size: int = 64, lr: float = 1e-3) -> None:
    """train_bc_trust with the error signal precomputed under `mode` (wm is frozen during BC)."""
    obs_t, act_t, pe_all = transition_errors(wm, demos, mode)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    for _ in range(epochs):
        perm = torch.randperm(obs_t.size(0), device=DEVICE)
        for i in range(0, obs_t.size(0), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            with torch.no_grad():
                trust = scorer.compute_trust(pe_all[idx], obs=obs_t[idx], act=act_t[idx])
                w = trust.clamp(0.1, 1.0)
            loss = (F.mse_loss(policy(obs_t[idx]), act_t[idx], reduction="none").mean(dim=-1) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def diagnostics(wm, scorer_factory, demos) -> dict:
    out = {}
    pes = {}
    modes = ["asis", "true"] + (["warm"] if hasattr(wm, "rnn") else [])
    for mode in modes:
        obs, act, pe = transition_errors(wm, demos, mode)
        pes[mode] = pe.cpu().numpy()
        tr = scorer_factory().compute_trust(pe, obs=obs, act=act).cpu().numpy()
        out[mode] = {"pe_mean": float(pe.mean()), "pe_std": float(pe.std()), "pe_cv": float(pe.std() / (pe.mean() + 1e-12)),
                     "trust_mean": float(tr.mean()), "trust_std": float(tr.std()), "trust_min": float(tr.min()), "trust_max": float(tr.max())}
    for a, b in [("asis", "true"), ("true", "warm")]:
        if a in pes and b in pes:
            out[f"corr_{a}_{b}"] = float(np.corrcoef(pes[a], pes[b])[0, 1])
    return out


def run(backbone: str, arm: str, demos, order, seed) -> list[float]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    obs_dim = demos[0][0]["obs"].shape[-1]; act_dim = demos[0][0]["acts"].shape[-1]
    wm = get_backbone(backbone, obs_dim, act_dim).to(DEVICE)
    policy = torch.nn.Sequential(torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                                 torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                 torch.nn.Linear(128, act_dim)).to(DEVICE)
    scorer = make_trust("ema", obs_dim, act_dim)
    errors_per_task = []
    for i, t in enumerate(order):
        task_demos = demos[t]
        train_wm(wm, task_demos, m.WM_EPOCHS, device=DEVICE)
        if arm == "none":
            train_bc(policy, task_demos, m.BC_EPOCHS, device=DEVICE)
        elif arm == "ps_asis":
            train_bc_trust(policy, wm, scorer, task_demos, m.BC_EPOCHS, device=DEVICE)
        else:
            train_bc_variant(policy, wm, scorer, task_demos, arm.split("_")[1])
        errors_per_task.append(float(np.mean([eval_bc(policy, demos[o], DEVICE) for o in order[:i + 1]])))
    return errors_per_task


def by_ordering(vals) -> np.ndarray:
    return np.array(vals).reshape(N_ORD, N_SEED).mean(1)


def summarise(per_arm: dict[str, list[list[float]]]) -> dict:
    metrics = {}
    for arm, runs in per_arm.items():
        e = np.array(runs)
        metrics[arm] = {"first": by_ordering(e[:, 0]), "final": by_ordering(e[:, -1]),
                        "abs_forget": by_ordering(e[:, -1] - e[:, 0]),
                        "rel_forget": by_ordering(compute_forgetting(runs)["per_ordering_forgetting"])}
    out = {}
    for arm, mm in metrics.items():
        row = {k: float(v.mean()) for k, v in mm.items()}
        if arm != "none":
            for k in mm:
                row[f"p_{k}"] = float(stats.ttest_rel(metrics["none"][k], mm[k])[1])
        out[arm] = row
    return out


def main() -> None:
    demos = load_demos(SUITE_DIRS["spatial"], 10, 5)
    rng = random.Random(0)
    orderings = []
    for _ in range(N_ORD):
        o = list(range(10)); rng.shuffle(o); orderings.append(o)
    obs_dim = demos[0][0]["obs"].shape[-1]; act_dim = demos[0][0]["acts"].shape[-1]
    results = {"orderings": orderings}
    for backbone, arms in BACKBONES.items():
        torch.manual_seed(0)
        wm = get_backbone(backbone, obs_dim, act_dim).to(DEVICE)
        train_wm(wm, demos[orderings[0][0]], m.WM_EPOCHS, device=DEVICE)
        diag = diagnostics(wm, lambda: make_trust("ema", obs_dim, act_dim), demos[orderings[0][0]])
        print(f"[diag] {backbone}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in diag.items()}), flush=True)
        per_arm = {a: [] for a in arms}
        for oi, order in enumerate(orderings):
            for seed in range(N_SEED):
                for arm in arms:
                    errs = run(backbone, arm, demos, order, seed)
                    per_arm[arm].append(errs)
                    print(f"[{backbone}] ord={oi} seed={seed} arm={arm:8s} first={errs[0]:.4f} final={errs[-1]:.4f}", flush=True)
        summary = summarise(per_arm)
        print(f"== {backbone}: " + json.dumps({a: {k: round(v, 4) for k, v in r.items()} for a, r in summary.items()}), flush=True)
        results[backbone] = {"diagnostics": diag, "summary": summary, "raw": per_arm}
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
