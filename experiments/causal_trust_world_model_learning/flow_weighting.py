#!/usr/bin/env python3
"""Sample-space weighting on the task-order protocol: FLOW vs world-model trust.

FLOW (Sanyal et al., ICML 2025) up-weights samples the *pre-trained* model already
fits well: w_i = exp(-l_i(theta*) / tau), tau = median_i l_i(theta*), weights fixed
for the whole fine-tuning run. In the sequential setting theta* is the policy after
the previous task, so the weights measure how much each new sample would pull the
policy away from what it already knows. Arms (all paired on experiment A's
orderings/seeds; task 0 is always plain BC since there is nothing to protect yet):

  none      plain BC
  flow      FLOW weights from the policy's own BC loss under theta*
  flow_wm   FLOW's functional form applied to the world model's next-obs error
  inv_trust 1 - trust (world model), i.e. the inverse of the paper's coupling
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
from continualwam import eval_bc, get_backbone, load_demos, make_trust, train_bc
from continualwam.training import train_wm
from task_order_sensitivity import SUITE_DIRS, compute_forgetting

DEVICE = m.DEVICE
N_ORD, N_SEED = 6, 3
ARMS = ["none", "flow", "flow_wm", "inv_trust"]
SELECTED = sys.argv[1:] or ["mlp", "rssm"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_per_param_consolidation",
                   f"flow_weighting_{'_'.join(SELECTED)}.json")


def to_t(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def transitions(demos: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (to_t(np.concatenate([d["obs"][:-1] for d in demos])),
            to_t(np.concatenate([d["acts"][:-1] for d in demos])),
            to_t(np.concatenate([d["obs"][1:] for d in demos])))


@torch.no_grad()
def sample_weights(arm: str, policy, wm, scorer, obs, act, nxt) -> torch.Tensor:
    if arm == "flow":
        loss = F.mse_loss(policy(obs), act, reduction="none").mean(dim=-1)
        return torch.exp(-loss / loss.median().clamp_min(1e-8))
    wm.eval()
    pe = wm.predict_error(obs, act, nxt)
    if arm == "flow_wm":
        return torch.exp(-pe / pe.median().clamp_min(1e-8))
    trust = scorer.compute_trust(pe, obs=obs, act=act)
    return (1.0 - trust).clamp(0.1, 1.0)


def train_bc_weighted(policy, demos, weights: torch.Tensor, epochs: int = m.BC_EPOCHS, batch_size: int = 64, lr: float = 1e-3) -> None:
    obs_t, act_t, _ = transitions(demos)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    for _ in range(epochs):
        perm = torch.randperm(obs_t.size(0), device=DEVICE)
        for i in range(0, obs_t.size(0), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            loss = (F.mse_loss(policy(obs_t[idx]), act_t[idx], reduction="none").mean(dim=-1) * weights[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def run(backbone: str, arm: str, demos, order, seed) -> dict:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    obs_dim = demos[0][0]["obs"].shape[-1]; act_dim = demos[0][0]["acts"].shape[-1]
    wm = get_backbone(backbone, obs_dim, act_dim).to(DEVICE)
    policy = torch.nn.Sequential(torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                                 torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                 torch.nn.Linear(128, act_dim)).to(DEVICE)
    scorer = make_trust("ema", obs_dim, act_dim)
    errors_per_task, weight_stats = [], []
    for i, t in enumerate(order):
        task_demos = demos[t]
        train_wm(wm, task_demos, m.WM_EPOCHS, device=DEVICE)
        if arm == "none" or i == 0:
            train_bc(policy, task_demos, m.BC_EPOCHS, device=DEVICE)
        else:
            policy.eval()
            wts = sample_weights(arm, policy, wm, scorer, *transitions(task_demos))
            weight_stats.append({"mean": float(wts.mean()), "std": float(wts.std()), "min": float(wts.min())})
            train_bc_weighted(policy, task_demos, wts)
        errors_per_task.append(float(np.mean([eval_bc(policy, demos[o], DEVICE) for o in order[:i + 1]])))
    return {"errors": errors_per_task, "weights": weight_stats}


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
    results = {"orderings": orderings}
    for backbone in SELECTED:
        per_arm = {a: [] for a in ARMS}
        wstats = {a: [] for a in ARMS}
        for oi, order in enumerate(orderings):
            for seed in range(N_SEED):
                for arm in ARMS:
                    r = run(backbone, arm, demos, order, seed)
                    per_arm[arm].append(r["errors"]); wstats[arm].extend(r["weights"])
                    print(f"[{backbone}] ord={oi} seed={seed} arm={arm:9s} first={r['errors'][0]:.4f} final={r['errors'][-1]:.4f}", flush=True)
        summary = summarise(per_arm)
        wsum = {a: {k: float(np.mean([s[k] for s in v])) for k in ("mean", "std", "min")} for a, v in wstats.items() if v}
        print(f"== {backbone}: " + json.dumps({a: {k: round(v, 4) for k, v in r.items()} for a, r in summary.items()}), flush=True)
        print(f"== {backbone} weights: " + json.dumps({a: {k: round(v, 3) for k, v in d.items()} for a, d in wsum.items()}), flush=True)
        results[backbone] = {"summary": summary, "weight_stats": wsum, "raw": per_arm}
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
