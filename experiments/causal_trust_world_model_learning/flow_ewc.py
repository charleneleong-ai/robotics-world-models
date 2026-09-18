#!/usr/bin/env python3
"""FLOW self-loss weighting combined with EWC on the task-order protocol
(experiment A's orderings, 6 x 3, all arms paired).

  none          plain BC
  flow          FLOW weights, no consolidation
  ewc           plain EWC on the policy (lambda = EWC_LAMBDA_POLICY), unweighted loss
  flow_ewc      FLOW-weighted loss + EWC penalty
  flow_ewc_norm as flow_ewc with weights rescaled to mean 1, so the data term keeps
                its magnitude against the penalty (FLOW's raw mean weight is ~0.4)
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

import flow_weighting as fw
import per_param_consolidation as m
from continualwam import eval_bc, get_backbone, load_demos, train_bc
from continualwam.training import train_wm

ARMS = ["none", "flow", "ewc", "flow_ewc", "flow_ewc_norm"]
SELECTED = sys.argv[1:] or ["mlp"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_per_param_consolidation",
                   f"flow_ewc_{'_'.join(SELECTED)}.json")


@torch.no_grad()
def flow_weights(policy, obs, act, normalise: bool) -> torch.Tensor:
    loss = F.mse_loss(policy(obs), act, reduction="none").mean(dim=-1)
    w = torch.exp(-loss / loss.median().clamp_min(1e-8))
    return w / w.mean() if normalise else w


def train_bc_weighted_ewc(policy, demos, weights: torch.Tensor | None, anchors, lam: float,
                          epochs: int = m.BC_EPOCHS, batch_size: int = 64, lr: float = 1e-3) -> None:
    obs_t, act_t, _ = fw.transitions(demos)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    for _ in range(epochs):
        perm = torch.randperm(obs_t.size(0), device=fw.DEVICE)
        for i in range(0, obs_t.size(0), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            per_sample = F.mse_loss(policy(obs_t[idx]), act_t[idx], reduction="none").mean(dim=-1)
            loss = (per_sample * weights[idx]).mean() if weights is not None else per_sample.mean()
            if anchors:
                loss = loss + m.ewc_penalty(policy, anchors, lam)
            opt.zero_grad(); loss.backward(); opt.step()


def run(backbone: str, arm: str, demos, order, seed) -> list[float]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    obs_dim = demos[0][0]["obs"].shape[-1]; act_dim = demos[0][0]["acts"].shape[-1]
    wm = get_backbone(backbone, obs_dim, act_dim).to(fw.DEVICE)
    policy = torch.nn.Sequential(torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                                 torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                 torch.nn.Linear(128, act_dim)).to(fw.DEVICE)
    use_flow, use_ewc = arm.startswith("flow"), "ewc" in arm
    anchors: list[tuple[float, dict, dict]] = []
    errors_per_task = []
    for i, t in enumerate(order):
        task_demos = demos[t]
        train_wm(wm, task_demos, m.WM_EPOCHS, device=fw.DEVICE)
        if arm == "none" or i == 0:
            train_bc(policy, task_demos, m.BC_EPOCHS, device=fw.DEVICE)
        else:
            weights = None
            if use_flow:
                policy.eval()
                obs, act, _ = fw.transitions(task_demos)
                weights = flow_weights(policy, obs, act, normalise=arm.endswith("norm"))
            train_bc_weighted_ewc(policy, task_demos, weights, anchors if use_ewc else [], m.EWC_LAMBDA_POLICY)
        if use_ewc:
            anchors.append((1.0, m.policy_fisher(policy, task_demos),
                            {n: p.detach().clone() for n, p in policy.named_parameters()}))
        errors_per_task.append(float(np.mean([eval_bc(policy, demos[o], fw.DEVICE) for o in order[:i + 1]])))
    return errors_per_task


def summarise(per_arm: dict[str, list[list[float]]]) -> dict:
    bo = lambda x: np.array(x).reshape(fw.N_ORD, fw.N_SEED).mean(1)
    met = {a: {"seq": bo(np.array(r)[:, 1:].mean(1)), "final": bo(np.array(r)[:, -1])} for a, r in per_arm.items()}
    out = {}
    for a, mm in met.items():
        row = {k: float(v.mean()) for k, v in mm.items()}
        for ref in ("none", "ewc", "flow"):
            if a != ref:
                for k in mm:
                    row[f"p_{k}_vs_{ref}"] = float(stats.ttest_rel(met[ref][k], mm[k])[1])
        out[a] = row
    return out


def main() -> None:
    demos = load_demos(fw.SUITE_DIRS["spatial"], 10, 5)
    rng = random.Random(0)
    orderings = []
    for _ in range(fw.N_ORD):
        o = list(range(10)); rng.shuffle(o); orderings.append(o)
    results = {"orderings": orderings}
    for backbone in SELECTED:
        per_arm = {a: [] for a in ARMS}
        for oi, order in enumerate(orderings):
            for seed in range(fw.N_SEED):
                for arm in ARMS:
                    errs = run(backbone, arm, demos, order, seed)
                    per_arm[arm].append(errs)
                    print(f"[{backbone}] ord={oi} seed={seed} arm={arm:13s} first={errs[0]:.4f} final={errs[-1]:.4f}", flush=True)
        summary = summarise(per_arm)
        print(f"== {backbone}: " + json.dumps({a: {k: round(v, 4) for k, v in r.items()} for a, r in summary.items()}), flush=True)
        results[backbone] = {"summary": summary, "raw": per_arm}
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
