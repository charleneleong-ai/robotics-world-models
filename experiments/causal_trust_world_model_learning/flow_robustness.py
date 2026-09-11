#!/usr/bin/env python3
"""Robustness sweep for FLOW self-loss weighting (task-order protocol, paired arms).

  temp         FLOW temperature tau = median * {0.25, 0.5, 1, 2, 4}      (Spatial, 6 x 3)
  suites       LIBERO-Object and LIBERO-Goal, none / flow                (6 x 3 each)
  lambda       EWC lambda in {10, 100, 1000}, ewc vs flow_ewc            (Spatial, 6 x 3)
  scale        10 fresh orderings x 3 seeds, none/flow/ewc/flow_ewc      (Spatial)
  cross_suite  Object -> Spatial -> Goal, 3 tasks each, fixed order, 5 seeds
  demos        20 demos per task instead of 5, none/flow/ewc/flow_ewc    (Spatial, 3 x 3)
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
from task_order_sensitivity import SUITE_DIRS

EXP = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_per_param_consolidation", f"flow_robustness_{EXP}.json")


@torch.no_grad()
def flow_weights(policy, obs, act, tau_mult: float) -> torch.Tensor:
    loss = F.mse_loss(policy(obs), act, reduction="none").mean(dim=-1)
    return torch.exp(-loss / (tau_mult * loss.median().clamp_min(1e-8)))


def train_step_loop(policy, demos, weights, anchors, lam, epochs=m.BC_EPOCHS, batch_size=64, lr=1e-3) -> None:
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


def run(arm: str, demos, order, seed, tau_mult: float = 1.0, lam: float = 100.0) -> list[float]:
    """arm in {none, flow, ewc, flow_ewc}; tau_mult scales FLOW's temperature; lam is the EWC weight."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    obs_dim = demos[0][0]["obs"].shape[-1]; act_dim = demos[0][0]["acts"].shape[-1]
    wm = get_backbone("mlp", obs_dim, act_dim).to(fw.DEVICE)
    policy = torch.nn.Sequential(torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                                 torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                 torch.nn.Linear(128, act_dim)).to(fw.DEVICE)
    use_flow, use_ewc = arm.startswith("flow"), "ewc" in arm
    anchors: list[tuple[float, dict, dict]] = []
    errs = []
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
                weights = flow_weights(policy, obs, act, tau_mult)
            train_step_loop(policy, task_demos, weights, anchors if use_ewc else [], lam)
        if use_ewc:
            anchors.append((1.0, m.policy_fisher(policy, task_demos), {n: p.detach().clone() for n, p in policy.named_parameters()}))
        errs.append(float(np.mean([eval_bc(policy, demos[o], fw.DEVICE) for o in order[:i + 1]])))
    return errs


def orderings_for(n_ord: int, n_tasks: int, rseed: int) -> list[list[int]]:
    rng = random.Random(rseed); out = []
    for _ in range(n_ord):
        o = list(range(n_tasks)); rng.shuffle(o); out.append(o)
    return out


def summarise(per_arm: dict[str, list[list[float]]], n_ord: int, n_seed: int, refs=("none", "ewc"), by_seed: bool = False) -> dict:
    """Pairs by ordering (seeds averaged within ordering) unless by_seed, for single-ordering designs."""
    bo = (lambda x: np.array(x)) if by_seed else (lambda x: np.array(x).reshape(n_ord, n_seed).mean(1))
    met = {a: {"seq": bo(np.array(r)[:, 1:].mean(1)), "final": bo(np.array(r)[:, -1])} for a, r in per_arm.items()}
    cell = {a: np.array(r)[:, 1:].mean(1) for a, r in per_arm.items()}
    out = {}
    for a, mm in met.items():
        row = {k: float(v.mean()) for k, v in mm.items()}
        for ref in refs:
            if ref in met and a != ref:
                row[f"seq_delta_vs_{ref}"] = float(100 * (mm["seq"].mean() / met[ref]["seq"].mean() - 1))
                row[f"p_seq_vs_{ref}"] = float(stats.ttest_rel(met[ref]["seq"], mm["seq"])[1])
                row[f"p_final_vs_{ref}"] = float(stats.ttest_rel(met[ref]["final"], mm["final"])[1])
                row[f"cells_better_vs_{ref}"] = int((cell[a] < cell[ref]).sum())
        out[a] = row
    return out


def sweep(label: str, demos, orderings, n_seed: int, arms: list[tuple[str, dict]], refs=("none", "ewc")) -> dict:
    per_arm = {name: [] for name, _ in arms}
    for oi, order in enumerate(orderings):
        for seed in range(n_seed):
            for name, kw in arms:
                errs = run(kw.get("arm", name), demos, order, seed, kw.get("tau_mult", 1.0), kw.get("lam", 100.0))
                per_arm[name].append(errs)
                print(f"[{label}] ord={oi} seed={seed} arm={name:14s} first={errs[0]:.4f} final={errs[-1]:.4f}", flush=True)
    summary = summarise(per_arm, len(orderings), n_seed, refs, by_seed=(len(orderings) == 1))
    print(f"== {label}: " + json.dumps({a: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for a, r in summary.items()}), flush=True)
    return {"summary": summary, "raw": per_arm, "orderings": orderings}


def main() -> None:
    results = {}
    if EXP == "temp":
        demos = load_demos(SUITE_DIRS["spatial"], 10, 5)
        arms = [("none", {})] + [(f"flow_tau{t}", {"arm": "flow", "tau_mult": t}) for t in (0.25, 0.5, 1.0, 2.0, 4.0)]
        results["temp"] = sweep("temp", demos, orderings_for(6, 10, 0), 3, arms, refs=("none",))
    elif EXP == "suites":
        for suite in ("object", "goal"):
            demos = load_demos(SUITE_DIRS[suite], 10, 5)
            results[suite] = sweep(suite, demos, orderings_for(6, 10, 0), 3, [("none", {}), ("flow", {})], refs=("none",))
            json.dump(results, open(OUT, "w"), indent=2)
    elif EXP == "lambda":
        demos = load_demos(SUITE_DIRS["spatial"], 10, 5)
        arms = [("none", {})]
        for lam in (10.0, 100.0, 1000.0):
            arms += [(f"ewc_lam{int(lam)}", {"arm": "ewc", "lam": lam}), (f"flow_ewc_lam{int(lam)}", {"arm": "flow_ewc", "lam": lam})]
        results["lambda"] = sweep("lambda", demos, orderings_for(6, 10, 0), 3, arms, refs=("none",))
        r = results["lambda"]["raw"]
        for lam in (10, 100, 1000):
            a, b = np.array(r[f"ewc_lam{lam}"])[:, 1:].mean(1).reshape(6, 3).mean(1), np.array(r[f"flow_ewc_lam{lam}"])[:, 1:].mean(1).reshape(6, 3).mean(1)
            results["lambda"]["summary"][f"flow_ewc_lam{lam}"]["p_seq_vs_ewc_same_lam"] = float(stats.ttest_rel(a, b)[1])
            results["lambda"]["summary"][f"flow_ewc_lam{lam}"]["seq_delta_vs_ewc_same_lam"] = float(100 * (b.mean() / a.mean() - 1))
        print("== lambda vs same-lambda ewc: " + json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if "same_lam" in kk} for k, v in results["lambda"]["summary"].items() if "same_lam" in json.dumps(v)}), flush=True)
    elif EXP == "scale":
        demos = load_demos(SUITE_DIRS["spatial"], 10, 5)
        results["scale"] = sweep("scale", demos, orderings_for(10, 10, 3), 3, [("none", {}), ("flow", {}), ("ewc", {}), ("flow_ewc", {})])
    elif EXP == "cross_suite":
        demos = load_demos(SUITE_DIRS["object"], 3, 5) + load_demos(SUITE_DIRS["spatial"], 3, 5) + load_demos(SUITE_DIRS["goal"], 3, 5)
        results["cross_suite"] = sweep("cross_suite", demos, [list(range(9))], 5, [("none", {}), ("flow", {}), ("ewc", {}), ("flow_ewc", {})])
    elif EXP == "demos":
        demos = load_demos(SUITE_DIRS["spatial"], 10, 20)
        results["demos"] = sweep("demos", demos, orderings_for(3, 10, 0), 3, [("none", {}), ("flow", {}), ("ewc", {}), ("flow_ewc", {})])
    else:
        raise SystemExit(f"unknown experiment {EXP}")
    json.dump(results, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
