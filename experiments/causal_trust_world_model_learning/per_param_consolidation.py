#!/usr/bin/env python3
"""Experiments A and C: the per-parameter coupling the paper's theory describes,
which the task-order harness never actually ran.

task_order_sensitivity.train_sequential has no consolidation at all: per task it
trains the world model with plain MSE, then the BC policy with a *per-sample*
trust-weighted regression loss. Result 2's effect therefore came from per-sample
loss weighting -- the coupling that hurt on classification -- not from
precision_k = tau * F_k as written in Section 3.

A (policy):  arms none | per_sample (existing) | ewc (tau=1) | trust_ewc (tau_task * F_k)
             on the BC policy; metric = forgetting rate and std of final error across
             orderings (order-robustness), Result 2's own protocol.
C (world model): arms none | ewc | trust_ewc on the world model's OWN parameters, tau from
             its own prediction error; metric = next-obs prediction error on all seen tasks
             (world-model forgetting), i.e. the objective where the only positive appeared.

Same orderings and seeds across arms so comparisons are paired.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continualwam import get_backbone, load_demos, make_trust, train_bc, train_bc_trust, eval_bc
from task_order_sensitivity import SUITE_DIRS, compute_forgetting

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUITE = "spatial"
N_TASKS = 10
N_ORDERINGS = 6
N_SEEDS = 3
BACKBONES = ["mlp", "rssm"]
WM_EPOCHS = 20
BC_EPOCHS = 50
EWC_LAMBDA_POLICY = 100.0
EWC_LAMBDA_WM = 100.0
RESULTS_DIR = Path(__file__).parent / "results_per_param_consolidation"
RESULTS_DIR.mkdir(exist_ok=True)


def transitions(demos_task: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = np.concatenate([d["obs"][:-1] for d in demos_task])
    act = np.concatenate([d["acts"][:-1] for d in demos_task])
    nxt = np.concatenate([d["obs"][1:] for d in demos_task])
    t = lambda x: torch.tensor(x, dtype=torch.float32).to(DEVICE)
    return t(obs), t(act), t(nxt)


def task_trust(wm, scorer, demos_task: list[dict]) -> float:
    obs, act, nxt = transitions(demos_task)
    wm.eval()
    with torch.no_grad():
        pe = wm.predict_error(obs, act, nxt)
    tr = scorer.compute_trust(pe, obs=obs, act=act, next_obs=nxt)
    return float(tr.mean().item())


def diag_fisher(model, loss_batches) -> dict[str, torch.Tensor]:
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    n = 0
    for loss in loss_batches:
        model.zero_grad()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
        n += 1
    return {k: v / max(n, 1) for k, v in fisher.items()}


def ewc_penalty(model, anchors: list[tuple[float, dict, dict]], lam: float) -> torch.Tensor:
    pen = torch.tensor(0.0, device=DEVICE)
    for tau, fisher, theta_star in anchors:
        for name, p in model.named_parameters():
            pen = pen + tau * (fisher[name] * (p - theta_star[name]) ** 2).sum()
    return lam / 2 * pen


# ----------------------------------------------------------------------------- A

def train_bc_ewc(policy, demos_task, anchors, lam, epochs=BC_EPOCHS, bs=64, lr=1e-3):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    obs = torch.tensor(np.concatenate([d["obs"] for d in demos_task]), dtype=torch.float32).to(DEVICE)
    act = torch.tensor(np.concatenate([d["acts"] for d in demos_task]), dtype=torch.float32).to(DEVICE)
    policy.train()
    for _ in range(epochs):
        perm = torch.randperm(obs.size(0))
        for i in range(0, obs.size(0), bs):
            idx = perm[i:i + bs]
            loss = F.mse_loss(policy(obs[idx]), act[idx]) + ewc_penalty(policy, anchors, lam)
            opt.zero_grad()
            loss.backward()
            opt.step()


def policy_fisher(policy, demos_task, bs=64):
    obs = torch.tensor(np.concatenate([d["obs"] for d in demos_task]), dtype=torch.float32).to(DEVICE)
    act = torch.tensor(np.concatenate([d["acts"] for d in demos_task]), dtype=torch.float32).to(DEVICE)
    policy.train()
    def batches():
        for i in range(0, obs.size(0), bs):
            yield F.mse_loss(policy(obs[i:i + bs]), act[i:i + bs])
    return diag_fisher(policy, batches())


def run_A(backbone: str, arm: str, demos, order, seed) -> list[float]:
    torch.manual_seed(seed); np.random.seed(seed)
    obs_dim = demos[0][0]["obs"].shape[-1]; act_dim = demos[0][0]["acts"].shape[-1]
    wm = get_backbone(backbone, obs_dim, act_dim).to(DEVICE)
    policy = torch.nn.Sequential(torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                                 torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                 torch.nn.Linear(128, act_dim)).to(DEVICE)
    scorer = make_trust("ema", obs_dim, act_dim)
    if isinstance(scorer, torch.nn.Module):
        scorer = scorer.to(DEVICE)
    anchors: list[tuple[float, dict, dict]] = []
    errors_per_task = []
    for i, t in enumerate(order):
        task_demos = demos[t]
        from continualwam.training import train_wm
        train_wm(wm, task_demos, WM_EPOCHS, device=DEVICE)
        if arm == "none":
            train_bc(policy, task_demos, BC_EPOCHS, device=DEVICE)
        elif arm == "per_sample":
            train_bc_trust(policy, wm, scorer, task_demos, BC_EPOCHS, device=DEVICE)
        else:
            train_bc_ewc(policy, task_demos, anchors, EWC_LAMBDA_POLICY)
            tau = 1.0 if arm == "ewc" else task_trust(wm, scorer, task_demos)
            anchors.append((tau, policy_fisher(policy, task_demos),
                            {n: p.detach().clone() for n, p in policy.named_parameters()}))
        errs = [eval_bc(policy, demos[o], DEVICE) for o in order[:i + 1]]
        errors_per_task.append(float(np.mean(errs)))
    return errors_per_task


# ----------------------------------------------------------------------------- C

def train_wm_ewc(wm, demos_task, anchors, lam, epochs=WM_EPOCHS, lr=1e-3):
    opt = torch.optim.Adam(wm.parameters(), lr=lr)
    wm.train()
    for _ in range(epochs):
        for d in demos_task:
            o = torch.tensor(d["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a = torch.tensor(d["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            loss = wm.train_loss(o, a) + ewc_penalty(wm, anchors, lam)
            opt.zero_grad()
            loss.backward()
            opt.step()


def wm_fisher(wm, demos_task):
    wm.train()
    def batches():
        for d in demos_task:
            o = torch.tensor(d["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a = torch.tensor(d["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            yield wm.train_loss(o, a)
    return diag_fisher(wm, batches())


def wm_error(wm, demos_task) -> float:
    obs, act, nxt = transitions(demos_task)
    wm.eval()
    with torch.no_grad():
        return float(wm.predict_error(obs, act, nxt).mean().item())


def run_C(backbone: str, arm: str, demos, order, seed) -> list[float]:
    torch.manual_seed(seed); np.random.seed(seed)
    obs_dim = demos[0][0]["obs"].shape[-1]; act_dim = demos[0][0]["acts"].shape[-1]
    wm = get_backbone(backbone, obs_dim, act_dim).to(DEVICE)
    scorer = make_trust("ema", obs_dim, act_dim)
    if isinstance(scorer, torch.nn.Module):
        scorer = scorer.to(DEVICE)
    anchors: list[tuple[float, dict, dict]] = []
    errors_per_task = []
    for i, t in enumerate(order):
        task_demos = demos[t]
        if arm == "none":
            from continualwam.training import train_wm
            train_wm(wm, task_demos, WM_EPOCHS, device=DEVICE)
        else:
            train_wm_ewc(wm, task_demos, anchors, EWC_LAMBDA_WM)
            tau = 1.0 if arm == "ewc" else task_trust(wm, scorer, task_demos)
            anchors.append((tau, wm_fisher(wm, task_demos),
                            {n: p.detach().clone() for n, p in wm.named_parameters()}))
        errs = [wm_error(wm, demos[o]) for o in order[:i + 1]]
        errors_per_task.append(float(np.mean(errs)))
    return errors_per_task


# ----------------------------------------------------------------------------- driver

def summarise(per_arm: dict[str, list[list[float]]]) -> dict:
    out = {}
    base = per_arm["none"]
    for arm, runs in per_arm.items():
        m = compute_forgetting(runs)
        entry = {"forgetting_mean": m["mean_forgetting_rate"], "forgetting_std": m["std_forgetting_rate"],
                 "final_error_mean": m["mean_final_error"], "final_error_std_across_orderings": m["std_final_error"]}
        if arm != "none":
            fb = compute_forgetting(base)["per_ordering_forgetting"]
            t, p = stats.ttest_rel(fb, m["per_ordering_forgetting"])
            entry["p_vs_none_paired"] = float(p)
        out[arm] = entry
    return out


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    demos = load_demos(SUITE_DIRS[SUITE], N_TASKS, 5)
    rng = random.Random(0)
    orderings = []
    for _ in range(N_ORDERINGS):
        o = list(range(N_TASKS)); rng.shuffle(o); orderings.append(o)
    arms = ["none", "per_sample", "ewc", "trust_ewc"] if which == "A" else ["none", "ewc", "trust_ewc"]
    runner = run_A if which == "A" else run_C
    results = {}
    for backbone in BACKBONES:
        per_arm = {a: [] for a in arms}
        for oi, order in enumerate(orderings):
            for seed in range(N_SEEDS):
                for arm in arms:
                    errs = runner(backbone, arm, demos, order, seed)
                    per_arm[arm].append(errs)
                    print(f"[{which}] {backbone} ord={oi} seed={seed} arm={arm:10s} final={errs[-1]:.4f}", flush=True)
        results[backbone] = summarise(per_arm)
        results[backbone]["raw"] = per_arm
        print(f"== [{which}] {backbone}: " + json.dumps({a: {k: round(v, 4) for k, v in d.items()} for a, d in results[backbone].items() if a != 'raw'}), flush=True)
    with open(RESULTS_DIR / f"results_{which}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
