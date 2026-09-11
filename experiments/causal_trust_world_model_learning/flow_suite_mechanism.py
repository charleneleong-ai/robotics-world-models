#!/usr/bin/env python3
"""Why does FLOW help on LIBERO-Spatial and not on Object/Goal or across suites?

For each condition and seed: train plain BC on three tasks, then on the next task's
samples measure, under the previous policy theta*,
  l_i      per-sample BC loss (what FLOW weights by),
  s_i      interference score = -cos(grad_i l_i, grad L_old): positive means a step on
           sample i raises the old tasks' loss,
and report Spearman(l, s), the share of destructive samples in the top- vs bottom-weight
quartiles, and an outcome check: fine-tune 5 epochs on the top-weight, bottom-weight and
a random quartile and measure the change in old-task error and new-task error.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "temp"]
import flow_robustness as fr  # noqa: E402
from continualwam import eval_bc, load_demos, train_bc  # noqa: E402

DEVICE = fr.fw.DEVICE
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_per_param_consolidation", "flow_suite_mechanism.json")
N_SEEDS, PROBE_EPOCHS = 3, 5


def make_policy(od: int, ad: int) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(od, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, ad)).to(DEVICE)


def flat_grad(policy, loss: torch.Tensor) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, list(policy.parameters()), retain_graph=False)])


def per_sample_grads(policy, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    out = []
    for i in range(obs.shape[0]):
        loss = F.mse_loss(policy(obs[i:i + 1]), act[i:i + 1])
        out.append(flat_grad(policy, loss).detach())
    return torch.stack(out)


def probe_subset(policy, obs, act, idx, old_tasks, new_task) -> dict[str, float]:
    pol = copy.deepcopy(policy)
    old_before = float(np.mean([eval_bc(pol, t, DEVICE) for t in old_tasks])); new_before = eval_bc(pol, new_task, DEVICE)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3); pol.train()
    o, a = obs[idx], act[idx]
    for _ in range(PROBE_EPOCHS):
        perm = torch.randperm(o.shape[0], device=DEVICE)
        for s in range(0, o.shape[0], 64):
            b = perm[s:s + 64]
            if len(b) < 2:
                continue
            loss = F.mse_loss(pol(o[b]), a[b]); opt.zero_grad(); loss.backward(); opt.step()
    old_after = float(np.mean([eval_bc(pol, t, DEVICE) for t in old_tasks])); new_after = eval_bc(pol, new_task, DEVICE)
    return {"old_error_change": old_after - old_before, "new_error_change": new_after - new_before}


def analyse(train_tasks, next_task, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    od, ad = train_tasks[0][0]["obs"].shape[-1], train_tasks[0][0]["acts"].shape[-1]
    pol = make_policy(od, ad)
    errs_after_each = []
    for t in train_tasks:
        train_bc(pol, t, fr.m.BC_EPOCHS, device=DEVICE)
        errs_after_each.append(float(np.mean([eval_bc(pol, u, DEVICE) for u in train_tasks[:len(errs_after_each) + 1]])))
    pol.eval()
    obs, act, _ = fr.fw.transitions(next_task)
    with torch.no_grad():
        loss = F.mse_loss(pol(obs), act, reduction="none").mean(dim=-1)
    w = torch.exp(-loss / loss.median())
    old_obs, old_act, _ = fr.fw.transitions([d for t in train_tasks for d in t])
    g_old = flat_grad(pol, F.mse_loss(pol(old_obs), old_act)).detach()
    g_new = per_sample_grads(pol, obs, act)
    cos = F.cosine_similarity(g_new, g_old.unsqueeze(0), dim=1)
    interference = (-cos).cpu().numpy(); lo = loss.cpu().numpy(); wn = w.cpu().numpy()
    rho, p = stats.spearmanr(lo, interference)
    q_hi, q_lo = np.argsort(-wn)[: len(wn) // 4], np.argsort(wn)[: len(wn) // 4]
    rng = np.random.default_rng(seed); q_rand = rng.choice(len(wn), len(wn) // 4, replace=False)
    out = {"n_samples": int(len(wn)), "prev_loss_median": float(np.median(lo)), "loss_iqr_ratio": float(np.percentile(lo, 75) / np.percentile(lo, 25)),
           "spearman_loss_vs_interference": float(rho), "spearman_p": float(p),
           "mean_interference": float(interference.mean()), "frac_destructive": float((interference > 0).mean()),
           "frac_destructive_top_weight_quartile": float((interference[q_hi] > 0).mean()),
           "frac_destructive_bottom_weight_quartile": float((interference[q_lo] > 0).mean()),
           "old_error_after_3_tasks": errs_after_each[-1]}
    for name, idx in (("top_weight", q_hi), ("bottom_weight", q_lo), ("random", q_rand)):
        out[f"probe_{name}"] = probe_subset(pol, obs, act, torch.tensor(idx, device=DEVICE), train_tasks, next_task)
    return out


def main() -> None:
    S = fr.SUITE_DIRS
    spa, obj, goal = load_demos(S["spatial"], 4, 5), load_demos(S["object"], 4, 5), load_demos(S["goal"], 4, 5)
    conditions = {"spatial": (spa[:3], spa[3]), "object": (obj[:3], obj[3]), "goal": (goal[:3], goal[3]),
                  "cross_object_to_spatial": (obj[:3], spa[0]), "cross_spatial_to_goal": (spa[:3], goal[0])}
    results = {}
    for name, (train, nxt) in conditions.items():
        runs = [analyse(train, nxt, seed) for seed in range(N_SEEDS)]
        mean = {}
        for k in runs[0]:
            if isinstance(runs[0][k], dict):
                mean[k] = {kk: float(np.mean([r[k][kk] for r in runs])) for kk in runs[0][k]}
            else:
                mean[k] = float(np.mean([r[k] for r in runs]))
        results[name] = {"per_seed": runs, "mean": mean}
        print(f"== {name}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in mean.items()}), flush=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
