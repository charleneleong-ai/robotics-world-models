#!/usr/bin/env python3
"""Interference-based signals measured on the model being protected, with a small memory.

Task-order protocol of flow_robustness (LIBERO BC, paired arms). A memory of MEM_PER_TASK
random transitions from each finished task is available to the memory arms.

  none        plain BC
  flow        FLOW self-loss weighting (memory-free reference)
  ewc         EWC at the tuned lambda from the lambda curves (3000)
  er          experience replay: each step trains on 64 new + 64 random memory samples
  mir         maximally interfered retrieval (Aljundi et al. 2019): rehearse the 64 of 256
              candidate memory samples whose loss rises most under a virtual step on the new batch
  agem        A-GEM: project the new-batch gradient off the memory gradient when they conflict
  conflict_w  per-sample conflict weighting: s_i = -<g_i, g_mem>/|g_mem| from per-sample
              gradients; w_i = exp(-relu(s_i)/median(relu(s))) (FLOW's form on the conflict)
  ensemble_w  memory-free: N_HEADS policies trained through the sequence; each new sample is
              weighted by exp(-d_i/median d), d_i the heads' action disagreement under the
              previous task's parameters; only head 0 is evaluated
  lookahead   batch weight exp(-relu(dL_old)/running median) where dL_old is the memory-batch
              loss change after a virtual step on the new batch (the outcome, not a proxy)

Usage: interference_signals.py spatial <ordering indices e.g. 0,1> | cross
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODE = sys.argv[1]
ORDS = [int(x) for x in sys.argv[2].split(",")] if MODE == "spatial" else []
sys.argv = [sys.argv[0], "temp"]
import flow_robustness as fr  # noqa: E402
from continualwam import eval_bc, load_demos, train_bc  # noqa: E402

DEVICE = fr.fw.DEVICE
ARMS = ["none", "flow", "ewc100", "flow_ewc100", "ewc", "er", "mir", "agem", "conflict_w", "ensemble_w", "lookahead"]
MEM_PER_TASK, MEM_BATCH, MIR_POOL, EWC_LAM, VIRTUAL_LR, N_HEADS = 100, 64, 256, 3000.0, 1e-3, 5
EPOCHS, BATCH = fr.m.BC_EPOCHS, 64
HELDOUT = os.environ.get("HELDOUT", "0") == "1"   # train on demos[:-N_EVAL] of each task, evaluate on the rest
N_DEMOS, N_EVAL = (10, 2) if HELDOUT else (5, 0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_per_param_consolidation",
                   f"interference_{MODE}{'_heldout' if HELDOUT else ''}{'_ord' + '_'.join(map(str, ORDS)) if ORDS else ''}.json")


def make_policy(od: int, ad: int) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(od, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, ad)).to(DEVICE)


def flat_grad(policy, loss) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, list(policy.parameters()))]).detach()


def per_sample_grads(policy, obs, act) -> torch.Tensor:
    params = {k: v.detach() for k, v in policy.named_parameters()}

    def loss_fn(p, o, a):
        return F.mse_loss(functional_call(policy, p, (o.unsqueeze(0),)), a.unsqueeze(0))

    g = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, obs, act)
    return torch.cat([g[k].reshape(obs.shape[0], -1) for k in params], 1)


def mir_select(policy, obs_new, act_new, mem_obs, mem_act) -> torch.Tensor:
    pool = torch.randperm(mem_obs.shape[0], device=DEVICE)[:MIR_POOL]
    po, pa = mem_obs[pool], mem_act[pool]
    with torch.no_grad():
        before = F.mse_loss(policy(po), pa, reduction="none").mean(-1)
    virt = copy.deepcopy(policy)
    loss = F.mse_loss(virt(obs_new), act_new)
    grads = torch.autograd.grad(loss, list(virt.parameters()))
    with torch.no_grad():
        for p, g in zip(virt.parameters(), grads):
            p -= VIRTUAL_LR * g
        after = F.mse_loss(virt(po), pa, reduction="none").mean(-1)
    top = (after - before).topk(min(MEM_BATCH, len(pool))).indices
    return pool[top]


def old_loss_change(policy, obs_new, act_new, mem_obs, mem_act) -> float:
    mi = torch.randperm(mem_obs.shape[0], device=DEVICE)[:MEM_BATCH]
    with torch.no_grad():
        before = F.mse_loss(policy(mem_obs[mi]), mem_act[mi]).item()
    virt = copy.deepcopy(policy)
    grads = torch.autograd.grad(F.mse_loss(virt(obs_new), act_new), list(virt.parameters()))
    with torch.no_grad():
        for p, g in zip(virt.parameters(), grads):
            p -= VIRTUAL_LR * g
        return F.mse_loss(virt(mem_obs[mi]), mem_act[mi]).item() - before


@torch.no_grad()
def disagreement_weights(heads, obs, act) -> torch.Tensor:
    preds = torch.stack([h(obs) for h in heads])            # (H, N, A) under previous-task parameters
    d = preds.var(0).mean(-1)
    return torch.exp(-d / d.median().clamp_min(1e-12))


def set_flat_grad(policy, flat: torch.Tensor) -> None:
    i = 0
    for p in policy.parameters():
        n = p.numel(); p.grad = flat[i:i + n].view_as(p).clone(); i += n


def batch_loss(policy, arm, o, a, idx, mem, flow_w, deltas: list[float]) -> torch.Tensor:
    """Per-arm training loss on one new batch (memory arms need MEM_BATCH stored samples)."""
    mem_obs, mem_act = mem
    has_mem = mem_obs is not None and mem_obs.shape[0] >= MEM_BATCH
    per_sample = F.mse_loss(policy(o), a, reduction="none").mean(-1)
    if arm == "lookahead" and has_mem:
        d = max(old_loss_change(policy, o, a, mem_obs, mem_act), 0.0); deltas.append(d)
        pos = [x for x in deltas if x > 0]; med = float(np.median(pos)) if pos else 1.0
        return per_sample.mean() * float(np.exp(-d / (med + 1e-12)))
    if arm == "conflict_w" and has_mem:
        mi = torch.randperm(mem_obs.shape[0], device=DEVICE)[:MEM_BATCH]
        g_mem = flat_grad(policy, F.mse_loss(policy(mem_obs[mi]), mem_act[mi]))
        s = -(per_sample_grads(policy, o, a) @ g_mem) / (g_mem.norm() + 1e-8)
        pos = torch.relu(s); med = pos[pos > 0].median() if (pos > 0).any() else torch.tensor(1.0, device=DEVICE)
        return (per_sample * torch.exp(-pos / (med + 1e-8))).mean()
    if arm in ("flow", "flow_ewc100", "ensemble_w") and flow_w is not None:  # flow_w holds disagreement weights for ensemble_w
        return (per_sample * flow_w[idx]).mean()
    if arm in ("er", "mir") and has_mem:
        mi = mir_select(policy, o, a, mem_obs, mem_act) if arm == "mir" else torch.randperm(mem_obs.shape[0], device=DEVICE)[:MEM_BATCH]
        return 0.5 * (per_sample.mean() + F.mse_loss(policy(mem_obs[mi]), mem_act[mi]))
    return per_sample.mean()


def agem_project(policy, mem_obs, mem_act) -> None:
    g = torch.cat([p.grad.reshape(-1) for p in policy.parameters()])
    mi = torch.randperm(mem_obs.shape[0], device=DEVICE)[:MEM_BATCH]
    g_ref = flat_grad(policy, F.mse_loss(policy(mem_obs[mi]), mem_act[mi]))
    dot = g @ g_ref
    if dot < 0:
        set_flat_grad(policy, g - dot / (g_ref @ g_ref + 1e-12) * g_ref)


def train_task(policy, arm, demos_task, mem, anchors, flow_w) -> None:
    obs_t, act_t, _ = fr.fw.transitions(demos_task)
    mem_obs, mem_act = mem
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    policy.train(); deltas: list[float] = []
    for _ in range(EPOCHS):
        perm = torch.randperm(obs_t.size(0), device=DEVICE)
        for i in range(0, obs_t.size(0), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = batch_loss(policy, arm, obs_t[idx], act_t[idx], idx, mem, flow_w, deltas)
            if anchors and arm in ("ewc", "ewc100", "flow_ewc100"):
                loss = loss + fr.m.ewc_penalty(policy, anchors, EWC_LAM if arm == "ewc" else 100.0)
            loss.backward()
            if arm == "agem" and mem_obs is not None and mem_obs.shape[0] >= MEM_BATCH:
                agem_project(policy, mem_obs, mem_act)
            opt.step()


def run(arm: str, demos, order, seed) -> list[float]:
    """demos[t] holds a task's demonstrations; the last N_EVAL are held out for evaluation (none if N_EVAL == 0)."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    od, ad = demos[0][0]["obs"].shape[-1], demos[0][0]["acts"].shape[-1]
    train = [d[:len(d) - N_EVAL] if N_EVAL else d for d in demos]; evald = [d[len(d) - N_EVAL:] if N_EVAL else d for d in demos]
    policy = make_policy(od, ad)
    heads = [policy] + [make_policy(od, ad) for _ in range(N_HEADS - 1)] if arm == "ensemble_w" else [policy]
    anchors: list[tuple[float, dict, dict]] = []
    mem_obs = mem_act = None
    errs = []
    for i, t in enumerate(order):
        task = train[t]
        if i == 0 or arm == "none":
            for h in heads:
                train_bc(h, task, EPOCHS, device=DEVICE)
        else:
            flow_w = None
            if arm in ("flow", "flow_ewc100"):
                policy.eval(); o, a, _ = fr.fw.transitions(task); flow_w = fr.flow_weights(policy, o, a, 1.0)
            if arm == "ensemble_w":
                for h in heads:
                    h.eval()
                o, a, _ = fr.fw.transitions(task); flow_w = disagreement_weights(heads, o, a)
            for h in heads:  # every head trains with the same weights; only heads[0] (= policy) is evaluated
                train_task(h, arm, task, (mem_obs, mem_act), anchors, flow_w)
        if arm in ("ewc", "ewc100", "flow_ewc100"):
            anchors.append((1.0, fr.m.policy_fisher(policy, task), {n: p.detach().clone() for n, p in policy.named_parameters()}))
        o, a, _ = fr.fw.transitions(task); keep = torch.randperm(o.shape[0], device=DEVICE)[:MEM_PER_TASK]
        mem_obs = o[keep] if mem_obs is None else torch.cat([mem_obs, o[keep]]); mem_act = a[keep] if mem_act is None else torch.cat([mem_act, a[keep]])
        errs.append(float(np.mean([eval_bc(policy, evald[k], DEVICE) for k in order[:i + 1]])))
    return errs


def main() -> None:
    S = fr.SUITE_DIRS
    if MODE == "spatial":
        demos = load_demos(S["spatial"], 10, N_DEMOS); orderings = [fr.orderings_for(6, 10, 0)[i] for i in ORDS]; n_seed = 3
    else:
        demos = load_demos(S["object"], 3, N_DEMOS) + load_demos(S["spatial"], 3, N_DEMOS) + load_demos(S["goal"], 3, N_DEMOS); orderings = [list(range(9))]; n_seed = 5
    print(f"[data] {len(demos)} tasks, demos per task {[len(d) for d in demos]}, held-out per task {N_EVAL}", flush=True)
    per_arm = {a: [] for a in ARMS}
    for oi, order in enumerate(orderings):
        for seed in range(n_seed):
            for arm in ARMS:
                errs = run(arm, demos, order, seed); per_arm[arm].append(errs)
                print(f"[{MODE}] ord={ORDS[oi] if ORDS else 0} seed={seed} arm={arm:10s} first={errs[0]:.4f} final={errs[-1]:.4f}", flush=True)
            json.dump({"orderings": orderings, "raw": per_arm}, open(OUT, "w"), indent=2)
    if len(orderings) > 1 or MODE == "cross":
        summary = fr.summarise(per_arm, len(orderings), n_seed, refs=("none", "ewc", "er"), by_seed=(MODE == "cross"))
        print(f"== {MODE}: " + json.dumps({a: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for a, r in summary.items()}), flush=True)
        json.dump({"orderings": orderings, "raw": per_arm, "summary": summary}, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
