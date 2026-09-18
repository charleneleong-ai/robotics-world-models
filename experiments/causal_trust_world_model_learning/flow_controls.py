#!/usr/bin/env python3
"""Controls for FLOW self-loss weighting on fresh orderings (Random(1)).

  none          plain BC
  flow          FLOW: w_i = exp(-l_i(theta*)/median l), theta* = policy after the previous task
  flow_const    every sample gets the mean FLOW weight (tests the effective-step-size explanation)
  flow_shuffled FLOW weights permuted across samples (same distribution, no alignment)
  flow_hard     the inverse assignment, w_i = 1 - exp(-l_i/median l) (up-weights hard samples)
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flow_weighting as fw
from continualwam import eval_bc, get_backbone, load_demos, train_bc
from continualwam.training import train_wm

ARMS = ["none", "flow", "flow_const", "flow_shuffled", "flow_hard"]
SELECTED = sys.argv[1:] or ["mlp"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_per_param_consolidation",
                   f"flow_controls_{'_'.join(SELECTED)}.json")


@torch.no_grad()
def control_weights(arm: str, policy, obs, act) -> torch.Tensor:
    loss = F.mse_loss(policy(obs), act, reduction="none").mean(dim=-1)
    w = torch.exp(-loss / loss.median().clamp_min(1e-8))
    match arm:
        case "flow":
            return w
        case "flow_const":
            return torch.full_like(w, float(w.mean()))
        case "flow_shuffled":
            return w[torch.randperm(w.numel(), device=w.device)]
        case "flow_hard":
            return 1.0 - w
    raise ValueError(arm)


def run(backbone: str, arm: str, demos, order, seed) -> list[float]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    obs_dim = demos[0][0]["obs"].shape[-1]; act_dim = demos[0][0]["acts"].shape[-1]
    wm = get_backbone(backbone, obs_dim, act_dim).to(fw.DEVICE)
    policy = torch.nn.Sequential(torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                                 torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                 torch.nn.Linear(128, act_dim)).to(fw.DEVICE)
    errors_per_task = []
    for i, t in enumerate(order):
        task_demos = demos[t]
        train_wm(wm, task_demos, fw.m.WM_EPOCHS, device=fw.DEVICE)
        if arm == "none" or i == 0:
            train_bc(policy, task_demos, fw.m.BC_EPOCHS, device=fw.DEVICE)
        else:
            policy.eval()
            obs, act, _ = fw.transitions(task_demos)
            fw.train_bc_weighted(policy, task_demos, control_weights(arm, policy, obs, act))
        errors_per_task.append(float(np.mean([eval_bc(policy, demos[o], fw.DEVICE) for o in order[:i + 1]])))
    return errors_per_task


def main() -> None:
    demos = load_demos(fw.SUITE_DIRS["spatial"], 10, 5)
    rng = random.Random(1)
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
        summary = fw.summarise(per_arm)
        print(f"== {backbone}: " + json.dumps({a: {k: round(v, 4) for k, v in r.items()} for a, r in summary.items()}), flush=True)
        results[backbone] = {"summary": summary, "raw": per_arm}
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
