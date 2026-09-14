#!/usr/bin/env python3
"""Closing the loop: CUSUM-triggered FLOW weighting for task-free continual learning.

Result 9 showed CUSUM-triggered EWC consolidation is no better than periodic or
random-time anchoring, despite the detector discriminating true boundaries
(F1~0.52) -- an EWC anchor's *timing* barely matters because Fisher-weighted
regularisation toward a hard parameter snapshot is forgiving of a few chunks'
error. Result 3 showed FLOW (self-loss weighting) is the one signal that
reduces forgetting, but only tested with oracle task boundaries. This asks the
question those two results leave open: does closing the loop -- CUSUM detects
the boundary, FLOW does the weighting, no oracle boundaries anywhere -- let
detection quality matter after all, since FLOW is a soft per-sample weighting
scheme rather than a hard anchor and so may be more sensitive to *when* the
reference snapshot was taken?

Arms (paired on identical orderings/seeds; reuses triggered_consolidation.py's
stream, chunking and CUSUM exactly):
  none            plain BC, no snapshot ever taken
  triggered_flow  snapshot the policy (FLOW reference) when CUSUM fires (ours)
  periodic_flow   snapshot every P chunks, P matched to triggered_flow's count
  random_flow     snapshot at the same count of uniformly random times

At each chunk after the first snapshot, the K policy BC steps are weighted by
FLOW: w_i = exp(-loss_i(theta*) / median loss(theta*)), theta* frozen at the
most recent snapshot. Before any snapshot, BC is unweighted (there is nothing
to protect yet, matching flow_weighting.py's task-0 convention).
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continualwam import get_backbone, load_demos
from task_order_sensitivity import SUITE_DIRS
from boundary_cusum import score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUITE, N_TASKS, MAX_DEMOS, EVAL_FRAC = "spatial", 10, 20, 0.2
CHUNK, K_POLICY_STEPS = 16, 5
N_ORDERINGS, N_SEEDS = 5, 3
WINDOW, LAG, REFRACTORY, CUSUM_K, CUSUM_H = 30, 5, 10, 0.5, 12.0
WM_BACKBONE = "mlp"
RESULTS_DIR = Path(__file__).parent / "results_triggered_flow"
RESULTS_DIR.mkdir(exist_ok=True)


def split_demos(demos):
    stream, evals = [], []
    for task in demos:
        n_eval = max(1, int(len(task) * EVAL_FRAC))
        stream.append(task[:-n_eval]); evals.append(task[-n_eval:])
    return stream, evals


def chunks_for(task_demos):
    out = []
    for d in task_demos:
        o, a = d["obs"], d["acts"]
        for i in range(0, len(o) - CHUNK - 1, CHUNK):
            out.append((o[i:i + CHUNK + 1], a[i:i + CHUNK + 1]))
    return out


def make_policy(obs_dim, act_dim):
    return torch.nn.Sequential(torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, act_dim)).to(DEVICE)


def eval_policy(policy, eval_demos_task) -> float:
    obs = torch.tensor(np.concatenate([d["obs"] for d in eval_demos_task]), dtype=torch.float32, device=DEVICE)
    act = torch.tensor(np.concatenate([d["acts"] for d in eval_demos_task]), dtype=torch.float32, device=DEVICE)
    policy.eval()
    with torch.no_grad():
        return float(F.mse_loss(policy(obs), act).item())


class FlowConsolidator:
    """Holds a frozen snapshot of the policy as the FLOW reference theta*."""

    def __init__(self, policy):
        self.template = policy
        self.theta_star = None
        self.n_snapshots = 0

    def snapshot(self, policy):
        self.theta_star = copy.deepcopy(policy.state_dict())
        self.n_snapshots += 1

    @torch.no_grad()
    def flow_weights(self, ot: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
        ref = make_policy_like(self.template)
        ref.load_state_dict(self.theta_star)
        ref.eval()
        loss = F.mse_loss(ref(ot), at, reduction="none").mean(dim=-1)
        return torch.exp(-loss / loss.median().clamp_min(1e-8))


def make_policy_like(policy):
    # policy is nn.Sequential(Linear, ReLU, Linear, ReLU, Linear); rebuild the same shape.
    dims = [m.in_features for m in policy if isinstance(m, torch.nn.Linear)]
    out_dim = [m.out_features for m in policy if isinstance(m, torch.nn.Linear)][-1]
    return make_policy(dims[0], out_dim)


def cusum_step(errors: list[float], t: int, e: float, s: float, cool: int) -> tuple[bool, float, int]:
    lo, hi = max(0, t - LAG - WINDOW), max(0, t - LAG)
    win = errors[lo:hi]
    if len(win) < 5:
        return False, s, cool
    mu, sd = float(np.mean(win)), float(np.std(win) + 1e-8)
    s = max(0.0, s + (e - mu) / sd - CUSUM_K)
    if cool > 0:
        return False, s, cool - 1
    if s > CUSUM_H:
        return True, 0.0, REFRACTORY
    return False, s, cool


def should_snapshot(arm: str, t: int, fire: bool, period: int | None,
                     random_times: set[int] | None) -> bool:
    match arm:
        case "triggered_flow":
            return fire
        case "periodic_flow":
            return bool(period) and t > 0 and t % period == 0
        case "random_flow":
            return random_times is not None and t in random_times
    return False


def run(arm: str, order: list[int], seed: int, stream_demos, eval_demos, period: int | None = None,
        random_times: set[int] | None = None) -> dict:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    obs_dim = stream_demos[0][0]["obs"].shape[-1]; act_dim = stream_demos[0][0]["acts"].shape[-1]
    wm = get_backbone(WM_BACKBONE, obs_dim, act_dim).to(DEVICE)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    policy = make_policy(obs_dim, act_dim)
    pol_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    cons = FlowConsolidator(policy)

    stream, boundaries, t = [], [], 0
    for task in order:
        ch = chunks_for(stream_demos[task]); random.shuffle(ch)
        boundaries.append(t); stream += ch; t += len(ch)
    boundaries = boundaries[1:]

    errors, alarms, s, cool = [], [], 0.0, 0
    for t, (o, a) in enumerate(stream):
        ot = torch.tensor(o, dtype=torch.float32, device=DEVICE)
        at = torch.tensor(a, dtype=torch.float32, device=DEVICE)
        wm.eval()
        with torch.no_grad():
            e = wm.predict_error(ot[:-1], at[:-1], ot[1:]).mean().item()
        errors.append(e)
        fire, s, cool = cusum_step(errors, t, e, s, cool)
        if fire:
            alarms.append(t)
        if arm != "none" and should_snapshot(arm, t, fire, period, random_times):
            cons.snapshot(policy)
        wm.train(); wm_opt.zero_grad()
        wm.train_loss(ot.unsqueeze(0), at.unsqueeze(0)).backward(); wm_opt.step()
        policy.train()
        if arm != "none" and cons.theta_star is not None:
            weights = cons.flow_weights(ot[:-1], at[:-1])
            for _ in range(K_POLICY_STEPS):
                pol_opt.zero_grad()
                loss = (F.mse_loss(policy(ot[:-1]), at[:-1], reduction="none").mean(dim=-1) * weights).mean()
                loss.backward(); pol_opt.step()
        else:
            for _ in range(K_POLICY_STEPS):
                pol_opt.zero_grad()
                loss = F.mse_loss(policy(ot[:-1]), at[:-1])
                loss.backward(); pol_opt.step()

    per_task = {int(task): eval_policy(policy, eval_demos[task]) for task in order}
    out = {"final_mean_error": float(np.mean(list(per_task.values()))), "per_task": per_task,
           "n_snapshots": cons.n_snapshots, "n_alarms": len(alarms)}
    if arm == "triggered_flow":
        out["detection"] = score(alarms, boundaries)
    return out


def main() -> None:
    demos = load_demos(SUITE_DIRS[SUITE], N_TASKS, MAX_DEMOS)
    stream_demos, eval_demos = split_demos(demos)
    rng = random.Random(0)
    orderings = []
    for _ in range(N_ORDERINGS):
        o = list(range(N_TASKS)); rng.shuffle(o); orderings.append(o)

    arms = ["none", "triggered_flow", "periodic_flow", "random_flow"]
    results = {a: [] for a in arms}
    for oi, order in enumerate(orderings):
        for seed in range(N_SEEDS):
            r_trig = run("triggered_flow", order, seed, stream_demos, eval_demos)
            n_chunks = sum(len(chunks_for(stream_demos[t])) for t in order)
            n_snap = max(1, r_trig["n_snapshots"])
            period = max(1, n_chunks // n_snap)
            rand_rng = random.Random(1000 * oi + seed)
            random_times = set(rand_rng.sample(range(WINDOW + LAG, n_chunks), n_snap))
            cell = {"triggered_flow": r_trig,
                    "none": run("none", order, seed, stream_demos, eval_demos),
                    "periodic_flow": run("periodic_flow", order, seed, stream_demos, eval_demos, period=period),
                    "random_flow": run("random_flow", order, seed, stream_demos, eval_demos, random_times=random_times)}
            for a, r in cell.items():
                results[a].append(r)
            print(f"ord={oi} seed={seed} " + " ".join(f"{a}={cell[a]['final_mean_error']:.4f}" for a in cell)
                  + f" | trig_snap={r_trig['n_snapshots']} F1={r_trig['detection']['f1']:.2f} period={period}", flush=True)

    summary = {}
    base = [r["final_mean_error"] for r in results["none"]]
    for a, runs in results.items():
        v = [r["final_mean_error"] for r in runs]
        entry = {"final_mean_error": float(np.mean(v)), "std": float(np.std(v)),
                 "n_snapshots": float(np.mean([r["n_snapshots"] for r in runs])) if a != "none" else 0.0}
        if a != "none":
            entry["p_vs_none_paired"] = float(stats.ttest_rel(base, v)[1])
        if a in ("periodic_flow", "random_flow"):
            vt = [r["final_mean_error"] for r in results["triggered_flow"]]
            entry["p_vs_triggered_paired"] = float(stats.ttest_rel(vt, v)[1])
        summary[a] = entry
    summary["triggered_flow"]["detection_f1"] = float(np.mean([r["detection"]["f1"] for r in results["triggered_flow"]]))
    print("== " + json.dumps({a: {k: round(x, 4) for k, x in d.items()} for a, d in summary.items()}), flush=True)
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump({"summary": summary, "raw": results}, f, indent=2)


if __name__ == "__main__":
    main()
