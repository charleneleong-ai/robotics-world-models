#!/usr/bin/env python3
"""Reliability-triggered consolidation for task-free continual learning.

The audit shows every *level*-based coupling of prediction-error trust fails
(per-sample weighting, batch gating, replay priority, eviction), while the
signal is a genuine *change* statistic (CUSUM boundary detection, F1 ~0.5-0.6).
This tests the method that follows: consolidate the policy (EWC anchor + Fisher
on the recent window) only when an online CUSUM on the world model's pre-update
prediction error fires. Task boundaries are never given to the method.

Arms (paired on identical orderings/seeds):
  none      no consolidation
  oracle    consolidate at the true boundaries (upper bound)
  triggered consolidate when CUSUM fires (ours)
  periodic  consolidate every P chunks, P chosen so the anchor count matches
            triggered's mean alarm count (control for "just consolidate often")

Stream: LIBERO-Spatial 10 tasks in a random order, 16-transition chunks; per
chunk: pre-update WM error -> CUSUM; WM update; K policy BC steps with EWC.
Eval: policy BC error on held-out demos of every task at the end of the stream.
"""
from __future__ import annotations

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
SUITE, N_TASKS, MAX_DEMOS, EVAL_FRAC = "spatial", 10, 20, 0.2   # 20 demos/task: the regime where the CUSUM detector was validated (Appendix B)
CHUNK, K_POLICY_STEPS = 16, 5
N_ORDERINGS, N_SEEDS = 5, 3
WINDOW, LAG, REFRACTORY, CUSUM_K, CUSUM_H = 30, 5, 10, 0.5, 12.0
FISHER_WINDOW, FISHER_SKIP = 30, 3     # Fisher on chunks [-30, -3): the task that just ended
EWC_LAMBDA = 100.0
WM_BACKBONE = "mlp"
RESULTS_DIR = Path(__file__).parent / "results_triggered_consolidation"
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


class Consolidator:
    def __init__(self, policy):
        self.policy = policy
        self.anchors: list[tuple[dict, dict]] = []

    def penalty(self):
        pen = torch.tensor(0.0, device=DEVICE)
        for fisher, theta in self.anchors:
            for n, p in self.policy.named_parameters():
                pen = pen + (fisher[n] * (p - theta[n]) ** 2).sum()
        return EWC_LAMBDA / 2 * pen

    def consolidate(self, recent: list[tuple[np.ndarray, np.ndarray]]):
        if len(recent) < 2:
            return
        fisher = {n: torch.zeros_like(p) for n, p in self.policy.named_parameters()}
        self.policy.train()
        for o, a in recent:
            ot = torch.tensor(o[:-1], dtype=torch.float32, device=DEVICE)
            at = torch.tensor(a[:-1], dtype=torch.float32, device=DEVICE)
            self.policy.zero_grad()
            F.mse_loss(self.policy(ot), at).backward()
            for n, p in self.policy.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        fisher = {n: v / len(recent) for n, v in fisher.items()}
        self.anchors.append((fisher, {n: p.detach().clone() for n, p in self.policy.named_parameters()}))


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


def should_consolidate(arm: str, t: int, fire: bool, true_b: set[int], period: int | None,
                       random_times: set[int] | None) -> bool:
    match arm:
        case "oracle":
            return t in true_b
        case "triggered":
            return fire
        case "periodic":
            return bool(period) and t > 0 and t % period == 0
        case "random":
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
    cons = Consolidator(policy)

    # build the stream (method never sees `boundaries`)
    stream, boundaries, t = [], [], 0
    for task in order:
        ch = chunks_for(stream_demos[task]); random.shuffle(ch)
        boundaries.append(t); stream += ch; t += len(ch)
    boundaries = boundaries[1:]
    true_b = set(boundaries)

    errors, alarms, s, cool = [], [], 0.0, 0
    recent: deque = deque(maxlen=FISHER_WINDOW)
    for t, (o, a) in enumerate(stream):
        ot = torch.tensor(o, dtype=torch.float32, device=DEVICE)
        at = torch.tensor(a, dtype=torch.float32, device=DEVICE)
        # --- change statistic from pre-update WM error
        wm.eval()
        with torch.no_grad():
            e = wm.predict_error(ot[:-1], at[:-1], ot[1:]).mean().item()
        errors.append(e)
        fire, s, cool = cusum_step(errors, t, e, s, cool)
        if fire:
            alarms.append(t)
        # --- decide whether to consolidate
        if should_consolidate(arm, t, fire, true_b, period, random_times):
            rec = list(recent)[:-FISHER_SKIP] if len(recent) > FISHER_SKIP else list(recent)
            cons.consolidate(rec)
        # --- updates
        wm.train(); wm_opt.zero_grad()
        wm.train_loss(ot.unsqueeze(0), at.unsqueeze(0)).backward(); wm_opt.step()
        policy.train()
        for _ in range(K_POLICY_STEPS):
            pol_opt.zero_grad()
            loss = F.mse_loss(policy(ot[:-1]), at[:-1])
            if arm != "none" and cons.anchors:
                loss = loss + cons.penalty()
            loss.backward(); pol_opt.step()
        recent.append((o, a))

    per_task = {int(task): eval_policy(policy, eval_demos[task]) for task in order}
    out = {"final_mean_error": float(np.mean(list(per_task.values()))), "per_task": per_task,
           "n_consolidations": len(cons.anchors), "n_alarms": len(alarms)}
    if arm == "triggered":
        out["detection"] = score(alarms, boundaries)
    return out


def main() -> None:
    demos = load_demos(SUITE_DIRS[SUITE], N_TASKS, MAX_DEMOS)
    stream_demos, eval_demos = split_demos(demos)
    rng = random.Random(0)
    orderings = []
    for _ in range(N_ORDERINGS):
        o = list(range(N_TASKS)); rng.shuffle(o); orderings.append(o)

    results = {a: [] for a in ["none", "oracle", "triggered", "periodic", "random"]}
    for oi, order in enumerate(orderings):
        for seed in range(N_SEEDS):
            r_trig = run("triggered", order, seed, stream_demos, eval_demos)
            n_chunks = sum(len(chunks_for(stream_demos[t])) for t in order)
            n_cons = max(1, r_trig["n_consolidations"])
            period = max(1, n_chunks // n_cons)
            # random-time control: same anchor count as triggered, positions uniform over the stream
            # (after the detector's warm-up), drawn from a per-cell RNG independent of the model seed
            rand_rng = random.Random(1000 * oi + seed)
            random_times = set(rand_rng.sample(range(WINDOW + LAG, n_chunks), n_cons))
            cell = {"triggered": r_trig,
                    "none": run("none", order, seed, stream_demos, eval_demos),
                    "oracle": run("oracle", order, seed, stream_demos, eval_demos),
                    "periodic": run("periodic", order, seed, stream_demos, eval_demos, period=period),
                    "random": run("random", order, seed, stream_demos, eval_demos, random_times=random_times)}
            for a, r in cell.items():
                results[a].append(r)
            print(f"ord={oi} seed={seed} " + " ".join(f"{a}={cell[a]['final_mean_error']:.4f}" for a in cell)
                  + f" | trig_cons={r_trig['n_consolidations']} F1={r_trig['detection']['f1']:.2f} period={period}", flush=True)

    summary = {}
    base = [r["final_mean_error"] for r in results["none"]]
    for a, runs in results.items():
        v = [r["final_mean_error"] for r in runs]
        entry = {"final_mean_error": float(np.mean(v)), "std": float(np.std(v)),
                 "n_consolidations": float(np.mean([r["n_consolidations"] for r in runs]))}
        if a != "none":
            entry["p_vs_none_paired"] = float(stats.ttest_rel(base, v)[1])
        if a in ("periodic", "oracle", "random"):
            vt = [r["final_mean_error"] for r in results["triggered"]]
            entry["p_vs_triggered_paired"] = float(stats.ttest_rel(vt, v)[1])
        summary[a] = entry
    summary["triggered"]["detection_f1"] = float(np.mean([r["detection"]["f1"] for r in results["triggered"]]))
    print("== " + json.dumps({a: {k: round(x, 4) for k, x in d.items()} for a, d in summary.items()}), flush=True)
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump({"summary": summary, "raw": results}, f, indent=2)


if __name__ == "__main__":
    main()
