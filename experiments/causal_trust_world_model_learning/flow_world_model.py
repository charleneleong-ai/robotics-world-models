#!/usr/bin/env python3
"""Self-loss weighting for the WORLD MODEL on random-action dynamics streams.

The paper's question is whose reliability should weight learning. Here the learner
is the world model itself: a one-step MLP dynamics model f(o, a) -> o' trained
sequentially on one environment per task, with the world model's next-observation
error on every seen environment as the metric (world-model forgetting).

  none      sequential training
  flow      per-transition weights w_i = exp(-l_i(theta*) / median l), l_i the previous
            task's world model loss on the new transition (first task unweighted)
  ewc       diagonal-Fisher EWC on the world model
  flow_ewc  both

Usage: flow_world_model.py maniskill|kinder  [n_orderings n_seeds]
Data is collected once per environment with random actions (padded to 64/10 dims,
as in maniskill_benchmark / kinder_benchmark) and cached under results_per_param_consolidation/.
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

BENCH = sys.argv[1]
N_ORD = int(sys.argv[2]) if len(sys.argv) > 2 else 6
N_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPISODES, MAX_STEPS, EPOCHS, LAM, HOLDOUT = 30, 100, 20, 100.0, 0.2
ARMS = ["none", "flow", "ewc", "flow_ewc"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_per_param_consolidation")
CACHE = os.path.join(OUT_DIR, f"wm_stream_{BENCH}.npz")
OUT = os.path.join(OUT_DIR, f"flow_world_model_{BENCH}{'_norm' if os.environ.get('NORMALISE', '0') == '1' else ''}.json")


KINDER_ENVS = ["kinder/Motion2D-p0-v0", "kinder/Obstruction2D-o0-v0", "kinder/ClutteredRetrieval2D-o1-v0", "kinder/ClutteredStorage2D-b1-v0",
               "kinder/DynObstruction2D-o0-v0", "kinder/DynPushPullHook2D-o0-v0", "kinder/BaseMotion3D-v0", "kinder/KinematicShelf3D-o1-v0"]
ACT_DIM = 12 if BENCH == "kinder" else 10  # KinDER 3D envs have 11-dim actions


def pad(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, np.float32).reshape(-1)
    return np.pad(x, (0, n - len(x))) if len(x) < n else x[:n]


def flatten_obs(o) -> np.ndarray:
    if isinstance(o, dict):
        return np.concatenate([np.asarray(v, np.float32).reshape(-1) for _, v in sorted(o.items())])
    return np.asarray(o, np.float32).reshape(-1)


def collect_kinder(env_id: str) -> dict[str, np.ndarray]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import kinder
    kinder.register_all_environments()
    env = kinder.make(env_id)  # raises if the id is wrong: no synthetic fallback
    obs, act, nxt = [], [], []
    for _ in range(EPISODES):
        o, _ = env.reset()
        for _ in range(MAX_STEPS):
            a = env.action_space.sample(); o2, _, term, trunc, _ = env.step(a)
            obs.append(pad(flatten_obs(o), 64)); act.append(pad(a, ACT_DIM)); nxt.append(pad(flatten_obs(o2), 64)); o = o2
            if term or trunc:
                break
    env.close()
    return {"obs": np.stack(obs), "act": np.stack(act), "nxt": np.stack(nxt)}


def collect() -> list[dict[str, np.ndarray]]:
    if os.path.exists(CACHE):
        z = np.load(CACHE); n = int(z["n_tasks"])
        return [{k: z[f"{k}{i}"] for k in ("obs", "act", "nxt")} for i in range(n)]
    def no_synthetic(*_a, **_k):
        raise RuntimeError("environment failed to build; refusing the benchmark's synthetic-data fallback")

    tasks = []
    if BENCH == "maniskill":
        from maniskill_benchmark import ManiSkillBenchmark
        bench = ManiSkillBenchmark(num_tasks=7, episodes_per_task=EPISODES, max_steps=MAX_STEPS, obs_dim=64, action_dim=10)
        bench._generate_synthetic_data = no_synthetic
        for e in bench.ENVIRONMENTS:
            d = bench.collect_task_data(e, EPISODES)
            tasks.append({"obs": np.asarray(d["observations"], np.float32), "act": np.asarray(d["actions"], np.float32),
                          "nxt": np.asarray(d["next_observations"], np.float32)})
            print(f"[collect] {e}: {len(tasks[-1]['obs'])} transitions, {int((np.abs(tasks[-1]['obs']).sum(0) > 0).sum())} live obs dims", flush=True)
    else:
        for e in KINDER_ENVS:
            tasks.append(collect_kinder(e))
            print(f"[collect] {e}: {len(tasks[-1]['obs'])} transitions, {int((np.abs(tasks[-1]['obs']).sum(0) > 0).sum())} live obs dims", flush=True)
    np.savez(CACHE, n_tasks=len(tasks), **{f"{k}{i}": t[k] for i, t in enumerate(tasks) for k in t})
    return tasks


NORMALISE = os.environ.get("NORMALISE", "0") == "1"


def split(tasks):
    tr, te = [], []
    for t in tasks:
        n = len(t["obs"]); k = int(n * (1 - HOLDOUT))
        tr.append({k2: v[:k] for k2, v in t.items()}); te.append({k2: v[k:] for k2, v in t.items()})
    if NORMALISE:  # one global scaler fit on the union of training transitions, so every environment contributes on the same scale
        allobs = np.concatenate([t["obs"] for t in tr]); allact = np.concatenate([t["act"] for t in tr])
        mo, so = allobs.mean(0), allobs.std(0) + 1e-6; ma, sa = allact.mean(0), allact.std(0) + 1e-6
        for group in (tr, te):
            for t in group:
                t["obs"] = (t["obs"] - mo) / so; t["nxt"] = (t["nxt"] - mo) / so; t["act"] = (t["act"] - ma) / sa
    return tr, te


def make_wm(od: int, ad: int) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(od + ad, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(),
                               torch.nn.Linear(256, od)).to(DEVICE)


def tens(t):
    return tuple(torch.tensor(t[k], device=DEVICE) for k in ("obs", "act", "nxt"))


def per_sample_loss(wm, obs, act, nxt) -> torch.Tensor:
    return F.mse_loss(wm(torch.cat([obs, act], -1)), nxt, reduction="none").mean(-1)


@torch.no_grad()
def eval_wm(wm, t) -> float:
    wm.eval(); obs, act, nxt = tens(t)
    return float(per_sample_loss(wm, obs, act, nxt).mean())


def ewc_penalty(wm, anchors, lam) -> torch.Tensor:
    pen = torch.tensor(0.0, device=DEVICE)
    for fisher, theta in anchors:
        for n, p in wm.named_parameters():
            pen = pen + (fisher[n] * (p - theta[n]) ** 2).sum()
    return lam / 2 * pen


def fisher(wm, t) -> dict[str, torch.Tensor]:
    obs, act, nxt = tens(t); f = {n: torch.zeros_like(p) for n, p in wm.named_parameters()}
    wm.train(); k = 0
    for i in range(0, len(obs), 64):
        wm.zero_grad(); per_sample_loss(wm, obs[i:i + 64], act[i:i + 64], nxt[i:i + 64]).mean().backward()
        for n, p in wm.named_parameters():
            if p.grad is not None:
                f[n] += p.grad.detach() ** 2
        k += 1
    return {n: v / max(k, 1) for n, v in f.items()}


def train(wm, t, weights, anchors, lam) -> None:
    obs, act, nxt = tens(t); opt = torch.optim.Adam(wm.parameters(), lr=1e-3); wm.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(obs), device=DEVICE)
        for i in range(0, len(obs), 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            ls = per_sample_loss(wm, obs[idx], act[idx], nxt[idx])
            loss = (ls * weights[idx]).mean() if weights is not None else ls.mean()
            if anchors:
                loss = loss + ewc_penalty(wm, anchors, lam)
            opt.zero_grad(); loss.backward(); opt.step()


def run(arm: str, train_tasks, test_tasks, order, seed) -> list[float]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    od, ad = train_tasks[0]["obs"].shape[-1], train_tasks[0]["act"].shape[-1]
    wm = make_wm(od, ad); anchors = []; errs = []
    for i, t in enumerate(order):
        task = train_tasks[t]; weights = None
        if arm.startswith("flow") and i > 0:
            wm.eval()
            with torch.no_grad():
                obs, act, nxt = tens(task); ls = per_sample_loss(wm, obs, act, nxt)
                weights = torch.exp(-ls / ls.median().clamp_min(1e-8))
        train(wm, task, weights, anchors if "ewc" in arm else [], LAM)
        if "ewc" in arm:
            anchors.append((fisher(wm, task), {n: p.detach().clone() for n, p in wm.named_parameters()}))
        errs.append(float(np.mean([eval_wm(wm, test_tasks[o]) for o in order[:i + 1]])))
    return errs


def summarise(per_arm) -> dict:
    bo = lambda x: np.array(x).reshape(N_ORD, N_SEED).mean(1)
    seq = {a: bo(np.array(r)[:, 1:].mean(1)) for a, r in per_arm.items()}; cell = {a: np.array(r)[:, 1:].mean(1) for a, r in per_arm.items()}
    out = {}
    for a in per_arm:
        row = {"seq": float(seq[a].mean()), "first": float(np.array(per_arm[a])[:, 0].mean())}
        for ref in ("none", "ewc"):
            if a != ref and ref in seq:
                row[f"delta_vs_{ref}"] = float(100 * (seq[a].mean() / seq[ref].mean() - 1)); row[f"p_vs_{ref}"] = float(stats.ttest_rel(seq[ref], seq[a])[1])
                row[f"cells_vs_{ref}"] = int((cell[a] < cell[ref]).sum())
        out[a] = row
    return out


def main() -> None:
    tasks = collect(); train_tasks, test_tasks = split(tasks); n = len(tasks)
    rng = random.Random(0); orderings = []
    for _ in range(N_ORD):
        o = list(range(n)); rng.shuffle(o); orderings.append(o)
    per_arm = {a: [] for a in ARMS}
    for oi, order in enumerate(orderings):
        for seed in range(N_SEED):
            for arm in ARMS:
                errs = run(arm, train_tasks, test_tasks, order, seed); per_arm[arm].append(errs)
                print(f"[{BENCH}] ord={oi} seed={seed} arm={arm:9s} first={errs[0]:.4f} final={errs[-1]:.4f}", flush=True)
    summary = summarise(per_arm)
    print(f"== {BENCH}: " + json.dumps({a: {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for a, r in summary.items()}), flush=True)
    json.dump({"summary": summary, "raw": per_arm, "orderings": orderings, "n_tasks": n}, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
