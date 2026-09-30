"""Re-run the world-model sweep with the real reliability estimators.

The earlier re-runs used a simplified trust function in which the ensemble
estimator was the exponential of prediction error, making it identical to the
multi-step estimator, and the trust signal was computed with next_obs set to
obs. This uses the estimator classes from full_backbone_sweep.py instead, and
supplies genuine one-step transitions, so:

  ema        EMATrust               exp(-error / running EMA of mean error)
  multi_step MultiStepAdaptiveTrust exp(-error), with an adaptive horizon
  ensemble   EnsembleDisagreement   exp(-variance across five trained heads)

The ensemble heads are trained alongside the policy on (obs -> next_obs), which
is why real transitions are required rather than obs repeated.

Configurations, chosen with `python rerun_v2.py <mode>`:

  n5        one task, train on demos 0-4, score on demos 3-4  (matches the
            original sweep's protocol, seeds 0-4)
  n9ext     the same, seeds 5-8, for the n=9 extension
  heldout   one task, train on demos 0-2, score on the held-out 3-4
  alltasks  all ten tasks, train on demos 0-2, score on the held-out 3-4

Everything else -- architectures, optimiser, learning rate, batch size, epoch
counts, per-seed torch/numpy seeding -- matches the earlier runs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SWEEP_DIR = Path(os.environ.get("SWEEP_DIR", Path(__file__).resolve().parent))
LIBERO_ROOT = Path(os.environ.get("LIBERO_ROOT",
                                 "/home/ubuntu/robotics_world_models/LIBERO"))
sys.path.insert(0, str(SWEEP_DIR))
# full_backbone_sweep pulls in logging helpers that are no longer present;
# a no-op shim satisfies the import without touching the sweep module.
sys.path.insert(0, os.environ.get("SHIM_DIR", "/home/ubuntu/rerun_shim"))

from full_backbone_sweep import BACKBONES, make_trust  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUITES = ["spatial", "object", "goal"]
BACKBONE_NAMES = ["mlp", "rssm", "jepa", "dreamerv3", "diffusion", "transformer"]
TRUSTS = ["ema", "multi_step", "ensemble"]

OBS_DIM, ACT_DIM = 21, 7
BATCH, LR, EPOCHS, WM_EPOCHS = 64, 1e-3, 50, 20
MAX_DEMOS = 5
OBS_FIELDS = ["ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"]

MODES = {
    # name:      (all_tasks, n_train_demos, seeds)
    "n5":        (False, 5, range(0, 5)),
    "n9ext":     (False, 5, range(5, 9)),
    "heldout":   (False, 3, range(0, 5)),
    "alltasks":  (True,  3, range(0, 5)),
}


class Policy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_suite(suite: str, n_tasks: int = 10) -> list[list[dict[str, np.ndarray]]]:
    suite_dir = LIBERO_ROOT / f"libero_{suite}"
    tasks = []
    for filename in sorted(f for f in os.listdir(suite_dir) if f.endswith(".hdf5"))[:n_tasks]:
        with h5py.File(suite_dir / filename, "r") as hf:
            demos = []
            for key in sorted(hf["data"].keys()):
                if not key.startswith("demo_") or len(demos) >= MAX_DEMOS:
                    continue
                demo = hf["data"][key]
                demos.append({
                    "obs": np.concatenate([np.array(demo["obs"][f]) for f in OBS_FIELDS], axis=-1),
                    "acts": np.array(demo["actions"]),
                })
            tasks.append(demos)
    return tasks


def transitions(demos: list[dict[str, np.ndarray]]) -> tuple[torch.Tensor, ...]:
    """Flatten demos into (obs, act, next_obs), dropping each demo's last step."""
    obs, acts, nxt = [], [], []
    for demo in demos:
        o, a = demo["obs"], demo["acts"]
        n = min(len(o), len(a))
        if n < 2:
            continue
        obs.append(o[:n - 1])
        acts.append(a[:n - 1])
        nxt.append(o[1:n])
    to = lambda arr: torch.tensor(np.concatenate(arr), dtype=torch.float32).to(DEVICE)
    return to(obs), to(acts), to(nxt)


def train_world_model(wm: nn.Module, demos: list[dict[str, np.ndarray]]) -> None:
    opt = torch.optim.Adam(wm.parameters(), lr=LR)
    wm.train()
    for _ in range(WM_EPOCHS):
        for demo in demos:
            o = torch.tensor(demo["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a = torch.tensor(demo["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            loss = wm.train_loss(o, a)
            opt.zero_grad()
            loss.backward()
            opt.step()


def to_device(estimator):
    for attr in ("heads", "verifier", "enc"):
        module = getattr(estimator, attr, None)
        if isinstance(module, nn.Module):
            module.to(DEVICE)
    return estimator


def train_policy(obs_t, act_t, next_t, wm=None, method=None, diagnostics=None) -> Policy:
    policy = Policy(OBS_DIM, ACT_DIM).to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    estimator = to_device(make_trust(method, OBS_DIM, ACT_DIM)) if method else None
    if wm is not None:
        wm.eval()
    policy.train()

    for _ in range(EPOCHS):
        order = torch.randperm(obs_t.size(0))
        for start in range(0, obs_t.size(0), BATCH):
            idx = order[start:start + BATCH]
            if estimator is None:
                loss = F.mse_loss(policy(obs_t[idx]), act_t[idx])
            else:
                if len(idx) < 2:
                    continue
                obs_b, act_b, next_b = obs_t[idx], act_t[idx], next_t[idx]
                if method == "ensemble":
                    estimator.train_step(obs_b, next_b)
                    weights = estimator.compute_trust(obs_b)
                else:
                    with torch.no_grad():
                        error = wm.predict_error(obs_b, act_b, next_b).mean(dim=-1)
                    weights = estimator.compute_trust(error, 0)
                weights = weights.detach().clamp(0.1, 1.0)
                if diagnostics is not None:
                    diagnostics.append(float(weights.mean()))
                per_sample = F.mse_loss(policy(obs_b), act_b, reduction="none").mean(dim=-1)
                loss = (per_sample * weights).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy


def score(policy: Policy, obs_t: torch.Tensor, act_t: torch.Tensor) -> float:
    policy.eval()
    with torch.no_grad():
        return F.mse_loss(policy(obs_t), act_t).item()


def run(mode: str, cells: list[tuple[str, str]] | None = None, diagnose: bool = False) -> None:
    all_tasks, n_train, seeds = MODES[mode]
    out_path = SWEEP_DIR / f"v2_{mode}_all18.json"
    targets = cells or [(s, b) for s in SUITES for b in BACKBONE_NAMES]
    print(f"[{mode}] tasks={'all 10' if all_tasks else '1'} train_demos=0-{n_train - 1} "
          f"score=3-4 seeds={list(seeds)} cells={len(targets)}", flush=True)

    results: dict[str, dict[str, list[float]]] = {}
    cache: dict[str, list[list[dict[str, np.ndarray]]]] = {}
    for suite, backbone in targets:
        if suite not in cache:
            cache[suite] = load_suite(suite)
        tasks = cache[suite] if all_tasks else cache[suite][:1]
        train_demos = [d for t in tasks for d in t[:n_train]]
        eval_demos = [d for t in tasks for d in t[3:]]
        train_obs, train_act, train_next = transitions(train_demos)
        eval_obs, eval_act, _ = transitions(eval_demos)

        cell: dict[str, list[float]] = {k: [] for k in ["none", *TRUSTS]}
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            started = time.time()

            wm = BACKBONES[backbone](OBS_DIM, ACT_DIM).to(DEVICE)
            train_world_model(wm, train_demos)

            cell["none"].append(score(train_policy(train_obs, train_act, train_next),
                                      eval_obs, eval_act))
            for method in TRUSTS:
                diags: list[float] | None = [] if diagnose else None
                policy = train_policy(train_obs, train_act, train_next, wm, method, diags)
                cell[method].append(score(policy, eval_obs, eval_act))
                if diagnose:
                    print(f"      {method:<11} mean trust weight "
                          f"{np.mean(diags):.4f} (n_batches={len(diags)})", flush=True)

            print(f"  {suite}/{backbone} s{seed}: "
                  + " ".join(f"{k}={cell[k][-1]:.4f}" for k in ["none", *TRUSTS])
                  + f" ({time.time() - started:.0f}s)", flush=True)

        results[f"{suite}/{backbone}"] = cell
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [{len(results)}/{len(targets)} cells] saved", flush=True)

    print(f"done -> {out_path}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "heldout"
    if mode not in MODES:
        raise SystemExit(f"usage: rerun_v2.py [{'|'.join(MODES)}] [--smoke]")
    smoke = "--smoke" in sys.argv
    run(mode, cells=[("spatial", "mlp")] if smoke else None, diagnose=smoke)
