#!/usr/bin/env python3
"""FLOW self-loss weighting on a second expert-demonstration benchmark: ManiSkill.

Five ManiSkill3 tasks with official motion-planning demonstrations (PickCube, PushCube,
StackCube, PegInsertionSide, PlugCharger). The demo files store per-step environment
states rather than observations, so the state observation is assembled from them:
robot articulation state followed by every actor pose, zero-padded to OBS_DIM; actions
are the recorded 8-dim joint-position commands. N_DEMOS successful episodes per task.
The task-order protocol, arms and statistics are flow_robustness.run / summarise.

Usage: flow_maniskill_bc.py <ordering indices, e.g. 0,1,2> [n_seeds]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ORDS = [int(x) for x in sys.argv[1].split(",")]
N_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 3
sys.argv = [sys.argv[0], "temp"]
import flow_robustness as fr  # noqa: E402

TASKS = ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PegInsertionSide-v1", "PlugCharger-v1"]
DEMO_ROOT = os.path.expanduser("~/.maniskill/demos")
OBS_DIM, N_DEMOS, N_ORD_TOTAL = 72, 10, 6
ARMS = ["none", "flow", "ewc", "flow_ewc"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_per_param_consolidation",
                   f"flow_maniskill_bc_ord{'_'.join(map(str, ORDS))}.json")


def state_vector(traj: h5py.Group) -> np.ndarray:
    parts = [np.asarray(traj["env_states/articulations"][k]) for k in sorted(traj["env_states/articulations"])]
    parts += [np.asarray(traj["env_states/actors"][k]) for k in sorted(traj["env_states/actors"])]
    s = np.concatenate(parts, axis=1).astype(np.float32)
    assert s.shape[1] <= OBS_DIM, s.shape
    return np.pad(s, ((0, 0), (0, OBS_DIM - s.shape[1])))


def load_task(name: str) -> list[dict[str, np.ndarray]]:
    f = glob.glob(f"{DEMO_ROOT}/{name}/motionplanning/trajectory.h5")[0]
    demos = []
    with h5py.File(f) as h:
        for k in sorted(h.keys(), key=lambda s: int(s.split("_")[1])):
            t = h[k]
            if "success" in t and not bool(np.asarray(t["success"])[-1]):
                continue
            acts = np.asarray(t["actions"], np.float32); obs = state_vector(t)[: len(acts) + 1]
            demos.append({"obs": obs, "acts": np.concatenate([acts, acts[-1:]], axis=0)})  # obs and acts length-matched as in LIBERO loader
            if len(demos) == N_DEMOS:
                break
    return demos


def main() -> None:
    demos = [load_task(t) for t in TASKS]
    for t, d in zip(TASKS, demos):
        print(f"[load] {t}: {len(d)} demos, {sum(len(x['obs']) for x in d)} steps, state dims {int((np.abs(np.concatenate([x['obs'] for x in d])).sum(0) > 0).sum())}", flush=True)
    all_orderings = fr.orderings_for(N_ORD_TOTAL, len(TASKS), 0)
    orderings = [all_orderings[i] for i in ORDS]
    per_arm = {a: [] for a in ARMS}
    for oi, order in zip(ORDS, orderings):
        for seed in range(N_SEED):
            for arm in ARMS:
                errs = fr.run(arm, demos, order, seed)
                per_arm[arm].append(errs)
                print(f"[maniskill_bc] ord={oi} seed={seed} arm={arm:9s} first={errs[0]:.4f} final={errs[-1]:.4f}", flush=True)
            json.dump({"orderings": orderings, "raw": per_arm}, open(OUT, "w"), indent=2)
    if len(ORDS) > 1:
        summary = fr.summarise(per_arm, len(ORDS), N_SEED)
        print("== maniskill_bc: " + json.dumps({a: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for a, r in summary.items()}), flush=True)
        json.dump({"orderings": orderings, "raw": per_arm, "summary": summary}, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
