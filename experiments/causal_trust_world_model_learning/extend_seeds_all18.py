"""Extend seeds 5-8 (4 additional seeds, on top of the existing 0-4) for ALL
18 LIBERO backbone/suite cells (not just the 2 closest to significance),
reusing the exact same methodology as sweep_5seeds_full.py (same MLP class,
same train/eval functions, same demo loading, same per-seed torch/numpy
seeding). Writes to a NEW file, does not touch the canonical
sweep_5seeds_all.json. Purpose: test whether the regression-to-the-mean
pattern observed for the 2 closest cells generalizes across the whole sweep,
not just the cells that were cherry-picked for the initial follow-up.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import BACKBONES

DEVICE = "cuda"
SUITES = {
    "spatial": "/home/ubuntu/robotics_world_models/LIBERO/libero_spatial",
    "object": "/home/ubuntu/robotics_world_models/LIBERO/libero_object",
    "goal": "/home/ubuntu/robotics_world_models/LIBERO/libero_goal",
}
N_TASKS = 10
BATCH = 64
LR = 1e-3
EPOCHS = 50
MAX_DEMOS = 5
NEW_SEEDS = range(5, 9)  # 4 additional seeds on top of the existing 0-4
TRUSTS = ["ema", "multi_step", "ensemble"]

BACKBONE_NAMES = ["mlp", "rssm", "jepa", "dreamerv3", "diffusion", "transformer"]
TARGET_CELLS = [(suite, bone) for suite in SUITES for bone in BACKBONE_NAMES]


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def load_demos(suite_dir, n_tasks=10, max_demos=MAX_DEMOS):
    all_demos = []
    files = sorted(f for f in os.listdir(suite_dir) if f.endswith(".hdf5"))
    for fn in files[:n_tasks]:
        with h5py.File(os.path.join(suite_dir, fn), "r") as hf:
            task_demos = []
            count = 0
            for k in sorted(hf["data"].keys()):
                if not k.startswith("demo_") or count >= max_demos:
                    continue
                demo = hf["data"][k]
                parts = [np.array(demo["obs"][f]) for f in
                         ["ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"]]
                task_demos.append({
                    "obs": np.concatenate(parts, axis=-1),
                    "acts": np.array(demo["actions"]),
                })
                count += 1
            all_demos.append(task_demos)
    return all_demos


def train_wm(wm, demos, epochs=20):
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    wm.train()
    for _ in range(epochs):
        for demo in demos:
            o = torch.tensor(demo["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a = torch.tensor(demo["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            loss = wm.train_loss(o, a)
            opt.zero_grad(); loss.backward(); opt.step()


def compute_trust_simple(wm, obs, act, next_obs, method="ema"):
    with torch.no_grad():
        pred = wm.predict_error(obs, act, next_obs)
        pe = pred.mean(dim=-1)
        if method == "ema":
            trust = torch.exp(-pe / (pe.mean() + 1e-8)).clamp(0, 1)
        else:
            trust = torch.exp(-pe).clamp(0, 1)
    return trust


def train_bc(policy, demos, epochs=EPOCHS):
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    all_obs = np.concatenate([d["obs"] for d in demos])
    all_acts = np.concatenate([d["acts"] for d in demos])
    obs_t = torch.tensor(all_obs, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(all_acts, dtype=torch.float32).to(DEVICE)
    policy.train()
    for _ in range(epochs):
        perm = torch.randperm(obs_t.size(0))
        for i in range(0, obs_t.size(0), BATCH):
            idx = perm[i:i+BATCH]
            loss = F.mse_loss(policy(obs_t[idx]), act_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()


def train_bc_trust(policy, wm, demos, trust_method, epochs=EPOCHS):
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    all_obs = np.concatenate([d["obs"] for d in demos])
    all_acts = np.concatenate([d["acts"] for d in demos])
    obs_t = torch.tensor(all_obs, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(all_acts, dtype=torch.float32).to(DEVICE)
    wm.eval(); policy.train()
    for _ in range(epochs):
        perm = torch.randperm(obs_t.size(0))
        for i in range(0, obs_t.size(0), BATCH):
            idx = perm[i:i+BATCH]
            if len(idx) < 2: continue
            pred = policy(obs_t[idx])
            with torch.no_grad():
                trust = compute_trust_simple(wm, obs_t[idx], act_t[idx], obs_t[idx], trust_method)
                w = trust.clamp(0.1, 1.0)
            loss = (F.mse_loss(pred, act_t[idx], reduction="none").mean(dim=-1) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def eval_bc(policy, demos):
    policy.eval()
    all_obs = np.concatenate([d["obs"] for d in demos])
    all_acts = np.concatenate([d["acts"] for d in demos])
    obs_t = torch.tensor(all_obs, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(all_acts, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        error = F.mse_loss(policy(obs_t), act_t).item()
    return error


def run():
    print(f"\n{'='*60}")
    print(f"  Extended Seeds 5-8: ALL 18 backbone/suite cells")
    print(f"{'='*60}")

    new_results = {}
    out = os.path.join(os.path.dirname(__file__), "extended_seeds_5to8_all18.json")
    demos_cache = {}

    for suite_name, bone_name in TARGET_CELLS:
        if suite_name not in demos_cache:
            demos_cache[suite_name] = load_demos(SUITES[suite_name], N_TASKS)
        demos = demos_cache[suite_name]
        obs_dim, act_dim = 21, 7

        cell_results = {"none": [], "ema": [], "multi_step": [], "ensemble": []}

        for seed in NEW_SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            t0 = time.time()

            wm = BACKBONES[bone_name](obs_dim, act_dim).to(DEVICE)
            train_wm(wm, demos[0])

            std = MLP(obs_dim, act_dim).to(DEVICE)
            train_bc(std, demos[0])
            std_err = eval_bc(std, demos[0][3:] if len(demos[0]) > 3 else demos[0][-1:])
            cell_results["none"].append(std_err)

            for tm in TRUSTS:
                tru = MLP(obs_dim, act_dim).to(DEVICE)
                train_bc_trust(tru, wm, demos[0], tm)
                tru_err = eval_bc(tru, demos[0][3:] if len(demos[0]) > 3 else demos[0][-1:])
                cell_results[tm].append(tru_err)

            elapsed = time.time() - t0
            print(f"  {suite_name}/{bone_name} s{seed}: none={std_err:.4f} "
                  f"ema={cell_results['ema'][-1]:.4f} "
                  f"multi={cell_results['multi_step'][-1]:.4f} "
                  f"ens={cell_results['ensemble'][-1]:.4f} "
                  f"({elapsed:.0f}s)")

        new_results[f"{suite_name}/{bone_name}"] = cell_results

        # Save incrementally after each cell so a partial run isn't lost.
        with open(out, "w") as f:
            json.dump(new_results, f, indent=2)
        print(f"  [{len(new_results)}/{len(TARGET_CELLS)} cells] saved to {out}")

    print(f"\nDone. Saved {out}")
    return new_results


if __name__ == "__main__":
    run()
