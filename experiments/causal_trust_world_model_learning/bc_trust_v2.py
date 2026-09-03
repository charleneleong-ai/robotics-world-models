"""BC + Trust on LIBERO-Spatial: 10 Sequential Tasks
Standard BC vs Trust-Weighted BC."""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import BACKBONES, make_trust

DEVICE = "cuda"
LIBERO_DIR = "/home/ubuntu/robotics_world_models/LIBERO/libero_spatial"
N_TASKS = 10
BATCH = 128
LR = 3e-4
EPOCHS = 100
EVAL_DEMOS = 2


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=21, act_dim=7, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h), nn.LayerNorm(h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, act_dim),
        )
    def forward(self, obs):
        return self.net(obs)


def load_demos(suite_dir, n_tasks=10):
    all_demos = []
    files = sorted(f for f in os.listdir(suite_dir) if f.endswith(".hdf5"))
    for fn in files[:n_tasks]:
        with h5py.File(os.path.join(suite_dir, fn), "r") as hf:
            task_demos = []
            for k in sorted(hf["data"].keys()):
                if not k.startswith("demo_"):
                    continue
                demo = hf["data"][k]
                parts = [np.array(demo["obs"][f]) for f in
                         ["ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"]]
                task_demos.append({
                    "obs": np.concatenate(parts, axis=-1),
                    "acts": np.array(demo["actions"]),
                })
            all_demos.append(task_demos)
    return all_demos


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


def train_wm(wm, demos, epochs=50):
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    wm.train()
    for _ in range(epochs):
        for demo in demos:
            o = torch.tensor(demo["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a = torch.tensor(demo["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            loss = wm.train_loss(o, a)
            opt.zero_grad(); loss.backward(); opt.step()


def eval_bc(policy, test_demos):
    policy.eval()
    all_obs = np.concatenate([d["obs"] for d in test_demos])
    all_acts = np.concatenate([d["acts"] for d in test_demos])
    obs_t = torch.tensor(all_obs, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(all_acts, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred = policy(obs_t)
        error = F.mse_loss(pred, act_t).item()
    return error


def run():
    print(f"\n{'='*60}\n  BC + Trust: LIBERO-Spatial 10 Tasks\n{'='*60}")
    
    demos = load_demos(LIBERO_DIR, N_TASKS)
    print(f"Loaded {len(demos)} tasks, {sum(len(d) for d in demos)} demos")
    
    obs_dim, act_dim = 21, 7
    
    # Train world model on Task 0
    wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    print("Training world model on Task 0...")
    train_wm(wm, demos[0])
    
    results = {"std_bc": [], "trust_bc": [], "wm_error": []}
    
    for t in range(N_TASKS):
        print(f"\nTask {t}:")
        
        # Standard BC
        std = BCPolicy(obs_dim, act_dim).to(DEVICE)
        train_bc(std, demos[t])
        std_err = eval_bc(std, demos[t][EVAL_DEMOS:])
        results["std_bc"].append(std_err)
        
        # Trust-weighted BC: weight training samples by WM trust
        trust = BCPolicy(obs_dim, act_dim).to(DEVICE)
        opt = torch.optim.Adam(trust.parameters(), lr=LR)
        all_obs = np.concatenate([d["obs"] for d in demos[t]])
        all_acts = np.concatenate([d["acts"] for d in demos[t]])
        obs_t = torch.tensor(all_obs, dtype=torch.float32).to(DEVICE)
        act_t = torch.tensor(all_acts, dtype=torch.float32).to(DEVICE)
        
        wm.eval()
        trust.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(obs_t.size(0))
            for i in range(0, obs_t.size(0), BATCH):
                idx = perm[i:i+BATCH]
                if len(idx) < 2: continue
                pred = trust(obs_t[idx])
                # Weight by WM trust: states the WM predicts well get higher weight
                with torch.no_grad():
                    pe = wm.predict_error(obs_t[idx], act_t[idx], obs_t[idx])
                    w = torch.exp(-pe.mean(dim=-1) / (pe.mean().item() + 1e-8)).clamp(0.1, 2.0)
                loss = (F.mse_loss(pred, act_t[idx], reduction="none").mean(dim=-1) * w).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        
        trust_err = eval_bc(trust, demos[t][EVAL_DEMOS:])
        results["trust_bc"].append(trust_err)
        
        # WM error on this task
        with torch.no_grad():
            o = torch.tensor(demos[t][0]["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a = torch.tensor(demos[t][0]["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            wm_e = wm.predict_error(o, a, o).mean().item()
        results["wm_error"].append(wm_e)
        
        print(f"  std_bc={std_err:.4f}  trust_bc={trust_err:.4f}  wm_err={wm_e:.6f}")
    
    # Summary
    std = np.array(results["std_bc"])
    tru = np.array(results["trust_bc"])
    wm_e = np.array(results["wm_error"])
    
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"Standard BC: {std.mean():.4f} ± {std.std():.4f}")
    print(f"Trust BC:    {tru.mean():.4f} ± {tru.std():.4f}")
    print(f"WM Error:    {wm_e.mean():.6f} ± {wm_e.std():.6f}")
    
    imp = (std.mean() - tru.mean()) / std.mean() * 100
    print(f"\nTrust BC improvement: {imp:+.1f}%")
    
    # Per-task improvement
    print(f"\nPer-task:")
    for t in range(N_TASKS):
        d = (std[t] - tru[t]) / std[t] * 100
        print(f"  Task {t}: std={std[t]:.4f} trust={tru[t]:.4f} ({d:+.1f}%)")
    
    # WM error reduction
    if N_TASKS > 1:
        wm_imp = (wm_e[0] - wm_e[-1]) / wm_e[0] * 100
        print(f"\nWM error reduction (Task 0→{N_TASKS-1}): {wm_imp:+.1f}%")
    
    return results


if __name__ == "__main__":
    R = run()
    out = os.path.join(os.path.dirname(__file__), "bc_trust_v2_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    print(f"\nSaved to {out}")
