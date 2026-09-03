"""BC + Trust on ALL LIBERO benchmarks: spatial, object, goal
Trains BC with/without trust weighting across all 3 suites."""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import BACKBONES, make_trust

DEVICE = "cuda"
LIBERO_DIR = "/home/ubuntu/robotics_world_models/LIBERO"
SUITES = ["libero_spatial", "libero_object", "libero_goal"]
N_TASKS = 10
BATCH = 64
LR = 1e-3
EPOCHS = 50
MAX_DEMOS = 5


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


def train_bc_trust(policy, wm, demos, epochs=EPOCHS):
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
                pe = wm.predict_error(obs_t[idx], act_t[idx], obs_t[idx])
                w = torch.exp(-pe.mean(dim=-1) / (pe.mean().item() + 1e-8)).clamp(0.1, 3.0)
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


def run_suite(suite_name):
    suite_dir = os.path.join(LIBERO_DIR, suite_name)
    print(f"\n{'='*60}\n  {suite_name}\n{'='*60}")
    
    demos = load_demos(suite_dir, N_TASKS)
    print(f"Loaded {len(demos)} tasks, {[len(d) for d in demos]} demos")
    
    obs_dim, act_dim = 21, 7
    wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    
    # Quick WM train on Task 0
    opt_wm = torch.optim.Adam(wm.parameters(), lr=1e-3)
    wm.train()
    for _ in range(20):
        for demo in demos[0]:
            o = torch.tensor(demo["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a = torch.tensor(demo["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            loss = wm.train_loss(o, a)
            opt_wm.zero_grad(); loss.backward(); opt_wm.step()
    
    results = {"std": [], "trust": [], "wm_err": []}
    
    for t in range(N_TASKS):
        train_demos = demos[t][:3]
        test_demos = demos[t][3:] if len(demos[t]) > 3 else demos[t][-1:]
        
        std = MLP(obs_dim, act_dim).to(DEVICE)
        train_bc(std, train_demos)
        std_err = eval_bc(std, test_demos)
        
        tru = MLP(obs_dim, act_dim).to(DEVICE)
        train_bc_trust(tru, wm, train_demos)
        tru_err = eval_bc(tru, test_demos)
        
        with torch.no_grad():
            o = torch.tensor(test_demos[0]["obs"][:50], dtype=torch.float32).to(DEVICE)
            a = torch.tensor(test_demos[0]["acts"][:50], dtype=torch.float32).to(DEVICE)
            wm_e = wm.predict_error(o, a, o).mean().item()
        
        results["std"].append(std_err)
        results["trust"].append(tru_err)
        results["wm_err"].append(wm_e)
        
        imp = (std_err - tru_err) / std_err * 100
        print(f"  Task {t}: std={std_err:.4f} trust={tru_err:.4f} ({imp:+.1f}%) wm={wm_e:.6f}")
    
    std_a = np.array(results["std"])
    tru_a = np.array(results["trust"])
    imp = (std_a.mean() - tru_a.mean()) / std_a.mean() * 100
    print(f"\n  Average: std={std_a.mean():.4f} trust={tru_a.mean():.4f} ({imp:+.1f}%)")
    
    return results


if __name__ == "__main__":
    all_results = {}
    for suite in SUITES:
        all_results[suite] = run_suite(suite)
    
    # Summary
    print(f"\n{'='*60}\n  CROSS-SUITE SUMMARY\n{'='*60}")
    for suite, r in all_results.items():
        std_a = np.array(r["std"])
        tru_a = np.array(r["trust"])
        imp = (std_a.mean() - tru_a.mean()) / std_a.mean() * 100
        print(f"  {suite}: std={std_a.mean():.4f} trust={tru_a.mean():.4f} ({imp:+.1f}%)")
    
    out = os.path.join(os.path.dirname(__file__), "bc_trust_all_suites.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")
