"""Test multi-step trust on JEPA specifically."""

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import h5py

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import JEPABackbone

DEVICE = "cuda"
SUITE_DIR = "/home/ubuntu/robotics_world_models/LIBERO/libero_spatial"
N_TASKS = 10
BATCH = 64
LR = 1e-3
EPOCHS = 50
MAX_DEMOS = 5
N_SEEDS = 5
WM_EPOCHS = 20


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, h=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, out_dim),
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


def train_wm(wm, demos, epochs=WM_EPOCHS):
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    wm.train()
    for _ in range(epochs):
        for demo in demos:
            o = torch.tensor(demo["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a = torch.tensor(demo["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            loss = wm.train_loss(o, a)
            opt.zero_grad(); loss.backward(); opt.step()


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


def eval_bc(policy, demos):
    policy.eval()
    all_obs = np.concatenate([d["obs"] for d in demos])
    all_acts = np.concatenate([d["acts"] for d in demos])
    obs_t = torch.tensor(all_obs, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(all_acts, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        error = F.mse_loss(policy(obs_t), act_t).item()
    return error


def compute_multistep_trust(wm, obs, act, next_obs, k=3):
    """Multi-step trust: average prediction error over k-step rollout."""
    with torch.no_grad():
        # Single-step error
        pe1 = wm.predict_error(obs, act, next_obs).mean(dim=-1)
        
        # Multi-step: use latent space rollout
        if hasattr(wm, 'encoder') and hasattr(wm, 'predictor'):
            z = wm.encoder(obs)
            # k-step rollout in latent space
            total_error = pe1.clone()
            for step in range(1, k):
                z = wm.predictor(torch.cat([z, act], dim=-1))
                # Compare with encoder output at step+1
                z_target = wm.encoder(next_obs)
                step_error = F.mse_loss(z, z_target, reduction="none").mean(dim=-1)
                total_error = total_error + step_error
            pe = total_error / k
        else:
            pe = pe1
        
        trust = torch.exp(-pe / (pe.mean() + 1e-8)).clamp(0, 1)
    return trust


def train_bc_multistep_trust(policy, wm, demos, k=3, epochs=EPOCHS):
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
                trust = compute_multistep_trust(wm, obs_t[idx], act_t[idx], obs_t[idx], k=k)
                w = trust.clamp(0.1, 1.0)
            loss = (F.mse_loss(pred, act_t[idx], reduction="none").mean(dim=-1) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def run():
    print(f"\n{'='*60}")
    print(f"  JEPA Multi-Step Trust Test (k=1,3,5)")
    print(f"{'='*60}")
    
    demos = load_demos(SUITE_DIR, N_TASKS)
    print(f"Loaded {len(demos)} tasks")
    
    obs_dim, act_dim = 21, 7
    results = {"none": [], "ema": [], "multi_k1": [], "multi_k3": [], "multi_k5": []}
    
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        t0 = __import__('time').time()
        
        # Train WM
        wm = JEPABackbone(obs_dim, act_dim).to(DEVICE)
        train_wm(wm, demos[0])
        
        # No-trust BC
        std = MLP(obs_dim, act_dim).to(DEVICE)
        train_bc(std, demos[0])
        std_err = eval_bc(std, demos[0][3:] if len(demos[0]) > 3 else demos[0][-1:])
        results["none"].append(std_err)
        
        # EMA trust (single-step)
        tru = MLP(obs_dim, act_dim).to(DEVICE)
        all_obs = np.concatenate([d["obs"] for d in demos[0][:3]])
        all_acts = np.concatenate([d["acts"] for d in demos[0][:3]])
        obs_t = torch.tensor(all_obs, dtype=torch.float32).to(DEVICE)
        act_t = torch.tensor(all_acts, dtype=torch.float32).to(DEVICE)
        opt = torch.optim.Adam(tru.parameters(), lr=LR)
        tru.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(obs_t.size(0))
            for i in range(0, obs_t.size(0), BATCH):
                idx = perm[i:i+BATCH]
                pred = tru(obs_t[idx])
                with torch.no_grad():
                    pe = wm.predict_error(obs_t[idx], act_t[idx], obs_t[idx]).mean(dim=-1)
                    trust = torch.exp(-pe / (pe.mean() + 1e-8)).clamp(0, 1)
                    w = trust.clamp(0.1, 1.0)
                loss = (F.mse_loss(pred, act_t[idx], reduction="none").mean(dim=-1) * w).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        results["ema"].append(eval_bc(tru, demos[0][3:] if len(demos[0]) > 3 else demos[0][-1:]))
        
        # Multi-step trust (k=1,3,5)
        for k in [1, 3, 5]:
            tru = MLP(obs_dim, act_dim).to(DEVICE)
            train_bc_multistep_trust(tru, wm, demos[0], k=k)
            err = eval_bc(tru, demos[0][3:] if len(demos[0]) > 3 else demos[0][-1:])
            results[f"multi_k{k}"].append(err)
        
        elapsed = __import__('time').time() - t0
        print(f"  Seed {seed}: none={results['none'][-1]:.4f} "
              f"ema={results['ema'][-1]:.4f} "
              f"k1={results['multi_k1'][-1]:.4f} "
              f"k3={results['multi_k3'][-1]:.4f} "
              f"k5={results['multi_k5'][-1]:.4f} "
              f"({elapsed:.1f}s)")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary (mean ± std)")
    print(f"{'='*60}")
    for method in results:
        arr = np.array(results[method])
        print(f"  {method:<12}: {arr.mean():.4f} ± {arr.std():.4f}")
    
    # Save
    out = os.path.join(os.path.dirname(__file__), "jepa_multistep_trust.json")
    with open(out, "w") as f:
        json.dump({m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "seeds": v} 
                   for m, v in results.items()}, f, indent=2)
    print(f"\nSaved to {out}")
    
    return results


if __name__ == "__main__":
    run()
