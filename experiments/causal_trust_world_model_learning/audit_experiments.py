"""Audit Improvements: Error Bars, Task-Order Sensitivity, Trust Threshold Ablation
Runs on LIBERO-Spatial (fastest suite) with RSSM backbone."""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from itertools import permutations

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import BACKBONES, make_trust

DEVICE = "cuda"
LIBERO_DIR = "/home/ubuntu/robotics_world_models/LIBERO/libero_spatial"
N_TASKS = 10
BATCH = 64
LR = 1e-3
EPOCHS = 50
MAX_DEMOS = 5
N_SEEDS = 3


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


def train_bc_trust(policy, wm, demos, threshold=0.3, epochs=EPOCHS):
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
                trust = torch.exp(-pe.mean(dim=-1) / (pe.mean().item() + 1e-8)).clamp(0, 1)
                # Apply threshold: downweight samples with low trust
                w = torch.where(trust >= threshold, trust, torch.tensor(0.1).to(DEVICE))
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


def run_experiment_1_error_bars(demos):
    """Experiment 1: Error bars for main sweep (3 seeds)."""
    print(f"\n{'='*60}")
    print(f"  Experiment 1: Error Bars (3 seeds)")
    print(f"{'='*60}")
    
    obs_dim, act_dim = 21, 7
    results = {"std": [], "trust": []}
    
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Train WM
        wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
        train_wm(wm, demos[0])
        
        seed_std_errors = []
        seed_trust_errors = []
        
        for t in range(N_TASKS):
            train_demos = demos[t][:3]
            test_demos = demos[t][3:] if len(demos[t]) > 3 else demos[t][-1:]
            
            # Standard BC
            std = MLP(obs_dim, act_dim).to(DEVICE)
            train_bc(std, train_demos)
            std_err = eval_bc(std, test_demos)
            seed_std_errors.append(std_err)
            
            # Trust BC
            tru = MLP(obs_dim, act_dim).to(DEVICE)
            train_bc_trust(tru, wm, train_demos)
            tru_err = eval_bc(tru, test_demos)
            seed_trust_errors.append(tru_err)
        
        results["std"].append(seed_std_errors)
        results["trust"].append(seed_trust_errors)
        
        std_mean = np.mean(seed_std_errors)
        tru_mean = np.mean(seed_trust_errors)
        imp = (std_mean - tru_mean) / std_mean * 100
        print(f"  Seed {seed}: std={std_mean:.4f} trust={tru_mean:.4f} ({imp:+.1f}%)")
    
    # Compute statistics
    std_arr = np.array(results["std"])
    trust_arr = np.array(results["trust"])
    
    stats = {
        "std_mean": float(std_arr.mean()),
        "std_std": float(std_arr.std()),
        "std_ci95": float(1.96 * std_arr.std() / np.sqrt(N_SEEDS)),
        "trust_mean": float(trust_arr.mean()),
        "trust_std": float(trust_arr.std()),
        "trust_ci95": float(1.96 * trust_arr.std() / np.sqrt(N_SEEDS)),
        "improvement": float((std_arr.mean() - trust_arr.mean()) / std_arr.mean() * 100),
        "per_task_std_mean": std_arr.mean(axis=0).tolist(),
        "per_task_std_std": std_arr.std(axis=0).tolist(),
        "per_task_trust_mean": trust_arr.mean(axis=0).tolist(),
        "per_task_trust_std": trust_arr.std(axis=0).tolist(),
    }
    
    print(f"\n  Summary: std={stats['std_mean']:.4f}±{stats['std_std']:.4f} "
          f"trust={stats['trust_mean']:.4f}±{stats['trust_std']:.4f} "
          f"({stats['improvement']:+.1f}%)")
    print(f"  95% CI: std=[{stats['std_mean']-stats['std_ci95']:.4f}, {stats['std_mean']+stats['std_ci95']:.4f}]")
    print(f"          trust=[{stats['trust_mean']-stats['trust_ci95']:.4f}, {stats['trust_mean']+stats['trust_ci95']:.4f}]")
    
    return stats


def run_experiment_2_task_order(demos):
    """Experiment 2: Task-order sensitivity (3 orderings)."""
    print(f"\n{'='*60}")
    print(f"  Experiment 2: Task-Order Sensitivity")
    print(f"{'='*60}")
    
    obs_dim, act_dim = 21, 7
    orderings = [
        list(range(N_TASKS)),  # Original
        list(reversed(range(N_TASKS))),  # Reversed
        np.random.permutation(N_TASKS).tolist(),  # Random
    ]
    
    results = []
    for oidx, order in enumerate(orderings):
        print(f"\n  Ordering {oidx}: {order[:5]}...")
        
        wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
        
        # Sequential training with trust
        task_errors = []
        for t_idx, t in enumerate(order):
            # Train WM on this task
            train_wm(wm, demos[t])
            
            # Evaluate
            test_demos = demos[t][3:] if len(demos[t]) > 3 else demos[t][-1:]
            with torch.no_grad():
                o = torch.tensor(test_demos[0]["obs"][:50], dtype=torch.float32).to(DEVICE)
                a = torch.tensor(test_demos[0]["acts"][:50], dtype=torch.float32).to(DEVICE)
                wm_e = wm.predict_error(o, a, o).mean().item()
            task_errors.append(wm_e)
        
        # Compute forgetting: error on task 0 after training on all tasks
        with torch.no_grad():
            o = torch.tensor(demos[order[0]][0]["obs"][:50], dtype=torch.float32).to(DEVICE)
            a = torch.tensor(demos[order[0]][0]["acts"][:50], dtype=torch.float32).to(DEVICE)
            final_error = wm.predict_error(o, a, o).mean().item()
        
        initial_error = task_errors[0]
        forgetting = (final_error - initial_error) / initial_error * 100
        
        results.append({
            "ordering": order,
            "task_errors": task_errors,
            "initial_error": initial_error,
            "final_error": final_error,
            "forgetting": forgetting,
            "mean_error": np.mean(task_errors),
        })
        
        print(f"    initial={initial_error:.6f} final={final_error:.6f} forgetting={forgetting:+.1f}%")
    
    # Summary
    forgettings = [r["forgetting"] for r in results]
    mean_errors = [r["mean_error"] for r in results]
    
    stats = {
        "mean_forgetting": float(np.mean(forgettings)),
        "std_forgetting": float(np.std(forgettings)),
        "mean_error": float(np.mean(mean_errors)),
        "std_error": float(np.std(mean_errors)),
        "orderings": results,
    }
    
    print(f"\n  Summary: forgetting={stats['mean_forgetting']:.1f}%±{stats['std_forgetting']:.1f}%")
    
    return stats


def run_experiment_3_threshold(demos):
    """Experiment 3: Trust threshold ablation."""
    print(f"\n{'='*60}")
    print(f"  Experiment 3: Trust Threshold Ablation")
    print(f"{'='*60}")
    
    obs_dim, act_dim = 21, 7
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Train WM
    wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    train_wm(wm, demos[0])
    
    results = {}
    for theta in thresholds:
        errors = []
        for t in range(N_TASKS):
            train_demos = demos[t][:3]
            test_demos = demos[t][3:] if len(demos[t]) > 3 else demos[t][-1:]
            
            policy = MLP(obs_dim, act_dim).to(DEVICE)
            train_bc_trust(policy, wm, train_demos, threshold=theta)
            err = eval_bc(policy, test_demos)
            errors.append(err)
        
        results[str(theta)] = {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "per_task": errors,
        }
        print(f"  θ={theta:.1f}: {np.mean(errors):.4f}±{np.std(errors):.4f}")
    
    # Find best threshold
    best_theta = min(results.keys(), key=lambda k: results[k]["mean"])
    print(f"\n  Best threshold: θ={best_theta} ({results[best_theta]['mean']:.4f})")
    
    return {"thresholds": results, "best": best_theta}


def run():
    print(f"\n{'='*60}")
    print(f"  Audit Experiments")
    print(f"{'='*60}")
    
    demos = load_demos(LIBERO_DIR, N_TASKS)
    print(f"Loaded {len(demos)} tasks")
    
    all_results = {}
    
    # Run experiments
    all_results["error_bars"] = run_experiment_1_error_bars(demos)
    all_results["task_order"] = run_experiment_2_task_order(demos)
    all_results["threshold"] = run_experiment_3_threshold(demos)
    
    # Save
    out = os.path.join(os.path.dirname(__file__), "audit_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")
    
    return all_results


if __name__ == "__main__":
    run()
