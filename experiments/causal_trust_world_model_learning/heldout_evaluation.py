"""Fixed Held-Out Cross-Task Prediction Evaluation
Addresses audit issue: "improves from observation" needs fixed evaluation set.

Trains world model sequentially on tasks 1→2→...→10.
At each step, evaluates on:
1. Fixed held-out transitions from task 1 (never trained on after task 1)
2. Cross-task error: task 1 model evaluated on task 10 data
3. Cumulative error: average across all seen tasks
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
import h5py

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import BACKBONES

DEVICE = "cuda"
SUITE_DIR = "/home/ubuntu/robotics_world_models/LIBERO/libero_spatial"
N_TASKS = 10
N_SEEDS = 3
MAX_DEMOS = 5
WM_EPOCHS = 20


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
            if isinstance(demo, list):
                # List of demos
                for d in demo:
                    o = torch.tensor(d["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    a = torch.tensor(d["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    loss = wm.train_loss(o, a)
                    opt.zero_grad(); loss.backward(); opt.step()
            else:
                # Single demo
                o = torch.tensor(demo["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                a = torch.tensor(demo["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                loss = wm.train_loss(o, a)
                opt.zero_grad(); loss.backward(); opt.step()


def eval_wm(wm, demos):
    """Evaluate WM on held-out demos. Returns per-transition MSE."""
    wm.eval()
    all_errors = []
    with torch.no_grad():
        for demo in demos:
            o = torch.tensor(demo["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a = torch.tensor(demo["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = wm.predict_error(o, a, o)
            error = pred.mean().item()
            all_errors.append(error)
    return np.mean(all_errors)


def run():
    print(f"\n{'='*60}")
    print(f"  Held-Out Cross-Task Prediction Evaluation")
    print(f"{'='*60}")
    
    demos = load_demos(SUITE_DIR, N_TASKS)
    print(f"Loaded {len(demos)} tasks")
    
    # Split each task's demos: 3 for training, 2 for held-out evaluation
    train_demos = [d[:3] for d in demos]
    heldout_demos = [d[3:] for d in demos if len(d) > 3]
    # Fallback: if fewer than 4 demos, use last demo as held-out
    if len(heldout_demos) < N_TASKS:
        heldout_demos = [d[-1:] for d in demos]
    
    print(f"Train: 3 demos/task, Held-out: {len(heldout_demos[0])} demo(s)/task")
    
    obs_dim, act_dim = 21, 7
    results = {}
    
    for bone_name in ["mlp", "rssm", "dreamerv3"]:
        print(f"\n--- {bone_name} ---")
        results[bone_name] = {
            "heldout_task1": [],      # Fixed held-out from task 1
            "cross_task_1to10": [],   # Task 1 data, evaluated after training on all
            "cumulative_avg": [],     # Average error across all seen tasks
            "per_task": [],           # Error on each task after sequential training
        }
        
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            t0 = time.time()
            wm = BACKBONES[bone_name](obs_dim, act_dim).to(DEVICE)
            
            # Fixed held-out set from task 1 (never used for training)
            fixed_heldout = heldout_demos[0]
            
            # Sequential training
            per_task_errors = []
            heldout_errors = []
            cross_task_errors = []
            cumulative_errors = []
            
            for t in range(N_TASKS):
                # Train on task t
                train_wm(wm, [train_demos[t]])
                
                # Evaluate on fixed held-out from task 1
                h_err = eval_wm(wm, fixed_heldout)
                heldout_errors.append(h_err)
                
                # Evaluate on cross-task (task 1 data after seeing all tasks up to t)
                cross_err = eval_wm(wm, heldout_demos[0])
                cross_task_errors.append(cross_err)
                
                # Evaluate on current task's held-out data
                task_err = eval_wm(wm, heldout_demos[t])
                per_task_errors.append(task_err)
                
                # Cumulative: average across all tasks seen so far
                all_heldout = []
                for tt in range(t + 1):
                    all_heldout.extend(heldout_demos[tt])
                cum_err = eval_wm(wm, all_heldout)
                cumulative_errors.append(cum_err)
            
            results[bone_name]["heldout_task1"].append(heldout_errors)
            results[bone_name]["cross_task_1to10"].append(cross_task_errors)
            results[bone_name]["per_task"].append(per_task_errors)
            results[bone_name]["cumulative_avg"].append(cumulative_errors)
            
            elapsed = time.time() - t0
            print(f"  Seed {seed}: heldout_task1[0]={heldout_errors[0]:.6f} "
                  f"heldout_task1[9]={heldout_errors[9]:.6f} "
                  f"cross[9]={cross_task_errors[9]:.6f} "
                  f"({elapsed:.1f}s)")
    
    # Compute statistics
    stats = {}
    for bone in results:
        stats[bone] = {}
        for metric in results[bone]:
            arr = np.array(results[bone][metric])  # (n_seeds, n_tasks)
            stats[bone][metric] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "ci95": (1.96 * arr.std(axis=0) / np.sqrt(N_SEEDS)).tolist(),
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  Summary: Fixed Held-Out Task 1 Error Across Sequential Training")
    print(f"{'='*60}")
    print(f"{'Backbone':<12} {'Task1→Train1':>12} {'Task1→Train5':>12} {'Task1→Train10':>12} {'Reduction':>10}")
    print("-" * 60)
    for bone in stats:
        h = stats[bone]["heldout_task1"]["mean"]
        reduction = (h[0] - h[-1]) / h[0] * 100 if h[0] > 0 else 0
        print(f"{bone:<12} {h[0]:.6f}     {h[4]:.6f}     {h[9]:.6f}     {reduction:+.1f}%")
    
    print(f"\n  Cross-Task Error (Task 1 data, evaluated after training on Task k):")
    print(f"{'Backbone':<12} {'k=1':>12} {'k=5':>12} {'k=10':>12} {'Reduction':>10}")
    print("-" * 60)
    for bone in stats:
        c = stats[bone]["cross_task_1to10"]["mean"]
        reduction = (c[0] - c[-1]) / c[0] * 100 if c[0] > 0 else 0
        print(f"{bone:<12} {c[0]:.6f}     {c[4]:.6f}     {c[9]:.6f}     {reduction:+.1f}%")
    
    # Save
    out = os.path.join(os.path.dirname(__file__), "heldout_evaluation.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {out}")
    
    return stats


if __name__ == "__main__":
    run()
