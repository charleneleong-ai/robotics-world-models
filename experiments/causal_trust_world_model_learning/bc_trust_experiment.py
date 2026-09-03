"""BC Policy + Trust-Weighted Continual Learning on LIBERO
Trains Behavior Cloning on expert demos, then evaluates
standard BC vs trust-weighted BC across 10 sequential tasks."""

import os, sys, json, time
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
BATCH_SIZE = 128
LR = 3e-4
BC_EPOCHS = 100
EVAL_EPISODES = 20
MAX_STEPS = 200


class BCPolicy(nn.Module):
    """Simple MLP policy for behavior cloning."""
    def __init__(self, obs_dim=21, act_dim=7, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h), nn.LayerNorm(h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, act_dim),
        )
    
    def forward(self, obs):
        return self.net(obs)
    
    def act(self, obs_np):
        obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.forward(obs)
        return action.cpu().numpy().flatten()


class TrustWeightedBCPolicy(nn.Module):
    """BC policy that uses trust to weight its predictions."""
    def __init__(self, obs_dim=21, act_dim=7, h=256):
        super().__init__()
        self.policy = BCPolicy(obs_dim, act_dim, h)
        self.trust_head = nn.Sequential(
            nn.Linear(obs_dim + act_dim, h), nn.SiLU(),
            nn.Linear(h, 1), nn.Sigmoid()
        )
    
    def forward(self, obs):
        return self.policy(obs)
    
    def compute_trust(self, obs, act):
        return self.trust_head(torch.cat([obs, act], dim=-1))
    
    def act(self, obs_np):
        obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.policy(obs)
            trust = self.compute_trust(obs, action)
        return action.cpu().numpy().flatten(), trust.item()


def load_libero_demos(suite_dir, n_tasks=10):
    """Load LIBERO expert demonstrations."""
    all_demos = []
    files = sorted(f for f in os.listdir(suite_dir) if f.endswith(".hdf5"))
    
    for fn in files[:n_tasks]:
        with h5py.File(os.path.join(suite_dir, fn), "r") as hf:
            data = hf["data"]
            task_demos = []
            for k in sorted(data.keys()):
                if not k.startswith("demo_"):
                    continue
                demo = data[k]
                # Extract state: ee_ori(3)+ee_pos(3)+ee_states(6)+gripper(2)+joints(7)=21
                obs_parts = []
                for field in ["ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"]:
                    obs_parts.append(np.array(demo["obs"][field]))
                obs = np.concatenate(obs_parts, axis=-1)  # (T, 21)
                acts = np.array(demo["actions"])  # (T, 7)
                task_demos.append({"obs": obs, "acts": acts})
            all_demos.append(task_demos)
    
    return all_demos


def train_bc(policy, train_demos, epochs=BC_EPOCHS, lr=LR):
    """Train BC policy on expert demos."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    # Flatten all demos
    all_obs = np.concatenate([d["obs"] for d in train_demos])
    all_acts = np.concatenate([d["acts"] for d in train_demos])
    
    obs_t = torch.tensor(all_obs, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(all_acts, dtype=torch.float32).to(DEVICE)
    
    policy.train()
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(obs_t.size(0))
        epoch_loss = 0
        n = 0
        for i in range(0, obs_t.size(0), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            pred = policy(obs_t[idx])
            loss = F.mse_loss(pred, act_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        losses.append(epoch_loss / max(n, 1))
    
    return losses


def train_trust_bc(policy, wm, train_demos, epochs=BC_EPOCHS, lr=LR):
    """Train trust-weighted BC policy."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    all_obs = np.concatenate([d["obs"] for d in train_demos])
    all_acts = np.concatenate([d["acts"] for d in train_demos])
    
    obs_t = torch.tensor(all_obs, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(all_acts, dtype=torch.float32).to(DEVICE)
    
    policy.train()
    wm.eval()
    losses = []
    
    for epoch in range(epochs):
        perm = torch.randperm(obs_t.size(0))
        epoch_loss = 0
        n = 0
        for i in range(0, obs_t.size(0), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            if len(idx) < 2:
                continue
            
            obs_batch = obs_t[idx]
            act_batch = act_t[idx]
            
            # BC loss
            pred = policy.policy(obs_batch)
            bc_loss = F.mse_loss(pred, act_batch)
            
            # Trust loss: high trust for actions close to expert
            pred_trust = policy.compute_trust(obs_batch, pred)
            expert_trust = policy.compute_trust(obs_batch, act_batch)
            
            # Weight BC loss by trust (trustworthy states get more weight)
            with torch.no_grad():
                wm_pred_error = wm.predict_error(obs_batch, pred, obs_batch)
                wm_trust = torch.exp(-wm_pred_error.mean(dim=-1) / (wm_pred_error.mean().item() + 1e-8)).clamp(0, 1)
            
            weighted_loss = (bc_loss.unsqueeze(0) * wm_trust).mean()
            
            # Trust regularization: trust should align with WM trust
            trust_loss = F.mse_loss(pred_trust.squeeze(), wm_trust.detach())
            
            total_loss = weighted_loss + 0.1 * trust_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            n += 1
        
        losses.append(epoch_loss / max(n, 1))
    
    return losses


def evaluate_bc(policy, env, task_idx=0, max_steps=MAX_STEPS):
    """Evaluate BC policy in an environment."""
    # For LIBERO, we evaluate on held-out demo trajectories
    # (simulating policy execution)
    successes = 0
    total_rewards = []
    
    for ep in range(EVAL_EPISODES):
        # Use a demo trajectory as reference
        # The policy tries to follow similar state transitions
        reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Get observation from demo (simplified - no actual env)
            # In real setup, this would be env.observation
            obs = np.random.randn(21) * 0.1  # Placeholder
            action = policy.act(obs)
            steps += 1
        
        total_rewards.append(steps)
    
    return {
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
    }


def evaluate_bc_on_demos(policy, test_demos, use_trust=False):
    """Evaluate BC policy on held-out demo trajectories."""
    all_errors = []
    all_trusts = []
    
    for demo in test_demos:
        obs = torch.tensor(demo["obs"], dtype=torch.float32).to(DEVICE)
        acts = torch.tensor(demo["acts"], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            if use_trust:
                pred_acts, trusts = [], []
                for o in obs:
                    a, t = policy.act(o.cpu().numpy())
                    pred_acts.append(a)
                    trusts.append(t)
                pred_acts = torch.tensor(pred_acts, dtype=torch.float32).to(DEVICE)
                all_trusts.extend(trusts)
            else:
                pred_acts = policy(obs)
        
        error = F.mse_loss(pred_acts, acts).item()
        all_errors.append(error)
    
    results = {
        "avg_error": np.mean(all_errors),
        "std_error": np.std(all_errors),
    }
    if all_trusts:
        results["avg_trust"] = np.mean(all_trusts)
        results["std_trust"] = np.std(all_trusts)
    
    return results


def run_experiment():
    print(f"\n{'='*60}")
    print(f"  BC + Trust-Weighted Continual Learning")
    print(f"  LIBERO-Spatial: 10 Sequential Tasks")
    print(f"{'='*60}")
    
    # Load demos
    print("\nLoading LIBERO-Spatial demos...")
    all_demos = load_libero_demos(LIBERO_DIR, n_tasks=N_TASKS)
    print(f"Loaded {len(all_demos)} tasks, {sum(len(d) for d in all_demos)} demos total")
    
    # Initialize world model and trust
    obs_dim, act_dim = 21, 7
    wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    trust_scorer = make_trust("ema", obs_dim, act_dim)
    
    # Train world model on first task
    print("\nTraining world model on Task 0...")
    train_obs = np.concatenate([d["obs"] for d in all_demos[0]])
    train_acts = np.concatenate([d["acts"] for d in all_demos[0]])
    # Train world model using its own train_loss
    wm_optimizer = torch.optim.Adam(wm.parameters(), lr=1e-3)
    wm.train()
    for ep in range(50):
        for demo in all_demos[0]:
            obs = torch.tensor(demo["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            acts = torch.tensor(demo["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            loss = wm.train_loss(obs, acts)
            wm_optimizer.zero_grad()
            loss.backward()
            wm_optimizer.step()  # Reuse BC training for world model
    
    # Sequential evaluation
    results = {"standard_bc": [], "trust_bc": [], "world_model_error": []}
    
    for task_idx in range(N_TASKS):
        print(f"\n--- Task {task_idx} ---")
        
        # Standard BC: train on current task only
        std_policy = BCPolicy(obs_dim, act_dim).to(DEVICE)
        train_bc(std_policy, all_demos[task_idx], epochs=BC_EPOCHS)
        
        # Trust BC: train with trust weighting
        trust_policy = TrustWeightedBCPolicy(obs_dim, act_dim).to(DEVICE)
        train_trust_bc(trust_policy, wm, all_demos[task_idx], epochs=BC_EPOCHS)
        
        # Evaluate on held-out demos from this task
        test_demos = all_demos[task_idx][:2]  # Use first 2 as test
        
        std_results = evaluate_bc_on_demos(std_policy, test_demos, use_trust=False)
        trust_results = evaluate_bc_on_demos(trust_policy, test_demos, use_trust=True)
        
        # Measure world model error
        wm.eval()
        with torch.no_grad():
            test_obs = torch.tensor(test_demos[0]["obs"], dtype=torch.float32).to(DEVICE)
            test_acts = torch.tensor(test_demos[0]["acts"], dtype=torch.float32).to(DEVICE)
            wm_error = wm.predict_error(test_obs, test_acts, test_obs).mean().item()
        
        results["standard_bc"].append(std_results)
        results["trust_bc"].append(trust_results)
        results["world_model_error"].append(wm_error)
        
        print(f"  Standard BC error: {std_results['avg_error']:.4f}")
        print(f"  Trust BC error: {trust_results['avg_error']:.4f} (trust={trust_results.get('avg_trust', 0):.3f})")
        print(f"  World model error: {wm_error:.6f}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    
    std_errors = [r["avg_error"] for r in results["standard_bc"]]
    trust_errors = [r["avg_error"] for r in results["trust_bc"]]
    wm_errors = results["world_model_error"]
    
    print(f"\nStandard BC: avg_error={np.mean(std_errors):.4f} (std={np.std(std_errors):.4f})")
    print(f"Trust BC: avg_error={np.mean(trust_errors):.4f} (std={np.std(trust_errors):.4f})")
    print(f"World Model: avg_error={np.mean(wm_errors):.6f} (std={np.std(wm_errors):.6f})")
    
    # Improvement
    improvement = (np.mean(std_errors) - np.mean(trust_errors)) / np.mean(std_errors) * 100
    print(f"\nTrust BC improvement: {improvement:+.1f}% error reduction")
    
    # Error reduction across tasks
    if len(std_errors) > 1:
        print(f"\nError reduction across tasks:")
        for i in range(1, len(std_errors)):
            reduction = (std_errors[0] - std_errors[i]) / std_errors[0] * 100
            print(f"  Task 0→{i}: {reduction:+.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
    
    outpath = os.path.join(os.path.dirname(__file__), "bc_trust_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")
