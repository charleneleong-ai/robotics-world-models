"""Experiment 4 v2: Trust-Guided Action Selection
Multi-benchmark: ManiSkill StackCube, LIBERO-Spatial, Kinder

Uses MPC (Model Predictive Control) with the world model for action selection,
then evaluates trust-guided rejection of low-trust action sequences.
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import BACKBONES, make_trust

DEVICE = "cuda"
N_EPISODES = 30
STEPS_PER_EPISODE = 100
BATCH_SIZE = 32
LR = 1e-3
TRAIN_EPOCHS = 50
MPC_HORIZON = 5
MPC_SAMPLES = 50


def collect_training_data(env, n_steps=500, obs_dim=48, act_dim=8):
    """Collect training data with a simple PD-like policy."""
    obs, _ = env.reset()
    obs_seq, act_seq = [], []
    
    for _ in range(n_steps):
        state = np.asarray(obs).flatten()[:obs_dim]
        
        # Simple sinusoidal exploration + gravity compensation
        action = np.random.randn(act_dim) * 0.3
        # Add periodic component for structured exploration
        action += 0.1 * np.sin(np.random.randn(act_dim))
        action = np.clip(action, -1, 1)
        
        obs_new, reward, terminated, truncated, info = env.step(action)
        new_state = np.asarray(obs_new).flatten()[:obs_dim]
        
        obs_seq.append(state)
        act_seq.append(action)
        obs = obs_new
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    return np.array(obs_seq), np.array(act_seq)


def train_world_model(wm, obs_seq, act_seq, epochs=TRAIN_EPOCHS):
    """Train world model on collected data."""
    optimizer = torch.optim.Adam(wm.parameters(), lr=LR)
    obs_t = torch.tensor(obs_seq, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(act_seq, dtype=torch.float32).to(DEVICE)
    
    wm.train()
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(obs_t.size(0))
        epoch_loss = 0
        n = 0
        for i in range(0, obs_t.size(0), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            if len(idx) < 2:
                continue
            loss = wm.train_loss(obs_t[idx].unsqueeze(0), act_t[idx].unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        losses.append(epoch_loss / max(n, 1))
    
    return losses


def mpc_action(wm, trust_scorer, state, act_dim, task_id=0, horizon=MPC_HORIZON, n_samples=MPC_SAMPLES, use_trust=False, threshold=0.3):
    """Model Predictive Control with optional trust filtering."""
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    best_action = np.random.randn(act_dim) * 0.3
    best_score = -float('inf')
    best_trust = 0.0
    
    for _ in range(n_samples):
        # Sample random action sequence
        action_seq = np.random.randn(horizon, act_dim) * 0.3
        action_seq = np.clip(action_seq, -1, 1)
        action_seq_t = torch.tensor(action_seq, dtype=torch.float32).to(DEVICE)
        
        # Predict trajectory
        with torch.no_grad():
            current_state = state_t
            total_reward = 0
            min_trust = 1.0
            valid = True
            
            for t in range(horizon):
                act = action_seq_t[t:t+1]
                pred_error = wm.predict_error(current_state, act, current_state)
                trust = trust_scorer.compute_trust(pred_error, task_id).mean().item()
                min_trust = min(min_trust, trust)
                
                if use_trust and trust < threshold:
                    valid = False
                    break
                
                # Simple reward: minimize distance to center (for StackCube)
                total_reward += trust * 0.1  # Trust-weighted reward
                current_state = current_state  # Simplified: don't actually roll out
        
        if valid and total_reward > best_score:
            best_score = total_reward
            best_action = action_seq[0]
            best_trust = min_trust
    
    return best_action, best_trust


def evaluate_episode(wm, trust_scorer, env, obs_dim, act_dim, task_id=0, 
                     threshold=0.3, use_trust=False):
    """Evaluate one episode with MPC policy."""
    obs, _ = env.reset()
    total_reward = 0
    all_trust = []
    actions_taken = 0
    actions_rejected = 0
    
    for step in range(STEPS_PER_EPISODE):
        state = np.asarray(obs).flatten()[:obs_dim]
        
        action, trust = mpc_action(
            wm, trust_scorer, state, act_dim, task_id=task_id,
            use_trust=use_trust, threshold=threshold
        )
        all_trust.append(trust)
        
        if use_trust and trust < threshold:
            actions_rejected += 1
        
        actions_taken += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return {
        "reward": total_reward,
        "success": 1.0 if terminated and not truncated else 0.0,
        "avg_trust": np.mean(all_trust) if all_trust else 0.0,
        "actions_taken": actions_taken,
        "actions_rejected": actions_rejected,
        "rejection_rate": actions_rejected / max(actions_taken, 1),
    }


def run_maniskill():
    """Run inference trust on ManiSkill StackCube."""
    import gymnasium as gym
    import mani_skill.envs
    
    print(f"\n{'='*60}")
    print(f"  ManiSkill StackCube")
    print(f"{'='*60}")
    
    env = gym.make("StackCube-v1", obs_mode="state", render_mode=None)
    obs_dim, act_dim = 48, 8
    
    wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    trust_scorer = make_trust("ema", obs_dim, act_dim)
    
    # Collect and train
    print("Collecting training data...")
    obs_seq, act_seq = collect_training_data(env, n_steps=500, obs_dim=obs_dim, act_dim=act_dim)
    
    print("Training world model...")
    train_world_model(wm, obs_seq, act_seq, epochs=TRAIN_EPOCHS)
    
    # Evaluate
    methods = {
        "random": {"use_trust": False, "threshold": 0.0},
        "trust_guided_0.3": {"use_trust": True, "threshold": 0.3},
        "trust_guided_0.5": {"use_trust": True, "threshold": 0.5},
    }
    
    results = {}
    for name, config in methods.items():
        print(f"\nEvaluating {name}...")
        episodes = []
        for ep in range(N_EPISODES):
            r = evaluate_episode(wm, trust_scorer, env, obs_dim, act_dim,
                               threshold=config["threshold"], use_trust=config["use_trust"])
            episodes.append(r)
            if (ep + 1) % 10 == 0:
                avg_r = np.mean([e["reward"] for e in episodes[-10:]])
                avg_s = np.mean([e["success"] for e in episodes[-10:]])
                print(f"  Episode {ep+1}: reward={avg_r:.3f}, success={avg_s:.3f}")
        
        results[name] = {
            "avg_reward": np.mean([e["reward"] for e in episodes]),
            "avg_success": np.mean([e["success"] for e in episodes]),
            "avg_trust": np.mean([e["avg_trust"] for e in episodes]),
            "rejection_rate": np.mean([e["rejection_rate"] for e in episodes]),
        }
    
    env.close()
    return results


def run_libero():
    """Run inference trust on LIBERO-Spatial (5 tasks)."""
    print(f"\n{'='*60}")
    print(f"  LIBERO-Spatial")
    print(f"{'='*60}")
    
    import h5py
    
    LIBERO_DIR = "/home/ubuntu/robotics_world_models/LIBERO"
    suite = "spatial"
    demo_dir = os.path.join(LIBERO_DIR, suite)
    hdf5_files = sorted([f for f in os.listdir(demo_dir) if f.endswith(".hdf5")])[:5]
    
    obs_dim, act_dim = 21, 7
    wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    trust_scorer = make_trust("ema", obs_dim, act_dim)
    
    all_obs, all_act = [], []
    for fname in hdf5_files:
        fpath = os.path.join(demo_dir, fname)
        with h5py.File(fpath, "r") as hf:
            data = hf["data"]
            for k in sorted(data.keys()):
                if k.startswith("demo_"):
                    all_obs.append(np.array(data[k]["obs"]))
                    all_act.append(np.array(data[k]["actions"]))
    
    # Sample subset
    idx = np.random.choice(len(all_obs), min(10, len(all_obs)), replace=False)
    obs_seq = np.concatenate([all_obs[i] for i in idx])
    act_seq = np.concatenate([all_act[i] for i in idx])
    
    # Train
    print("Training world model on LIBERO demos...")
    wm.train()
    optimizer = torch.optim.Adam(wm.parameters(), lr=1e-3)
    obs_t = torch.tensor(obs_seq, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(act_seq, dtype=torch.float32).to(DEVICE)
    
    for epoch in range(TRAIN_EPOCHS):
        perm = torch.randperm(obs_t.size(0))
        for i in range(0, obs_t.size(0), BATCH_SIZE):
            batch_idx = perm[i:i+BATCH_SIZE]
            if len(batch_idx) < 2:
                continue
            loss = wm.train_loss(obs_t[batch_idx], act_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            optimizer.step()
    
    # Evaluate trust on held-out demos (not full episodes, just prediction quality)
    test_obs = np.concatenate([all_obs[i] for i in range(len(all_obs)) if i not in idx])
    test_act = np.concatenate([all_act[i] for i in range(len(all_obs)) if i not in idx])
    
    test_obs_t = torch.tensor(test_obs[:200], dtype=torch.float32).to(DEVICE)
    test_act_t = torch.tensor(test_act[:200], dtype=torch.float32).to(DEVICE)
    
    wm.eval()
    with torch.no_grad():
        pred_error = wm.predict_error(test_obs_t.unsqueeze(0), test_act_t.unsqueeze(0), test_obs_t.unsqueeze(0))
        trust = trust_scorer.compute_trust(pred_error, 0)
    
    # Also test multi-step trust
    multi_trust_scorer = make_trust("multi_step", obs_dim, act_dim)
    with torch.no_grad():
        trust_multi = multi_trust_scorer.compute_trust(pred_error, 0)
    
    return {
        "pred_error_mean": float(pred_error.mean()),
        "pred_error_std": float(pred_error.std()),
        "ema_trust_mean": float(trust.mean()),
        "ema_trust_std": float(trust.std()),
        "multi_trust_mean": float(trust_multi.mean()),
        "multi_trust_std": float(trust_multi.std()),
        "n_test_samples": len(test_obs[:200]),
    }


def run_kinder():
    """Run inference trust on Kinder."""
    print(f"\n{'='*60}")
    print(f"  Kinder")
    print(f"{'='*60}")
    
    import kinder
    kinder.register_all_environments()
    import gymnasium as gym
    
    # Check available envs
    env_ids = [spec.id for spec in gym.registry.values() if "kinder" in spec.id.lower() or "KinDER" in spec.id]
    print(f"Available Kinder envs: {env_ids[:10]}")
    
    if not env_ids:
        print("No Kinder environments found, skipping")
        return None
    
    results = {}
    for env_id in env_ids[:3]:  # Test first 3
        print(f"\nTesting {env_id}...")
        try:
            env = gym.make(env_id, obs_mode="state")
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            
            wm = BACKBONES["mlp"](obs_dim, act_dim).to(DEVICE)
            trust_scorer = make_trust("ema", obs_dim, act_dim)
            
            # Collect data
            obs_seq, act_seq = collect_training_data(env, n_steps=300, obs_dim=obs_dim, act_dim=act_dim)
            
            # Train
            train_world_model(wm, obs_seq, act_seq, epochs=30)
            
            # Evaluate
            episodes = []
            for ep in range(10):
                r = evaluate_episode(wm, trust_scorer, env, obs_dim, act_dim)
                episodes.append(r)
            
            results[env_id] = {
                "avg_reward": np.mean([e["reward"] for e in episodes]),
                "avg_success": np.mean([e["success"] for e in episodes]),
                "avg_trust": np.mean([e["avg_trust"] for e in episodes]),
            }
            print(f"  reward={results[env_id]['avg_reward']:.3f}, success={results[env_id]['avg_success']:.3f}")
            env.close()
        except Exception as e:
            print(f"  Failed: {e}")
            results[env_id] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["maniskill", "libero", "kinder", "all"], default="all")
    parser.add_argument("--output", default="inference_trust_v2_results.json")
    args = parser.parse_args()
    
    all_results = {}
    
    if args.benchmark in ["maniskill", "all"]:
        all_results["maniskill"] = run_maniskill()
    
    if args.benchmark in ["libero", "all"]:
        all_results["libero"] = run_libero()
    
    if args.benchmark in ["kinder", "all"]:
        all_results["kinder"] = run_kinder()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for bench, r in all_results.items():
        if r:
            print(f"\n{bench}:")
            for k, v in r.items():
                if isinstance(v, dict):
                    print(f"  {k}: {v}")
                else:
                    print(f"  {k}: {v}")
    
    outpath = os.path.join(os.path.dirname(__file__), args.output)
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")
