import gymnasium as gym
"""Experiment 4: Trust-Guided Action Selection on ManiSkill StackCube
Reject actions with low trust, resample until trust exceeds threshold."""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import BACKBONES, make_trust

DEVICE = "cuda"
N_EPISODES = 30
STEPS_PER_EPISODE = 100
BATCH_SIZE = 32
LR = 1e-3
SEQ_LEN = 10
MAX_RESAMPLES = 5


def make_env():
    import mani_skill.envs
    env = gym.make("StackCube-v1", obs_mode="state", render_mode="rgb_array")
    return env


def train_world_model(wm, obs_seq, act_seq, epochs=20):
    optimizer = torch.optim.Adam(wm.parameters(), lr=LR)
    obs_t = torch.tensor(obs_seq, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(act_seq, dtype=torch.float32).to(DEVICE)
    wm.train()
    total_loss = 0
    n = 0
    for _ in range(epochs):
        perm = torch.randperm(obs_t.size(0))
        for i in range(0, obs_t.size(0), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            if len(idx) < 2:
                continue
            loss = wm.train_loss(obs_t[idx], act_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def select_action_trust_guided(wm, trust_scorer, state, threshold=0.3, method="resample"):
    """Select action with trust guidance."""
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    for attempt in range(MAX_RESAMPLES):
        action = np.random.randn(8) * 0.3
        action = np.clip(action, -1, 1)
        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_error = wm.predict_error(state_t, action_t, state_t)
            trust = trust_scorer.compute_trust(pred_error, 0).mean().item()
        
        if trust >= threshold:
            return action, trust, attempt + 1
    
    # If all resamples fail, return last action with low trust
    return action, trust, MAX_RESAMPLES


def evaluate_episode(wm, trust_scorer, env, threshold=0.3, use_trust=False):
    """Evaluate one episode."""
    obs, _ = env.reset()
    state = np.asarray(obs).flatten()[:48]
    
    total_reward = 0
    all_trust = []
    all_errors = []
    actions_taken = 0
    actions_rejected = 0
    
    for step in range(STEPS_PER_EPISODE):
        if use_trust:
            action, trust, resamples = select_action_trust_guided(
                wm, trust_scorer, state, threshold=threshold
            )
            all_trust.append(trust)
            actions_rejected += resamples - 1
        else:
            action = np.random.randn(8) * 0.3
            action = np.clip(action, -1, 1)
            # Still compute trust for comparison
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_error = wm.predict_error(state_t, action_t, state_t)
                trust = trust_scorer.compute_trust(pred_error, 0).mean().item()
                all_trust.append(trust)
        
        actions_taken += 1
        
        obs_new, reward, terminated, truncated, info = env.step(action)
        new_state = np.asarray(obs_new).flatten()[:48]
        
        state = new_state
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return {
        "reward": total_reward,
        "success": 1.0 if terminated and not truncated else 0.0,
        "avg_trust": np.mean(all_trust) if all_trust else 1.0,
        "trust_std": np.std(all_trust) if all_trust else 0.0,
        "actions_taken": actions_taken,
        "actions_rejected": actions_rejected,
        "rejection_rate": actions_rejected / max(actions_taken, 1),
    }


def run_experiment():
    print(f"\n{'='*60}")
    print(f"  Inference-Time Trust: StackCube")
    print(f"{'='*60}")
    
    env = make_env()
    obs_dim = 48  # ManiSkill StackCube state dim
    act_dim = 8
    
    wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    trust_scorer = make_trust("ema", obs_dim, act_dim)
    
    # Collect initial data
    print("Collecting initial data...")
    obs, _ = env.reset()
    obs_seq, act_seq = [], []
    
    for _ in range(200):
        state = np.asarray(obs).flatten()[:48]
        action = np.random.randn(8) * 0.3
        action = np.clip(action, -1, 1)
        obs_new, reward, terminated, truncated, info = env.step(action)
        obs_seq.append(state)
        act_seq.append(action)
        obs = obs_new
        if terminated or truncated:
            obs, _ = env.reset()
    
    obs_seq = np.array(obs_seq)
    act_seq = np.array(act_seq)
    
    # Train world model
    print("Training world model...")
    obs_padded = np.zeros((1, len(obs_seq), obs_dim))
    act_padded = np.zeros((1, len(obs_seq), act_dim))
    obs_padded[0, :len(obs_seq)] = obs_seq
    act_padded[0, :len(act_seq)] = act_seq
    train_world_model(wm, obs_padded, act_padded, epochs=30)
    print("World model trained.")
    
    # Evaluate different methods
    methods = {
        "random": {"use_trust": False, "threshold": 0.0},
        "trust_guided_0.3": {"use_trust": True, "threshold": 0.3},
        "trust_guided_0.5": {"use_trust": True, "threshold": 0.5},
        "trust_guided_0.7": {"use_trust": True, "threshold": 0.7},
    }
    
    results = {}
    for name, config in methods.items():
        print(f"\nEvaluating {name}...")
        episodes = []
        for ep in range(N_EPISODES):
            r = evaluate_episode(
                wm, trust_scorer, env,
                threshold=config["threshold"],
                use_trust=config["use_trust"]
            )
            episodes.append(r)
            
            if (ep + 1) % 10 == 0:
                avg_r = np.mean([e["reward"] for e in episodes[-10:]])
                avg_s = np.mean([e["success"] for e in episodes[-10:]])
                avg_t = np.mean([e["avg_trust"] for e in episodes[-10:]])
                print(f"  Episode {ep+1}/{N_EPISODES}: reward={avg_r:.3f}, success={avg_s:.3f}, trust={avg_t:.3f}")
        
        results[name] = {
            "episodes": episodes,
            "avg_reward": np.mean([e["reward"] for e in episodes]),
            "avg_success": np.mean([e["success"] for e in episodes]),
            "avg_trust": np.mean([e["avg_trust"] for e in episodes]),
            "trust_std": np.mean([e["trust_std"] for e in episodes]),
            "avg_rejection_rate": np.mean([e["rejection_rate"] for e in episodes]),
        }
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Success':>10} {'Reward':>10} {'Trust':>10} {'Reject%':>10}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<25} {r['avg_success']:>10.3f} {r['avg_reward']:>10.3f} {r['avg_trust']:>10.3f} {r['avg_rejection_rate']*100:>10.1f}%")
    
    # Calculate improvement
    baseline = results["random"]["avg_success"]
    for name, r in results.items():
        if name != "random":
            improvement = (r["avg_success"] - baseline) / max(baseline, 0.001) * 100
            print(f"\n{name} vs random: {improvement:+.1f}% success improvement")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="inference_trust_results.json")
    args = parser.parse_args()
    
    results = run_experiment()
    
    outpath = os.path.join(os.path.dirname(__file__), args.output)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")
