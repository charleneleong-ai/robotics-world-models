"""Experiment 1: Trust-Guided Exploration on ManiSkill
When trust is low, trigger targeted exploration to collect informative data.
Compare: random exploration vs trust-guided exploration."""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import BACKBONES, make_trust

DEVICE = "cuda"
N_EPISODES = 20
STEPS_PER_EPISODE = 100
TRUST_THRESHOLD = 0.3
EXPLORATION_STEPS = 10
BATCH_SIZE = 32
LR = 1e-3
SEQ_LEN = 10


def make_env():
    import mani_skill.envs
    from mani_skill.utils.wrappers import RecordEpisode
    env = mani_skill.envs.create("PushCube-v1", obs_mode="state", render_mode="rgb_array")
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


def evaluate(wm, trust_scorer, env, mode="random"):
    """Evaluate one episode. mode='random' or 'trust_guided'."""
    obs, _ = env.reset()
    state = obs["obs"].cpu().numpy().flatten() if hasattr(obs["obs"], "cpu") else obs["obs"].flatten()
    
    total_reward = 0
    all_trust = []
    all_errors = []
    hidden = None
    prev_state = None
    exploration_remaining = 0
    
    for step in range(STEPS_PER_EPISODE):
        obs_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_error = wm.predict_error(obs_t, torch.zeros(1, 7, device=DEVICE), obs_t)
            err_val = pred_error.mean().item()
            all_errors.append(err_val)
            
            trust = 1.0
            if trust_scorer is not None:
                trust = trust_scorer.compute_trust(pred_error, 0).mean().item()
            all_trust.append(trust)
        
        # Decision: explore or exploit?
        if mode == "trust_guided":
            if trust < TRUST_THRESHOLD:
                exploration_remaining = EXPLORATION_STEPS
            if exploration_remaining > 0:
                action = np.random.randn(7) * 0.5  # wider exploration
                exploration_remaining -= 1
            else:
                action = np.random.randn(7) * 0.1  # narrow exploitation
        else:
            action = np.random.randn(7) * 0.3  # random baseline
        
        action = np.clip(action, -1, 1)
        
        obs_new, reward, terminated, truncated, info = env.step(action)
        new_state = obs_new["obs"].cpu().numpy().flatten() if hasattr(obs_new["obs"], "cpu") else obs_new["obs"].flatten()
        
        prev_state = state
        state = new_state
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return {
        "reward": total_reward,
        "avg_trust": np.mean(all_trust) if all_trust else 1.0,
        "avg_error": np.mean(all_errors) if all_errors else 0.0,
        "trust_min": np.min(all_trust) if all_trust else 0.0,
        "exploration_fraction": sum(1 for t in all_trust if t < TRUST_THRESHOLD) / len(all_trust) if all_trust else 0.0,
    }


def run_experiment(backbone_name="rssm", trust_name="ema"):
    print(f"\n{'='*60}")
    print(f"  Active Observation Experiment: {backbone_name} + {trust_name}")
    print(f"{'='*60}")
    
    env = make_env()
    obs_dim = 35  # ManiSkill PushCube state dim
    act_dim = 8
    
    wm = BACKBONES[backbone_name](obs_dim, act_dim).to(DEVICE)
    trust_scorer = make_trust(trust_name, obs_dim, act_dim) if trust_name != "none" else None
    
    results_random = []
    results_trust_guided = []
    
    for ep in range(N_EPISODES):
        # Collect data
        obs, _ = env.reset()
        obs_seq, act_seq = [], []
        
        for _ in range(STEPS_PER_EPISODE):
            state = obs["obs"].cpu().numpy().flatten() if hasattr(obs["obs"], "cpu") else obs["obs"].flatten()
            action = np.random.randn(7) * 0.3
            action = np.clip(action, -1, 1)
            obs_new, reward, terminated, truncated, info = env.step(action)
            obs_seq.append(state)
            act_seq.append(action)
            obs = obs_new
            if terminated or truncated:
                break
        
        obs_seq = np.array(obs_seq)
        act_seq = np.array(act_seq)
        
        if len(obs_seq) > SEQ_LEN:
            # Pad for batch processing
            obs_padded = np.zeros((1, len(obs_seq), obs_dim))
            act_padded = np.zeros((1, len(obs_seq), act_dim))
            obs_padded[0, :len(obs_seq)] = obs_seq
            act_padded[0, :len(act_seq)] = act_seq
            train_world_model(wm, obs_padded, act_padded, epochs=10)
        
        # Evaluate both modes
        r_random = evaluate(wm, trust_scorer, env, mode="random")
        r_guided = evaluate(wm, trust_scorer, env, mode="trust_guided")
        
        results_random.append(r_random)
        results_guided.append(r_guided)
        
        if (ep + 1) % 5 == 0:
            avg_r_rand = np.mean([r["reward"] for r in results_random[-5:]])
            avg_r_guid = np.mean([r["reward"] for r in results_guided[-5:]])
            print(f"  Episode {ep+1}/{N_EPISODES}: random={avg_r_rand:.3f}, trust_guided={avg_r_guid:.3f}")
    
    # Summary
    print(f"\n--- SUMMARY ---")
    avg_random = np.mean([r["reward"] for r in results_random])
    avg_guided = np.mean([r["reward"] for r in results_guided])
    avg_trust_exploration = np.mean([r["exploration_fraction"] for r in results_guided])
    
    print(f"  Random: avg_reward={avg_random:.3f}")
    print(f"  Trust-guided: avg_reward={avg_guided:.3f}, avg_exploration={avg_trust_exploration:.3f}")
    print(f"  Improvement: {((avg_guided - avg_random) / abs(avg_random) * 100):.1f}%")
    
    return {
        "random": results_random,
        "trust_guided": results_guided,
        "summary": {
            "random_avg_reward": avg_random,
            "guided_avg_reward": avg_guided,
            "exploration_fraction": avg_trust_exploration,
            "improvement_pct": (avg_guided - avg_random) / abs(avg_random) * 100 if avg_random != 0 else 0,
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="rssm")
    parser.add_argument("--trust", default="ema")
    parser.add_argument("--output", default="active_obs_results.json")
    args = parser.parse_args()
    
    results = run_experiment(args.backbone, args.trust)
    
    outpath = os.path.join(os.path.dirname(__file__), args.output)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
