"""Experiment 2: Open-Loop vs Closed-Loop Trust on ManiSkill StackCube
Compare prediction-only trust vs prediction+correction trust."""

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
BATCH_SIZE = 32
LR = 1e-3
SEQ_LEN = 10


class OpenLoopTrust:
    """Trust based only on prediction error."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.errors = []
    
    def compute_trust(self, pred_error, obs=None, next_obs=None):
        e = float(pred_error.mean())
        self.errors.append(e)
        ema = np.mean(self.errors[-10:]) if self.errors else e
        return torch.exp(-self.alpha * pred_error / (ema + 1e-8)).clamp(0, 1)


class ClosedLoopTrust:
    """Trust based on prediction error + correction from actual observation."""
    def __init__(self, alpha=1.0, correction_weight=0.5):
        self.alpha = alpha
        self.correction_weight = correction_weight
        self.errors = []
    
    def compute_trust(self, pred_error, obs=None, next_obs=None):
        e = float(pred_error.mean())
        self.errors.append(e)
        ema = np.mean(self.errors[-10:]) if self.errors else e
        
        # Base trust from prediction error
        base_trust = torch.exp(-self.alpha * pred_error / (ema + 1e-8)).clamp(0, 1)
        
        # Correction signal: how much did the model learn from the actual observation?
        if obs is not None and next_obs is not None:
            with torch.no_grad():
                # Measure how well the model's prediction was corrected
                correction = F.mse_loss(obs, next_obs, reduction="none").mean(dim=-1)
                correction_trust = torch.exp(-self.correction_weight * correction).clamp(0, 1)
            # Combine: trust is high when both prediction is good AND correction is small
            combined_trust = base_trust * correction_trust
        else:
            combined_trust = base_trust
        
        return combined_trust


def make_env():
    import mani_skill.envs
    env = mani_skill.envs.create("StackCube-v1", obs_mode="state", render_mode="rgb_array")
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


def evaluate_trust(wm, trust_scorer, env):
    """Evaluate one episode, return trust statistics."""
    obs, _ = env.reset()
    state = obs["obs"].cpu().numpy().flatten() if hasattr(obs["obs"], "cpu") else obs["obs"].flatten()
    
    total_reward = 0
    all_trust = []
    all_errors = []
    prev_state = None
    
    for step in range(STEPS_PER_EPISODE):
        obs_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_error = wm.predict_error(obs_t, torch.zeros(1, 8, device=DEVICE), obs_t)
            err_val = pred_error.mean().item()
            all_errors.append(err_val)
            
            # Compute trust (open-loop or closed-loop)
            if prev_state is not None:
                prev_obs_t = torch.tensor(prev_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                trust = trust_scorer.compute_trust(pred_error, prev_obs_t, obs_t).mean().item()
            else:
                trust = trust_scorer.compute_trust(pred_error).mean().item()
            all_trust.append(trust)
        
        action = np.random.randn(8) * 0.3
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
        "trust_std": np.std(all_trust) if all_trust else 0.0,
        "trust_range": (np.min(all_trust), np.max(all_trust)) if all_trust else (0, 1),
    }


def run_experiment():
    print(f"\n{'='*60}")
    print(f"  Open-Loop vs Closed-Loop Trust: StackCube")
    print(f"{'='*60}")
    
    env = make_env()
    obs_dim = 48  # ManiSkill StackCube state dim
    act_dim = 8
    
    wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    
    open_loop = OpenLoopTrust(alpha=1.0)
    closed_loop = ClosedLoopTrust(alpha=1.0, correction_weight=0.5)
    
    results_open = []
    results_closed = []
    
    for ep in range(N_EPISODES):
        # Collect data
        obs, _ = env.reset()
        obs_seq, act_seq = [], []
        
        for _ in range(STEPS_PER_EPISODE):
            state = obs["obs"].cpu().numpy().flatten() if hasattr(obs["obs"], "cpu") else obs["obs"].flatten()
            action = np.random.randn(8) * 0.3
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
            obs_padded = np.zeros((1, len(obs_seq), obs_dim))
            act_padded = np.zeros((1, len(obs_seq), act_dim))
            obs_padded[0, :len(obs_seq)] = obs_seq
            act_padded[0, :len(act_seq)] = act_seq
            train_world_model(wm, obs_padded, act_padded, epochs=10)
        
        # Evaluate both trust methods
        r_open = evaluate_trust(wm, open_loop, env)
        r_closed = evaluate_trust(wm, closed_loop, env)
        
        results_open.append(r_open)
        results_closed.append(r_closed)
        
        if (ep + 1) % 5 == 0:
            avg_r_open = np.mean([r["reward"] for r in results_open[-5:]])
            avg_r_closed = np.mean([r["reward"] for r in results_closed[-5:]])
            avg_t_open = np.mean([r["avg_trust"] for r in results_open[-5:]])
            avg_t_closed = np.mean([r["avg_trust"] for r in results_closed[-5:]])
            print(f"  Episode {ep+1}/{N_EPISODES}:")
            print(f"    Open-loop: reward={avg_r_open:.3f}, trust={avg_t_open:.3f}")
            print(f"    Closed-loop: reward={avg_r_closed:.3f}, trust={avg_t_closed:.3f}")
    
    # Summary
    print(f"\n--- SUMMARY ---")
    avg_open = np.mean([r["reward"] for r in results_open])
    avg_closed = np.mean([r["reward"] for r in results_closed])
    trust_open = np.mean([r["avg_trust"] for r in results_open])
    trust_closed = np.mean([r["avg_trust"] for r in results_closed])
    trust_std_open = np.mean([r["trust_std"] for r in results_open])
    trust_std_closed = np.mean([r["trust_std"] for r in results_closed])
    
    print(f"  Open-loop: reward={avg_open:.3f}, trust={trust_open:.3f} +/- {trust_std_open:.3f}")
    print(f"  Closed-loop: reward={avg_closed:.3f}, trust={trust_closed:.3f} +/- {trust_std_closed:.3f}")
    print(f"  Trust calibration: closed-loop has {'better' if trust_std_closed < trust_std_open else 'worse'} calibration")
    
    return {
        "open_loop": results_open,
        "closed_loop": results_closed,
        "summary": {
            "open_avg_reward": avg_open,
            "closed_avg_reward": avg_closed,
            "open_avg_trust": trust_open,
            "closed_avg_trust": trust_closed,
            "open_trust_std": trust_std_open,
            "closed_trust_std": trust_std_closed,
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="closed_loop_results.json")
    args = parser.parse_args()
    
    results = run_experiment()
    
    outpath = os.path.join(os.path.dirname(__file__), args.output)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
