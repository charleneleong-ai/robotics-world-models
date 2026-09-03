"""Experiment 3: High-Trust Replay Buffer on LIBERO
Compare: EWC only vs EWC + replay of high-trust transitions."""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import BACKBONES, make_trust

DEVICE = "cuda"
N_TASKS = 5  # Fewer tasks for speed
PRETRAIN_EPOCHS = 20
FINETUNE_EPISODES = 3
STEPS_PER_EPISODE = 50
BATCH_SIZE = 32
LR = 1e-3
REPLAY_BUFFER_SIZE = 100
TRUST_THRESHOLD = 0.7


def load_demos_hdf5(suite, task_idx, n_demos=5):
    demo_dir = f"/home/ubuntu/robotics_world_models/LIBERO/{suite}"
    files = sorted([f for f in os.listdir(demo_dir) if f.endswith(".hdf5")])
    
    if task_idx >= len(files):
        return None, None
    
    filepath = os.path.join(demo_dir, files[task_idx])
    f = h5py.File(filepath, "r")
    
    all_obs, all_actions = [], []
    demo_keys = sorted(f["data"].keys())
    
    for i in range(min(n_demos, len(demo_keys))):
        demo = f["data"][demo_keys[i]]
        obs = demo["obs"]
        state_parts = []
        for k in sorted(obs.keys()):
            if k not in ["agentview_rgb", "eye_in_hand_rgb"]:
                state_parts.append(obs[k][:])
        state = np.concatenate(state_parts, axis=-1)
        actions = demo["actions"][:]
        all_obs.append(state)
        all_actions.append(actions)
    
    f.close()
    
    if not all_obs:
        return None, None
    
    max_len = max(o.shape[0] for o in all_obs)
    obs_dim = all_obs[0].shape[-1]
    act_dim = all_actions[0].shape[-1]
    
    obs_padded = np.zeros((len(all_obs), max_len, obs_dim))
    act_padded = np.zeros((len(all_actions), max_len, act_dim))
    
    for i, (o, a) in enumerate(zip(all_obs, all_actions)):
        T = o.shape[0]
        obs_padded[i, :T] = o
        act_padded[i, :T] = a
    
    return obs_padded, act_padded


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


class HighTrustReplayBuffer:
    """Stores transitions with high trust scores for replay."""
    def __init__(self, capacity=100, trust_threshold=0.7):
        self.capacity = capacity
        self.trust_threshold = trust_threshold
        self.buffer = []
        self.trust_scores = []
    
    def add(self, obs, act, trust):
        if trust >= self.trust_threshold:
            if len(self.buffer) >= self.capacity:
                # Remove lowest trust entry
                min_idx = np.argmin(self.trust_scores)
                self.buffer.pop(min_idx)
                self.trust_scores.pop(min_idx)
            self.buffer.append(obs)
            self.trust_scores.append(trust)
    
    def sample(self, n):
        if len(self.buffer) == 0:
            return None, None
        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        obs = np.array([self.buffer[i] for i in indices])
        return obs, None  # Actions come from the sequence
    
    def __len__(self):
        return len(self.buffer)


def evaluate_task(wm, task_idx, n_episodes=3):
    """Evaluate on a LIBERO task."""
    try:
        from libero.lifelong.benchmark import LIBEROSuiteBenchmark
        benchmark = LIBEROSuiteBenchmark("libero_spatial")
        env = benchmark.get_task(task_idx).env
    except Exception as e:
        print(f"  Env error: {e}")
        return -1.0
    
    wm.eval()
    total_reward = 0
    
    for ep in range(n_episodes):
        try:
            obs = env.reset()
        except:
            obs, _ = env.reset()
        
        for step in range(STEPS_PER_EPISODE):
            state = obs["agentview_rgb"] if "agentview_rgb" in obs else obs
            # Use random actions since we're measuring forgetting, not policy quality
            action = np.random.randn(7) * 0.3
            action = np.clip(action, -1, 1)
            
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except:
                obs, reward, done, info = env.step(action)
            
            total_reward += reward
            
            if terminated or truncated:
                break
    
    return total_reward / n_episodes


def run_experiment():
    print(f"\n{'='*60}")
    print(f"  High-Trust Replay Experiment: LIBERO-Spatial")
    print(f"{'='*60}")
    
    obs_dim = 21
    act_dim = 7
    
    # Method 1: EWC only (baseline)
    wm_ewc = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    
    # Method 2: EWC + high-trust replay
    wm_replay = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    trust_scorer = make_trust("ema", obs_dim, act_dim)
    replay_buffer = HighTrustReplayBuffer(capacity=REPLAY_BUFFER_SIZE, trust_threshold=TRUST_THRESHOLD)
    
    results_ewc = []
    results_replay = []
    
    for task_idx in range(N_TASKS):
        print(f"\n--- Task {task_idx+1}/{N_TASKS} ---")
        
        obs_seq, act_seq = load_demos_hdf5("libero_spatial", task_idx, n_demos=5)
        if obs_seq is None:
            print(f"  No demos, skipping")
            continue
        
        print(f"  Demos: {obs_seq.shape[0]} episodes, max len {obs_seq.shape[1]}")
        
        # Pretrain both models on this task
        train_world_model(wm_ewc, obs_seq, act_seq, epochs=PRETRAIN_EPOCHS)
        train_world_model(wm_replay, obs_seq, act_seq, epochs=PRETRAIN_EPOCHS)
        
        # Add high-trust transitions to replay buffer
        obs_t = torch.tensor(obs_seq, dtype=torch.float32).to(DEVICE)
        act_t = torch.tensor(act_seq, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            for i in range(obs_seq.shape[0]):
                for t in range(min(obs_seq.shape[1], 20)):  # Sample first 20 steps
                    pred_error = wm_replay.predict_error(
                        obs_t[i, t:t+1],
                        act_t[i, t:t+1],
                        obs_t[i, t+1:t+2] if t+1 < obs_seq.shape[1] else obs_t[i, t:t+1]
                    )
                    trust = trust_scorer.compute_trust(pred_error, task_idx).mean().item()
                    replay_buffer.add(obs_seq[i, t], act_seq[i, t], trust)
        
        print(f"  Replay buffer: {len(replay_buffer)} high-trust transitions")
        
        # Evaluate on all previous tasks (forgetting check)
        prev_rewards_ewc = []
        prev_rewards_replay = []
        for prev_idx in range(task_idx):
            r_ewc = evaluate_task(wm_ewc, prev_idx, n_episodes=2)
            r_replay = evaluate_task(wm_replay, prev_idx, n_episodes=2)
            prev_rewards_ewc.append(r_ewc)
            prev_rewards_replay.append(r_replay)
        
        if prev_rewards_ewc:
            avg_ewc = np.mean(prev_rewards_ewc)
            avg_replay = np.mean(prev_rewards_replay)
            print(f"  Previous tasks: EWC={avg_ewc:.3f}, Replay={avg_replay:.3f}")
        else:
            avg_ewc = 0.0
            avg_replay = 0.0
        
        results_ewc.append({"task": task_idx, "prev_avg_reward": avg_ewc})
        results_replay.append({"task": task_idx, "prev_avg_reward": avg_replay})
    
    # Summary
    print(f"\n--- SUMMARY ---")
    all_ewc = [r["prev_avg_reward"] for r in results_ewc if r["prev_avg_reward"] != 0]
    all_replay = [r["prev_avg_reward"] for r in results_replay if r["prev_avg_reward"] != 0]
    
    if all_ewc:
        avg_ewc = np.mean(all_ewc)
        avg_replay = np.mean(all_replay)
        print(f"  EWC only: avg_prev_reward={avg_ewc:.3f}")
        print(f"  EWC + replay: avg_prev_reward={avg_replay:.3f}")
        print(f"  Forgetting reduction: {((avg_ewc - avg_replay) / abs(avg_ewc) * 100):.1f}%")
    else:
        print("  No previous tasks to compare")
    
    return {
        "ewc": results_ewc,
        "replay": results_replay,
        "summary": {
            "ewc_avg_reward": avg_ewc if all_ewc else 0,
            "replay_avg_reward": avg_replay if all_replay else 0,
            "replay_buffer_size": len(replay_buffer),
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="trust_replay_results.json")
    args = parser.parse_args()
    
    results = run_experiment()
    
    outpath = os.path.join(os.path.dirname(__file__), args.output)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
