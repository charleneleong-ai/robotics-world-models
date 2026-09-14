import json
import numpy as np
from pathlib import Path
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import mani_skill.envs

base = Path("/home/ubuntu/robotics_world_models/experiments/causal_trust_world_model_learning")

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)

class RSSMWorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, hidden)
        self.dynamics = nn.Linear(hidden + act_dim, hidden)
        self.decoder = nn.Linear(hidden, obs_dim)
    def forward(self, obs, act, h=None):
        if h is None:
            h = torch.tanh(self.encoder(obs))
        else:
            h = torch.tanh(self.dynamics(torch.cat([h, act], dim=-1)))
        pred = self.decoder(h)
        return pred, h
    def compute_trust(self, obs, act, next_obs, alpha=0.99):
        with torch.no_grad():
            pred, _ = self.forward(obs, act)
            error = F.mse_loss(pred, next_obs, reduction="none").mean(dim=-1)
            trust = torch.exp(-alpha * error)
        return trust, error

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.trust_scores = deque(maxlen=capacity)
    def add(self, obs, act, next_obs, reward, done, trust=1.0):
        self.buffer.append((obs, act, next_obs, reward, done))
        self.trust_scores.append(trust)
    def sample(self, batch_size, method="random"):
        n = len(self.buffer)
        if n < batch_size:
            indices = np.random.choice(n, batch_size, replace=True)
        elif method == "random":
            indices = np.random.choice(n, batch_size, replace=False)
        elif method == "trust_priority":
            trust_arr = np.array(self.trust_scores)
            probs = 1.0 / (trust_arr + 1e-6)
            probs = probs / probs.sum()
            indices = np.random.choice(n, batch_size, replace=False, p=probs)
        elif method == "error_priority":
            trust_arr = np.array(self.trust_scores)
            error_arr = 1.0 - trust_arr
            probs = error_arr / error_arr.sum()
            indices = np.random.choice(n, batch_size, replace=False, p=probs)
        else:
            indices = np.random.choice(n, batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        obs = np.array([b[0] for b in batch])
        act = np.array([b[1] for b in batch])
        next_obs = np.array([b[2] for b in batch])
        reward = np.array([b[3] for b in batch])
        done = np.array([b[4] for b in batch])
        return obs, act, next_obs, reward, done
    def __len__(self):
        return len(self.buffer)

def train_one_seed(seed, method, n_tasks=5, episodes_per_task=15, buffer_size=500, batch_size=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("PushCube-v1", render_mode=None)
    obs_dim, act_dim = 35, 8
    policy = SimpleMLP(obs_dim, 128, act_dim).to(device)
    wm = RSSMWorldModel(obs_dim, act_dim, 128).to(device)
    opt = torch.optim.Adam(list(policy.parameters()) + list(wm.parameters()), lr=1e-3)
    buf = ReplayBuffer(buffer_size)
    task_rewards = []
    for task_id in range(n_tasks):
        obs, _ = env.reset(seed=task_id * 1000 + seed)
        obs = np.asarray(obs, dtype=np.float32).flatten()[:obs_dim]
        ep_rewards = []
        for ep in range(episodes_per_task):
            total_r = 0
            for step in range(50):
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    action = policy(obs_t).squeeze(0).cpu().numpy()
                    action = np.clip(action, -1, 1)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_obs = np.asarray(next_obs, dtype=np.float32).flatten()[:obs_dim]
                obs_b = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                act_b = torch.from_numpy(action).float().unsqueeze(0).to(device)
                next_obs_b = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
                trust, error = wm.compute_trust(obs_b, act_b, next_obs_b)
                buf.add(obs, action, next_obs, reward, terminated or truncated, trust.item())
                total_r += reward
                if len(buf) >= batch_size:
                    b_obs, b_act, b_next_obs, _, _ = buf.sample(batch_size, method)
                    b_obs_t = torch.from_numpy(b_obs).float().to(device)
                    b_act_t = torch.from_numpy(b_act).float().to(device)
                    b_next_obs_t = torch.from_numpy(b_next_obs).float().to(device)
                    pred, _ = wm(b_obs_t, b_act_t)
                    wm_loss = F.mse_loss(pred, b_next_obs_t)
                    pred_act = policy(b_obs_t)
                    pol_loss = F.mse_loss(pred_act, b_act_t)
                    opt.zero_grad()
                    (wm_loss + 0.1 * pol_loss).backward()
                    opt.step()
                obs = next_obs
                if terminated or truncated:
                    obs, _ = env.reset(seed=task_id * 1000 + ep + seed)
                    obs = np.asarray(obs, dtype=np.float32).flatten()[:obs_dim]
                    break
            ep_rewards.append(total_r)
        task_rewards.append(float(np.mean(ep_rewards)))
    env.close()
    return task_rewards

def main():
    print("="*60)
    print("SELECTIVE REPLAY 5-SEED EXPERIMENT")
    print("="*60)
    n_seeds = 5
    methods = ["random", "error_priority", "trust_priority"]
    all_results = {}
    for method in methods:
        print("\n--- %s ---" % method)
        seed_rewards = []
        for seed in range(n_seeds):
            rewards = train_one_seed(seed, method)
            avg = np.mean(rewards)
            seed_rewards.append(rewards)
            print("  Seed %d: avg=%.2f" % (seed, avg))
        all_results[method] = seed_rewards
    # Compute stats
    summary = {}
    for method in methods:
        avgs = [np.mean(r) for r in all_results[method]]
        summary[method] = {"mean": float(np.mean(avgs)), "std": float(np.std(avgs)), "seeds": [float(np.mean(r)) for r in all_results[method]]}
    with open(base / "selective_replay_5seeds.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "="*60)
    print("RESULTS (mean +/- std over %d seeds)" % n_seeds)
    print("="*60)
    for method in methods:
        s = summary[method]
        print("%-20s %.2f +/- %.2f" % (method, s["mean"], s["std"]))
    rand = summary["random"]
    for method in ["error_priority", "trust_priority"]:
        s = summary[method]
        imp = (s["mean"] - rand["mean"]) / abs(rand["mean"]) * 100
        print("%s vs random: %+.1f%%" % (method, imp))

if __name__ == "__main__":
    main()
