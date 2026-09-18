#!/usr/bin/env python3
"""Real trust-guided action selection on ManiSkill, 5 seeds.

Both conditions draw candidate actions from the identical uniform-random
action distribution -- the only difference is whether an ensemble world
model (trained on real rollout data, not randomly initialized) is used to
pick the lowest-predicted-outcome-disagreement candidate out of K samples,
versus executing a single uniformly-random sample. This isolates the causal
effect of trust-guided selection from the action-magnitude confound present
in the original (fabricated-on-failure) script.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maniskill_benchmark import ManiSkillBenchmark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK = "PushCube-v1"
OBS_DIM, ACT_DIM = 64, 10
N_ENSEMBLE = 3
K_CANDIDATES = 8
NUM_SEEDS = 5
EVAL_EPISODES = 15
MAX_STEPS = 60
TRAIN_EPOCHS = 30


class EnsembleMember(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


def train_ensemble(obs, act, next_obs, seed):
    torch.manual_seed(seed)
    models = [EnsembleMember(OBS_DIM, ACT_DIM).to(DEVICE) for _ in range(N_ENSEMBLE)]
    n = obs.shape[0]
    for m in models:
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        idx_all = torch.randperm(n)
        for _ in range(TRAIN_EPOCHS):
            perm = idx_all[torch.randperm(n)]
            for c in range(0, n, 32):
                idx = perm[c:c + 32]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                pred = m(obs[idx], act[idx])
                loss = ((pred - next_obs[idx]) ** 2).mean()
                loss.backward()
                opt.step()
    return models


def ensemble_disagreement(models, obs_batch, act_batch):
    with torch.no_grad():
        preds = torch.stack([m(obs_batch, act_batch) for m in models], dim=0)
    return preds.var(dim=0).mean(dim=-1)


def obs_to_vec(obs):
    if isinstance(obs, dict):
        return np.concatenate([np.asarray(v).flatten() for v in obs.values()])[:OBS_DIM]
    return np.asarray(obs).flatten()[:OBS_DIM]


def run_condition(env, models, seed, condition):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    rewards = []
    for ep in range(EVAL_EPISODES):
        obs, info = env.reset(seed=int(rng.integers(1_000_000)))
        total_reward = 0.0
        for _step in range(MAX_STEPS):
            obs_vec = obs_to_vec(obs)
            obs_vec = np.pad(obs_vec, (0, max(0, OBS_DIM - len(obs_vec))))[:OBS_DIM]
            if condition == "random":
                action = env.action_space.sample()
            else:  # trust_guided
                candidates = np.stack([env.action_space.sample() for _ in range(K_CANDIDATES)])
                cand_padded = np.zeros((K_CANDIDATES, ACT_DIM), dtype=np.float32)
                for i, c in enumerate(candidates):
                    flat = np.asarray(c).flatten()
                    cand_padded[i, :min(ACT_DIM, len(flat))] = flat[:ACT_DIM]
                obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0).repeat(K_CANDIDATES, 1)
                act_t = torch.tensor(cand_padded, dtype=torch.float32, device=DEVICE)
                disagreement = ensemble_disagreement(models, obs_t, act_t)
                best_idx = int(torch.argmin(disagreement).item())
                action = candidates[best_idx]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return float(np.mean(rewards)), rewards


def main():
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    bench = ManiSkillBenchmark(num_tasks=1, episodes_per_task=15, max_steps=MAX_STEPS,
                                obs_dim=OBS_DIM, action_dim=ACT_DIM)
    random_means, trust_means = [], []
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_data = bench.collect_task_data(TASK, num_episodes=15)
        obs = torch.tensor(np.asarray(train_data["observations"]), dtype=torch.float32, device=DEVICE)
        act = torch.tensor(np.asarray(train_data["actions"]), dtype=torch.float32, device=DEVICE)
        next_obs = torch.tensor(np.asarray(train_data["next_observations"]), dtype=torch.float32, device=DEVICE)
        models = train_ensemble(obs, act, next_obs, seed)

        env = gym.make(TASK, render_mode=None)
        r_mean, _ = run_condition(env, models, seed, "random")
        t_mean, _ = run_condition(env, models, seed, "trust_guided")
        env.close()
        random_means.append(r_mean)
        trust_means.append(t_mean)
        print(f"seed={seed} random={r_mean:.4f} trust_guided={t_mean:.4f}", flush=True)

    random_means = np.array(random_means)
    trust_means = np.array(trust_means)
    t_stat, p_val = stats.ttest_rel(random_means, trust_means)
    result = {
        "random": {"mean": float(random_means.mean()), "std": float(random_means.std()), "seeds": random_means.tolist()},
        "trust_guided": {"mean": float(trust_means.mean()), "std": float(trust_means.std()), "seeds": trust_means.tolist()},
        "paired_t": float(t_stat),
        "p_value": float(p_val),
        "relative_improvement_pct": float(100 * (trust_means.mean() - random_means.mean()) / abs(random_means.mean())),
    }
    print(json.dumps(result, indent=2))
    with open("action_selection_real_results.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
