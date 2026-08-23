"""Train RSSM on ManiSkill environments.

Collects data from ManiSkill, trains RSSM world model,
saves model for trust benchmarking.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mani_skill.envs
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from rssm_world_model import WorldModel


def collect_data(
    env_name: str,
    n_episodes: int = 100,
    max_steps: int = 100,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Collect experience from ManiSkill environment."""
    env = gym.make(env_name, render_mode=None)
    all_obs, all_actions, all_rewards, all_dones = [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        for step in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            all_obs.append(obs.flatten())
            all_actions.append(action.flatten())
            all_rewards.append(reward)
            all_dones.append(float(done))

            obs = next_obs
            if done:
                break

    env.close()

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_actions, dtype=np.float32)
    rew_arr = np.array(all_rewards, dtype=np.float32)
    done_arr = np.array(all_dones, dtype=np.float32)

    print(f"Collected {len(obs_arr)} steps from {env_name}")
    print(f"  obs: {obs_arr.shape}, actions: {act_arr.shape}")
    print(f"  reward mean: {rew_arr.mean():.4f}, done rate: {done_arr.mean():.2%}")

    return {
        "observations": obs_arr,
        "actions": act_arr,
        "rewards": rew_arr,
        "dones": done_arr,
    }


def make_sequences(
    data: dict[str, np.ndarray],
    seq_len: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert flat data to sequences for RSSM training."""
    obs = torch.from_numpy(data["observations"]).float()
    acts = torch.from_numpy(data["actions"]).float()
    rews = torch.from_numpy(data["rewards"]).float()
    dones = torch.from_numpy(data["dones"]).float()

    n = len(obs) // seq_len
    obs = obs[: n * seq_len].view(n, seq_len, -1)
    acts = acts[: n * seq_len].view(n, seq_len, -1)
    rews = rews[: n * seq_len].view(n, seq_len)
    dones = dones[: n * seq_len].view(n, seq_len)

    return obs, acts, rews, dones


def train_rssm(
    env_name: str,
    save_dir: Path,
    n_episodes: int = 200,
    seq_len: int = 32,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda",
) -> Path:
    """Train RSSM on ManiSkill and save."""
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    print(f"Collecting data from {env_name}...")
    data = collect_data(env_name, n_episodes=n_episodes)

    obs, acts, rews, dones = make_sequences(data, seq_len=seq_len)
    obs, acts, rews, dones = obs.to(device), acts.to(device), rews.to(device), dones.to(device)

    dataset = TensorDataset(obs, acts, rews, dones)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    obs_dim = obs.shape[-1]
    action_dim = acts.shape[-1]

    print(f"Training RSSM: obs_dim={obs_dim}, action_dim={action_dim}")
    model = WorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        stochastic_dim=32,
        stochastic_classes=32,
        deterministic_dim=512,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch_obs, batch_acts, batch_rews, batch_dones in loader:
            optimizer.zero_grad()
            result = model.training_step(batch_obs, batch_acts, batch_rews, batch_dones)
            loss = result["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(result)

        avg_loss = np.mean([l["total_loss"].item() if isinstance(l["total_loss"], torch.Tensor) else l["total_loss"] for l in epoch_losses])
        avg_obs = np.mean([l["obs_loss"] for l in epoch_losses])
        avg_reward = np.mean([l["reward_loss"] for l in epoch_losses])

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} obs={avg_obs:.4f} reward={avg_reward:.4f}")

    # Save model
    model_path = save_dir / f"rssm_{env_name.lower().replace('/', '_')}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Save metadata
    meta = {
        "env_name": env_name,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_episodes": n_episodes,
        "seq_len": seq_len,
        "epochs": epochs,
        "model_path": str(model_path),
    }
    meta_path = save_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PickCube-v1")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    save_dir = Path(__file__).parent / "trained_models"
    train_rssm(
        env_name=args.env,
        save_dir=save_dir,
        n_episodes=args.episodes,
        epochs=args.epochs,
        device=args.device,
    )
