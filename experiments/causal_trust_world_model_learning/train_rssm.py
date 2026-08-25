"""Train RSSM on ManiSkill environments.

Collects data from ManiSkill, trains RSSM world model,
saves model for trust benchmarking.
"""

from __future__ import annotations

import json
from pathlib import Path

import mani_skill.envs
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import typer

from rssm_world_model import WorldModel

app = typer.Typer(help="Train RSSM world model on ManiSkill environments.")


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

    typer.echo(f"Collected {len(obs_arr)} steps from {env_name}")
    typer.echo(f"  obs: {obs_arr.shape}, actions: {act_arr.shape}")
    typer.echo(f"  reward mean: {rew_arr.mean():.4f}, done rate: {done_arr.mean():.2%}")

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


@app.command()
def train(
    env: str = typer.Option("PickCube-v1", help="ManiSkill environment name"),
    episodes: int = typer.Option(200, help="Number of episodes to collect"),
    epochs: int = typer.Option(50, help="Training epochs"),
    seq_len: int = typer.Option(32, help="Sequence length for RSSM"),
    batch_size: int = typer.Option(32, help="Batch size"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
    device: str = typer.Option("cuda", help="Device (cuda or cpu)"),
    save_dir: Path = typer.Option(Path("trained_models"), help="Directory to save model"),
) -> None:
    """Train RSSM world model on ManiSkill."""
    save_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)

    data = collect_data(env, n_episodes=episodes)
    obs, acts, rews, dones = make_sequences(data, seq_len=seq_len)
    obs, acts, rews, dones = obs.to(torch_device), acts.to(torch_device), rews.to(torch_device), dones.to(torch_device)

    dataset = TensorDataset(obs, acts, rews, dones)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    obs_dim = obs.shape[-1]
    action_dim = acts.shape[-1]

    typer.echo(f"Training RSSM: obs_dim={obs_dim}, action_dim={action_dim}")
    model = WorldModel(
        obs_dim=obs_dim, action_dim=action_dim,
        hidden_dim=256, stochastic_dim=32,
        stochastic_classes=32, deterministic_dim=512,
    ).to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        if (epoch + 1) % 10 == 0:
            typer.echo(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} obs={avg_obs:.4f}")

    model_path = save_dir / f"rssm_{env.lower().replace('/', '_')}.pt"
    torch.save(model.state_dict(), model_path)
    typer.echo(f"Saved model to {model_path}")

    meta = {
        "env_name": env, "obs_dim": obs_dim, "action_dim": action_dim,
        "n_episodes": episodes, "seq_len": seq_len, "epochs": epochs,
        "model_path": str(model_path),
    }
    with open(save_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    app()
