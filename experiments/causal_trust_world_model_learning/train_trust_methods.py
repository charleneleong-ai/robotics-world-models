"""Train trust scoring methods.

Trains FFDC verifier, Ensemble heads, and calibrates Closed-Loop trust.
All methods share the same trained RSSM backbone.
"""

from __future__ import annotations

import json
from pathlib import Path

import mani_skill.envs
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import typer

from rssm_world_model import WorldModel
from trust_metric_comparison import (
    FFDCTrust,
    EnsembleDisagreementTrust,
    ForwardInverseCycleTrust,
)
from closed_loop_trust import ClosedLoopTrustCorrector

app = typer.Typer(help="Train trust scoring methods (FFDC, Ensemble, Cycle, Closed-Loop).")


def collect_labeled_data(
    env_name: str,
    n_episodes: int = 100,
    max_steps: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Collect episodes with diverse noise strategies for training trust methods."""
    env = gym.make(env_name, render_mode=None)
    episodes = []

    strategies = [("random", 0.0), ("noisy_low", 0.1), ("noisy_med", 0.3), ("noisy_high", 0.5), ("very_high", 1.0)]

    for strategy, noise in strategies:
        for ep in range(n_episodes // len(strategies)):
            obs, info = env.reset(seed=seed + ep)
            obs = np.asarray(obs, dtype=np.float32)
            ep_data = []

            for step in range(max_steps):
                base_action = env.action_space.sample()
                noisy_action = base_action + np.random.randn(*base_action.shape) * noise
                noisy_action = np.clip(noisy_action, -1, 1).astype(np.float32)

                next_obs, reward, terminated, truncated, info = env.step(noisy_action)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                done = terminated or truncated

                ep_data.append({
                    "obs": obs.flatten(),
                    "action": noisy_action.flatten(),
                    "next_obs": next_obs.flatten(),
                })
                obs = next_obs
                if done:
                    break

            episodes.append({"data": ep_data, "strategy": strategy, "noise": noise})

    env.close()
    total_steps = sum(len(e["data"]) for e in episodes)
    typer.echo(f"Collected {len(episodes)} episodes ({total_steps} steps)")
    return episodes


def create_training_data(episodes: list[dict], rssm: WorldModel, device: torch.device) -> dict[str, torch.Tensor]:
    """Create training tensors from episodes using RSSM features."""
    all_pred, all_actual, all_feat, all_act, all_labels = [], [], [], [], []

    rssm.eval()
    with torch.no_grad():
        for ep in episodes:
            h, z = rssm.rssm.initial_state(1, device)
            for step in ep["data"]:
                obs_t = torch.from_numpy(step["obs"]).float().unsqueeze(0).to(device)
                act_t = torch.from_numpy(step["action"]).float().unsqueeze(0).to(device)
                next_obs_t = torch.from_numpy(step["next_obs"]).float().unsqueeze(0).to(device)

                result = rssm.rssm.observe_step(h, z, act_t, obs_t)
                h, z = result["h"], result["z"]
                state = torch.cat([result["h"], result["z"]], dim=-1)
                pred_obs = rssm.obs_decoder(state)

                label = 1.0 if ep["noise"] < 0.2 else 0.0 if ep["noise"] > 0.4 else 0.5

                all_pred.append(pred_obs.squeeze(0).cpu())
                all_actual.append(next_obs_t.squeeze(0).cpu())
                all_feat.append(z.squeeze(0).cpu())
                all_act.append(act_t.squeeze(0).cpu())
                all_labels.append(label)

    return {
        "pred_obs": torch.stack(all_pred), "actual_obs": torch.stack(all_actual),
        "features": torch.stack(all_feat), "actions": torch.stack(all_act),
        "labels": torch.tensor(all_labels),
    }


def train_ffdc(data: dict[str, torch.Tensor], obs_dim: int, action_dim: int, feature_dim: int, epochs: int, device: torch.device) -> FFDCTrust:
    """Train FFDC verifier to predict match/mismatch."""
    verifier = FFDCTrust(obs_dim=obs_dim, action_dim=action_dim, feature_dim=feature_dim)
    verifier.verifier = verifier.verifier.to(device)
    optimizer = torch.optim.Adam(verifier.verifier.parameters(), lr=1e-3)

    keys = ["pred_obs", "actual_obs", "features", "actions"]
    tensors = [data[k].to(device) for k in keys]
    labels = data["labels"].to(device)
    loader = DataLoader(TensorDataset(*tensors, labels), batch_size=128, shuffle=True)

    typer.echo("Training FFDC verifier...")
    for epoch in range(epochs):
        verifier.verifier.train()
        total_loss, correct, total = 0, 0, 0
        for batch in loader:
            optimizer.zero_grad()
            inp = torch.cat(batch[:-1], dim=-1)
            pred = verifier.verifier(inp).squeeze(-1)
            loss = F.binary_cross_entropy(pred, batch[-1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += ((pred > 0.5).float() == batch[-1]).sum().item()
            total += len(batch[-1])
        if (epoch + 1) % 10 == 0:
            typer.echo(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f} acc={correct/total:.3f}")
    return verifier


def train_ensemble(data: dict[str, torch.Tensor], feature_dim: int, obs_dim: int, n_heads: int, epochs: int, device: torch.device) -> EnsembleDisagreementTrust:
    """Train ensemble heads to predict next observation from features."""
    ensemble = EnsembleDisagreementTrust(n_ensemble=n_heads, obs_dim=feature_dim)
    ensemble.ensemble_heads = nn.ModuleList([
        nn.Sequential(nn.Linear(feature_dim, 128), nn.SiLU(), nn.Linear(128, obs_dim))
        for _ in range(n_heads)
    ]).to(device)
    optimizer = torch.optim.Adam(ensemble.ensemble_heads.parameters(), lr=1e-3)

    loader = DataLoader(TensorDataset(data["features"].to(device), data["actual_obs"].to(device)), batch_size=128, shuffle=True)

    typer.echo("Training ensemble heads...")
    for epoch in range(epochs):
        for head in ensemble.ensemble_heads:
            head.train()
        total_loss = 0
        for feat, actual in loader:
            optimizer.zero_grad()
            preds = torch.stack([head(feat) for head in ensemble.ensemble_heads], dim=0)
            loss = F.mse_loss(preds, actual.unsqueeze(0).expand(n_heads, -1, -1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            typer.echo(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
    return ensemble


def train_inverse(data: dict[str, torch.Tensor], obs_dim: int, action_dim: int, epochs: int, device: torch.device) -> ForwardInverseCycleTrust:
    """Train inverse model for cycle consistency."""
    cycle = ForwardInverseCycleTrust(obs_dim=obs_dim, action_dim=action_dim)
    cycle.inverse_model = cycle.inverse_model.to(device)
    optimizer = torch.optim.Adam(cycle.inverse_model.parameters(), lr=1e-3)

    loader = DataLoader(TensorDataset(
        data["pred_obs"].to(device), data["actual_obs"].to(device), data["actions"].to(device)
    ), batch_size=128, shuffle=True)

    typer.echo("Training inverse model...")
    for epoch in range(epochs):
        cycle.inverse_model.train()
        total_loss = 0
        for pred, actual, act in loader:
            optimizer.zero_grad()
            pred_act = cycle.inverse_model(torch.cat([pred, actual], dim=-1))
            loss = F.mse_loss(pred_act, act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            typer.echo(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
    return cycle


def calibrate_closed_loop(data: dict[str, torch.Tensor], obs_dim: int, device: torch.device) -> ClosedLoopTrustCorrector:
    """Calibrate Closed-Loop trust corrector."""
    corrector = ClosedLoopTrustCorrector(obs_dim=obs_dim)
    typer.echo("Calibrating Closed-Loop trust...")
    for i in range(len(data["pred_obs"])):
        corrector.observe_and_update(
            data["pred_obs"][i].to(device), data["actual_obs"][i].to(device),
            data["actions"][i].to(device), task_id=0,
        )
    stats = corrector.get_error_statistics(task_id=0)
    typer.echo(f"  Calibrated: mean_error={stats['mean']:.4f}")
    return corrector


@app.command()
def train(
    env: str = typer.Option("PushCube-v1", help="ManiSkill environment"),
    rssm_path: Path = typer.Option(..., help="Path to trained RSSM"),
    episodes: int = typer.Option(100, help="Episodes to collect for training"),
    epochs: int = typer.Option(30, help="Training epochs"),
    n_ensemble: int = typer.Option(5, help="Number of ensemble heads"),
    device: str = typer.Option("cuda", help="Device"),
    save_dir: Path = typer.Option(Path("trained_models"), help="Save directory"),
) -> None:
    """Train all trust scoring methods on top of trained RSSM."""
    torch_device = torch.device(device)

    env_obj = gym.make(env, render_mode=None)
    obs_dim = int(np.prod(env_obj.observation_space.shape))
    action_dim = int(np.prod(env_obj.action_space.shape))
    env_obj.close()

    typer.echo(f"Loading RSSM from {rssm_path}...")
    rssm = WorldModel(
        obs_dim=obs_dim, action_dim=action_dim,
        hidden_dim=256, stochastic_dim=32,
        stochastic_classes=32, deterministic_dim=512,
    ).to(torch_device)
    rssm.load_state_dict(torch.load(rssm_path, map_location=torch_device))
    rssm.eval()

    episodes = collect_labeled_data(env, n_episodes=episodes)
    training_data = create_training_data(episodes, rssm, torch_device)
    typer.echo(f"  {len(training_data['labels'])} training samples")

    feature_dim = training_data["features"].shape[1]
    ffdc = train_ffdc(training_data, obs_dim, action_dim, feature_dim, epochs, torch_device)
    ensemble = train_ensemble(training_data, feature_dim, obs_dim, n_ensemble, epochs, torch_device)
    cycle = train_inverse(training_data, obs_dim, action_dim, epochs, torch_device)
    closed_loop = calibrate_closed_loop(training_data, obs_dim, torch_device)

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ffdc.verifier.state_dict(), save_dir / "ffdc_verifier.pt")
    torch.save(ensemble.ensemble_heads.state_dict(), save_dir / "ensemble_heads.pt")
    torch.save(cycle.inverse_model.state_dict(), save_dir / "inverse_model.pt")
    with open(save_dir / "closed_loop_meta.json", "w") as f:
        json.dump({"calibrated_thresholds": closed_loop.calibrated_thresholds, "ema_error": closed_loop.ema_error}, f, indent=2)

    typer.echo(f"\nAll trust methods saved to {save_dir}")


if __name__ == "__main__":
    app()
