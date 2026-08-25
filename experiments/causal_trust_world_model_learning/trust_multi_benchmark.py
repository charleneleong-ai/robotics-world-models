"""Multi-environment trust benchmark with noise variation.

Tests trust methods across multiple ManiSkill environments
with different action noise levels to create varied prediction quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import mani_skill.envs
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import typer

from rssm_world_model import WorldModel
from trust_metric_comparison import (
    EMAPredictionTrust,
    FFDCTrust,
    EnsembleDisagreementTrust,
)
from closed_loop_trust import ClosedLoopTrustCorrector

app = typer.Typer(help="Multi-environment trust benchmark with noise variation.")


def collect_noisy_data(
    env_name: str,
    n_episodes: int = 30,
    max_steps: int = 50,
    noise_levels: list[float] = [0.0, 0.1, 0.3, 0.5, 1.0],
    seed: int = 42,
) -> list[dict]:
    """Collect data with varying action noise."""
    env = gym.make(env_name, render_mode=None)
    all_data = []

    for noise in noise_levels:
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            ep_obs, ep_acts = [], []

            for step in range(max_steps):
                base_action = env.action_space.sample()
                noisy_action = base_action + np.random.randn(*base_action.shape) * noise
                noisy_action = np.clip(noisy_action, -1, 1)

                next_obs, reward, terminated, truncated, info = env.step(noisy_action)
                done = terminated or truncated

                ep_obs.append(obs.flatten())
                ep_acts.append(noisy_action.flatten())
                obs = next_obs
                if done:
                    break

            all_data.append({
                "observations": np.array(ep_obs, dtype=np.float32),
                "actions": np.array(ep_acts, dtype=np.float32),
                "noise_level": noise,
            })

    env.close()
    return all_data


def train_rssm_on_data(data: list[dict], obs_dim: int, action_dim: int, device: torch.device) -> WorldModel:
    """Train RSSM on collected data."""
    all_obs = np.concatenate([d["observations"] for d in data])
    all_acts = np.concatenate([d["actions"] for d in data])

    obs_t = torch.from_numpy(all_obs).float().to(device)
    acts_t = torch.from_numpy(all_acts).float().to(device)

    model = WorldModel(
        obs_dim=obs_dim, action_dim=action_dim,
        hidden_dim=256, stochastic_dim=32,
        stochastic_classes=32, deterministic_dim=512,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    seq_len = 32
    n = len(obs_t) // seq_len
    obs_seq = obs_t[:n*seq_len].view(n, seq_len, -1)
    act_seq = acts_t[:n*seq_len].view(n, seq_len, -1)
    rews_seq = torch.zeros(n, seq_len, device=device)
    done_seq = torch.zeros(n, seq_len, device=device)

    typer.echo(f"Training RSSM on {n} sequences...")
    for epoch in range(30):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0
        for i in range(0, n, 32):
            idx = perm[i:i+32]
            optimizer.zero_grad()
            result = model.training_step(obs_seq[idx], act_seq[idx], rews_seq[idx], done_seq[idx])
            result["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += result["obs_loss"]
        if (epoch + 1) % 10 == 0:
            typer.echo(f"  Epoch {epoch+1}: obs_loss={epoch_loss/n:.4f}")

    model.eval()
    return model


def evaluate_trust(model: WorldModel, data: list[dict], obs_dim: int, device: torch.device) -> dict:
    """Evaluate trust methods grouped by noise level."""
    methods = {
        "EMA": EMAPredictionTrust(),
        "FFDC": FFDCTrust(obs_dim=obs_dim, action_dim=obs_dim),
        "Ensemble": EnsembleDisagreementTrust(obs_dim=obs_dim),
        "Closed-Loop": ClosedLoopTrustCorrector(obs_dim=obs_dim),
    }

    noise_results = {}
    with torch.no_grad():
        for d in data:
            noise = d["noise_level"]
            if noise not in noise_results:
                noise_results[noise] = {name: {"trusts": [], "errors": []} for name in methods}

            obs_t = torch.from_numpy(d["observations"]).float().to(device)
            act_t = torch.from_numpy(d["actions"]).float().to(device)

            h, z = model.rssm.initial_state(1, device)
            for t in range(len(obs_t)):
                result = model.rssm.observe_step(h, z, act_t[t:t+1], obs_t[t:t+1])
                h, z = result["h"], result["z"]
                state = torch.cat([result["h"], result["z"]], dim=-1)
                pred_obs = model.obs_decoder(state)

                actual = obs_t[t+1:t+2] if t+1 < len(obs_t) else obs_t[t:t+1]
                error = F.mse_loss(pred_obs, actual).item()

                pred_2d, actual_2d = pred_obs.squeeze(0), actual.squeeze(0)
                z_1d, act_1d = z.squeeze(0), act_t[t]

                for name, method in methods.items():
                    try:
                        if "Closed-Loop" in name:
                            trust = method.observe_and_update(pred_2d, actual_2d, act_1d, task_id=0)["trust"].item()
                        elif "FFDC" in name:
                            inp = torch.cat([pred_2d, actual_2d, z_1d, act_1d], dim=-1).unsqueeze(0)
                            trust = method.verifier(inp).squeeze().item()
                        elif "Ensemble" in name:
                            trust = method.compute_trust(z_1d.unsqueeze(0), task_id=0).mean().item()
                        else:
                            trust = method.compute_trust(pred_2d, actual_2d, task_id=0).mean().item()
                    except Exception:
                        trust = 0.5

                    noise_results[noise][name]["trusts"].append(trust)
                    noise_results[noise][name]["errors"].append(error)

    return noise_results


@app.command()
def run(
    env: str = typer.Option("PushCube-v1", help="ManiSkill environment"),
    episodes: int = typer.Option(30, help="Episodes per noise level"),
    device: str = typer.Option("cuda", help="Device"),
    save_results: Path = typer.Option(None, help="Save results JSON"),
) -> None:
    """Run multi-environment trust benchmark with noise variation."""
    torch_device = torch.device(device)

    env_obj = gym.make(env, render_mode=None)
    obs_dim = int(np.prod(env_obj.observation_space.shape))
    action_dim = int(np.prod(env_obj.action_space.shape))
    env_obj.close()

    typer.echo(f"Collecting noisy data from {env}...")
    data = collect_noisy_data(env, n_episodes=episodes)

    typer.echo("Training RSSM...")
    model = train_rssm_on_data(data, obs_dim, action_dim, torch_device)

    typer.echo("Evaluating trust methods...")
    results = evaluate_trust(model, data, obs_dim, torch_device)

    typer.echo("\n" + "=" * 90)
    typer.echo(f"TRUST COMPARISON: {env} (5 noise levels × {episodes} episodes)")
    typer.echo("=" * 90)
    typer.echo(f"{'Noise':>6} | {'EMA':>14} {'FFDC':>14} {'Ensemble':>14} {'Closed-Loop':>14}")
    typer.echo("-" * 90)

    for noise in sorted(results.keys()):
        row = f"{noise:>6.1f}"
        for name in ["EMA", "FFDC", "Ensemble", "Closed-Loop"]:
            trusts = np.array(results[noise][name]["trusts"])
            errors = np.array(results[noise][name]["errors"])
            corr = -np.corrcoef(trusts, errors)[0, 1] if len(trusts) > 1 and np.std(trusts) > 1e-8 else 0.0
            row += f" | {corr:>5.3f} ({trusts.mean():.3f})"
        typer.echo(row)

    typer.echo("-" * 90)
    typer.echo("Corr: trust-error correlation (higher = trust predicts low error)")
    typer.echo("=" * 90)

    if save_results:
        save_data = {str(k): {name: {"trust_mean": float(np.array(v["trusts"]).mean()),
                                      "error_mean": float(np.array(v["errors"]).mean())}
                             for name, v in methods.items()}
                    for k, methods in results.items()}
        with open(save_results, "w") as f:
            json.dump(save_data, f, indent=2)
        typer.echo(f"\nSaved to {save_results}")


if __name__ == "__main__":
    app()
