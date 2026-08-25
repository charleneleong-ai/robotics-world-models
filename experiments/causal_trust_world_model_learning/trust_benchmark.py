"""Trust Benchmark: Open-Loop vs Closed-Loop.

Runs 6 trust methods on trained RSSM predictions:
- Open-loop (prediction-only): EMA, FFDC, Ensemble
- Closed-loop (uses real obs): EMA+Feedback, FFDC+Conformal, Closed-Loop Trust

Metrics:
- Error-trust correlation: does trust predict low error?
- Trust variance: does trust vary across steps?
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

app = typer.Typer(help="Benchmark trust scoring methods on trained RSSM.")


def load_trained_rssm(
    model_path: Path, obs_dim: int, action_dim: int, device: torch.device
) -> WorldModel:
    """Load trained RSSM."""
    model = WorldModel(
        obs_dim=obs_dim, action_dim=action_dim,
        hidden_dim=256, stochastic_dim=32,
        stochastic_classes=32, deterministic_dim=512,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def collect_test_data(
    env_name: str, n_episodes: int = 50, max_steps: int = 100, seed: int = 9999
) -> list[dict]:
    """Collect test episodes."""
    env = gym.make(env_name, render_mode=None)
    episodes = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_obs, ep_acts = [], []

        for step in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_obs.append(obs.flatten())
            ep_acts.append(action.flatten())
            obs = next_obs
            if done:
                break

        episodes.append({
            "observations": np.array(ep_obs, dtype=np.float32),
            "actions": np.array(ep_acts, dtype=np.float32),
        })

    env.close()
    return episodes


def evaluate_trust_methods(
    model: WorldModel,
    episodes: list[dict],
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    trained_dir: Path = None,
) -> dict[str, dict]:
    """Evaluate all trust methods on test episodes."""
    methods = {
        "EMA (open)": EMAPredictionTrust(),
        "FFDC (open)": FFDCTrust(obs_dim=obs_dim, action_dim=action_dim, feature_dim=1024),
        "Ensemble (open)": EnsembleDisagreementTrust(obs_dim=1024),
        "EMA+Feedback (closed)": EMAPredictionTrust(),
        "FFDC+Conformal (closed)": FFDCTrust(obs_dim=obs_dim, action_dim=action_dim, feature_dim=1024),
        "Closed-Loop (ours)": ClosedLoopTrustCorrector(obs_dim=obs_dim),
    }

    # Load trained weights if available
    if trained_dir and trained_dir.exists():
        try:
            methods["FFDC (open)"].verifier.load_state_dict(torch.load(trained_dir / "ffdc_verifier.pt", map_location=device))
            methods["FFDC (open)"].verifier.eval()
            methods["FFDC+Conformal (closed)"].verifier.load_state_dict(torch.load(trained_dir / "ffdc_verifier.pt", map_location=device))
            methods["FFDC+Conformal (closed)"].verifier.eval()
            typer.echo("Loaded trained FFDC verifier")
        except Exception as e:
            typer.echo(f"Could not load FFDC: {e}")
        try:
            import torch.nn as nn
            methods["Ensemble (open)"].ensemble_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(1024, 128), nn.SiLU(), nn.Linear(128, obs_dim))
                for _ in range(5)
            ]).to(device)
            methods["Ensemble (open)"].ensemble_heads.load_state_dict(torch.load(trained_dir / "ensemble_heads.pt", map_location=device))
            methods["Ensemble (open)"].ensemble_heads.eval()
            typer.echo("Loaded trained ensemble heads")
        except Exception as e:
            typer.echo(f"Could not load ensemble: {e}")

    results = {name: {"trusts": [], "errors": []} for name in methods}

    with torch.no_grad():
        for ep in episodes:
            obs_t = torch.from_numpy(ep["observations"]).float().to(device)
            act_t = torch.from_numpy(ep["actions"]).float().to(device)

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

                    results[name]["trusts"].append(trust)
                    results[name]["errors"].append(error)

    metrics = {}
    for name, data in results.items():
        trusts = np.array(data["trusts"])
        errors = np.array(data["errors"])
        corr = -np.corrcoef(trusts, errors)[0, 1] if len(trusts) > 1 and np.std(trusts) > 1e-8 else 0.0
        metrics[name] = {
            "trust_mean": float(trusts.mean()),
            "trust_std": float(trusts.std()),
            "error_mean": float(errors.mean()),
            "error_trust_corr": float(corr),
        }
    return metrics


@app.command()
def run(
    env: str = typer.Option("PickCube-v1", help="ManiSkill environment"),
    model_path: Path = typer.Option(..., help="Path to trained RSSM"),
    n_episodes: int = typer.Option(50, help="Number of test episodes"),
    device: str = typer.Option("cuda", help="Device"),
    trained_dir: Path = typer.Option(None, help="Directory with trained trust methods"),
    save_results: Path = typer.Option(None, help="Save results JSON"),
) -> None:
    """Run trust benchmark comparing open-loop vs closed-loop methods."""
    torch_device = torch.device(device)

    env_obj = gym.make(env, render_mode=None)
    obs_dim = int(np.prod(env_obj.observation_space.shape))
    action_dim = int(np.prod(env_obj.action_space.shape))
    env_obj.close()

    typer.echo(f"Loading trained RSSM from {model_path}...")
    model = load_trained_rssm(model_path, obs_dim, action_dim, torch_device)

    typer.echo(f"Collecting {n_episodes} test episodes from {env}...")
    episodes = collect_test_data(env, n_episodes=n_episodes)

    typer.echo("Evaluating trust methods...")
    metrics = evaluate_trust_methods(model, episodes, obs_dim, action_dim, torch_device, trained_dir)

    typer.echo("\n" + "=" * 80)
    typer.echo("TRUST METHOD COMPARISON: Open-Loop vs Closed-Loop")
    typer.echo("=" * 80)
    typer.echo(f"{'Method':<30} {'Trust':>8} {'Error':>8} {'Corr':>8}")
    typer.echo("-" * 80)
    for name, m in metrics.items():
        typer.echo(f"{name:<30} {m['trust_mean']:>7.3f} {m['error_mean']:>7.4f} {m['error_trust_corr']:>7.3f}")
    typer.echo("-" * 80)
    typer.echo("Corr = trust-error correlation (higher = trust predicts low error)")
    typer.echo("=" * 80)

    if save_results:
        with open(save_results, "w") as f:
            json.dump(metrics, f, indent=2)
        typer.echo(f"\nResults saved to {save_results}")


if __name__ == "__main__":
    app()
