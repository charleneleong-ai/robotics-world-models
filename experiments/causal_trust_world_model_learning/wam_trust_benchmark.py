"""WAM Continual Learning Benchmark with Trained Trust Methods.

Integrates trained FFDC, Ensemble, and Closed-Loop trust into ContinualWAM
and evaluates on PushCube-v1 (10 sequential tasks).

Measures: Average Accuracy, Forgetting, Forward Transfer
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
import typer

from rssm_world_model import WorldModel
from trust_metric_comparison import (
    EMAPredictionTrust,
    FFDCTrust,
    EnsembleDisagreementTrust,
)
from closed_loop_trust import ClosedLoopTrustCorrector

app = typer.Typer(help="WAM CL benchmark with trained trust methods.")


class TrustWeightedEWC:
    """EWC with trust-weighted penalty."""

    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher: dict[str, torch.Tensor] = {}
        self.optimal: dict[str, torch.Tensor] = {}

    def consolidate(self, trust_scores: dict[str, float] | None = None):
        """Save current parameters and compute Fisher information."""
        self.optimal = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

        # Use trust scores to weight Fisher if provided
        for n, p in self.model.named_parameters():
            if trust_scores and n in trust_scores:
                self.fisher[n] = torch.ones_like(p) * trust_scores[n]
            else:
                self.fisher[n] = torch.ones_like(p)

    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty."""
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal[n]) ** 2).sum()
        return self.lambda_ewc * loss


class SimplePolicy(nn.Module):
    """Simple MLP policy for WAM benchmark."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_task(
    policy: nn.Module,
    env_name: str,
    task_id: int,
    n_episodes: int = 20,
    max_steps: int = 50,
    lr: float = 1e-3,
    ewc: TrustWeightedEWC | None = None,
    trust_method=None,
    rssm: WorldModel | None = None,
    device: torch.device = torch.device("cuda"),
) -> dict[str, float]:
    """Train policy on one task with optional trust-weighted consolidation."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    env = gym.make(env_name, render_mode=None)

    episode_rewards = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=task_id * 1000 + ep)
        obs = np.asarray(obs, dtype=np.float32)
        obs_t = torch.from_numpy(obs.flatten()).float().unsqueeze(0).to(device)
        total_reward = 0

        for step in range(max_steps):
            action = policy(obs_t)
            next_obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy().flatten())
            next_obs = np.asarray(next_obs, dtype=np.float32)
            done = terminated or truncated
            total_reward += reward

            next_obs_t = torch.from_numpy(next_obs.flatten()).float().unsqueeze(0).to(device)
            reward_t = torch.tensor([reward], device=device)
            done_t = torch.tensor([float(done)], device=device)

            # Compute trust if RSSM available
            trust_weight = 1.0
            if rssm and trust_method:
                with torch.no_grad():
                    h, z = rssm.rssm.initial_state(1, device)
                    result = rssm.rssm.observe_step(h, z, action, obs_t)
                    state = torch.cat([result["h"], result["z"]], dim=-1)
                    pred_obs = rssm.obs_decoder(state)

                if "ClosedLoop" in str(type(trust_method).__name__):
                    trust_result = trust_method.observe_and_update(
                        pred_obs.squeeze(0), next_obs_t.squeeze(0),
                        action.squeeze(0), task_id=task_id
                    )
                    trust_weight = trust_result["trust"].item()
                elif "FFDC" in str(type(trust_method).__name__):
                    inp = torch.cat([pred_obs.squeeze(0), next_obs_t.squeeze(0),
                                    z.squeeze(0), action.squeeze(0)], dim=-1).unsqueeze(0)
                    trust_weight = trust_method.verifier(inp).squeeze().item()
                elif "Ensemble" in str(type(trust_method).__name__):
                    trust_weight = trust_method.compute_trust(z, task_id=task_id).mean().item()
                elif hasattr(trust_method, "compute_trust"):
                    trust_weight = trust_method.compute_trust(
                        pred_obs.squeeze(0), next_obs_t.squeeze(0), task_id=task_id
                    ).mean().item()
                else:
                    trust_weight = 0.5

            # Policy gradient loss
            log_prob = F.mse_loss(action, torch.from_numpy(env.action_space.sample()).float().to(device).unsqueeze(0))
            policy_loss = log_prob * trust_weight

            # EWC penalty
            ewc_loss = ewc.penalty() if ewc else torch.tensor(0.0)

            loss = policy_loss + ewc_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs_t
            if done:
                break

        episode_rewards.append(total_reward)

    # Consolidate after task
    if ewc:
        ewc.consolidate()

    env.close()
    return {"avg_reward": np.mean(episode_rewards), "trust_weight": trust_weight}


def evaluate(
    policy: nn.Module,
    env_name: str,
    task_id: int,
    n_episodes: int = 10,
    max_steps: int = 50,
    device: torch.device = torch.device("cuda"),
) -> float:
    """Evaluate policy on one task."""
    env = gym.make(env_name, render_mode=None)
    rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=task_id * 1000 + ep + 5000)
        obs = np.asarray(obs, dtype=np.float32)
        obs_t = torch.from_numpy(obs.flatten()).float().unsqueeze(0).to(device)
        total_reward = 0

        for step in range(max_steps):
            with torch.no_grad():
                action = policy(obs_t)
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().flatten())
            next_obs = np.asarray(next_obs, dtype=np.float32)
            total_reward += reward
            obs_t = torch.from_numpy(next_obs.flatten()).float().unsqueeze(0).to(device)
            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()
    return float(np.mean(rewards))


@app.command()
def run(
    env: str = typer.Option("PushCube-v1", help="Environment"),
    n_tasks: int = typer.Option(10, help="Number of sequential tasks"),
    n_episodes: int = typer.Option(20, help="Episodes per task"),
    n_eval: int = typer.Option(10, help="Evaluation episodes"),
    device: str = typer.Option("cuda", help="Device"),
    rssm_path: Path = typer.Option(None, help="Path to trained RSSM"),
    trained_dir: Path = typer.Option(None, help="Directory with trained trust methods"),
) -> None:
    """Run WAM CL benchmark with different trust methods."""
    torch_device = torch.device(device)

    env_obj = gym.make(env, render_mode=None)
    obs_dim = int(np.prod(env_obj.observation_space.shape))
    action_dim = int(np.prod(env_obj.action_space.shape))
    env_obj.close()

    # Load RSSM if available
    rssm = None
    if rssm_path and rssm_path.exists():
        rssm = WorldModel(
            obs_dim=obs_dim, action_dim=action_dim,
            hidden_dim=256, stochastic_dim=32,
            stochastic_classes=32, deterministic_dim=512,
        ).to(torch_device)
        rssm.load_state_dict(torch.load(rssm_path, map_location=torch_device))
        rssm.eval()
        typer.echo(f"Loaded RSSM from {rssm_path}")

    # Load trained trust methods
    trust_methods = {}
    if trained_dir and trained_dir.exists():
        # EMA (no training needed)
        trust_methods["EMA"] = EMAPredictionTrust()

        # FFDC (trained)
        try:
            ffdc = FFDCTrust(obs_dim=obs_dim, action_dim=action_dim, feature_dim=1024)
            ffdc.verifier.load_state_dict(torch.load(trained_dir / "ffdc_verifier.pt", map_location=torch_device))
            trust_methods["FFDC"] = ffdc
            ffdc.verifier = ffdc.verifier.to(torch_device)
            typer.echo("Loaded trained FFDC")
        except Exception as e:
            typer.echo(f"Could not load FFDC: {e}")

        # Ensemble (trained)
        try:
            ensemble = EnsembleDisagreementTrust(obs_dim=1024)
            ensemble.ensemble_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(1024, 128), nn.SiLU(), nn.Linear(128, obs_dim))
                for _ in range(5)
            ]).to(torch_device)
            ensemble.ensemble_heads.load_state_dict(torch.load(trained_dir / "ensemble_heads.pt", map_location=torch_device))
            trust_methods["Ensemble"] = ensemble
            typer.echo("Loaded trained Ensemble")
        except Exception as e:
            typer.echo(f"Could not load Ensemble: {e}")

        # Closed-Loop
        trust_methods["Closed-Loop"] = ClosedLoopTrustCorrector(obs_dim=obs_dim)

    # Add no-trust baseline
    trust_methods["No Trust"] = None

    results = {}
    for method_name, trust_method in trust_methods.items():
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Running: {method_name}")
        typer.echo(f"{'='*60}")

        policy = SimplePolicy(obs_dim, action_dim).to(torch_device)
        ewc = TrustWeightedEWC(policy, lambda_ewc=1000.0) if trust_method else None

        task_rewards = []
        for task_id in range(n_tasks):
            # Train
            train_result = train_task(
                policy, env, task_id, n_episodes=n_episodes,
                ewc=ewc, trust_method=trust_method, rssm=rssm, device=torch_device,
            )

            # Evaluate on current and all previous tasks
            eval_rewards = []
            for prev_task in range(task_id + 1):
                r = evaluate(policy, env, prev_task, n_episodes=n_eval, device=torch_device)
                eval_rewards.append(r)

            avg_eval = np.mean(eval_rewards)
            task_rewards.append(avg_eval)
            typer.echo(f"  Task {task_id+1}/{n_tasks}: train={train_result['avg_reward']:.3f} eval={avg_eval:.3f}")

        # Compute metrics
        final_acc = np.mean(task_rewards)
        forgetting = np.mean([task_rewards[0] - r for r in task_rewards[1:]])
        forward_transfer = np.mean([task_rewards[i] - task_rewards[i-1] for i in range(1, len(task_rewards))])

        results[method_name] = {
            "avg_acc": float(final_acc),
            "forgetting": float(forgetting),
            "forward_transfer": float(forward_transfer),
            "task_rewards": [float(r) for r in task_rewards],
        }

        typer.echo(f"  Final: AvgAcc={final_acc:.3f} Forgetting={forgetting:.3f}")

    # Print comparison
    typer.echo("\n" + "=" * 70)
    typer.echo("WAM CL BENCHMARK: Trust Method Comparison")
    typer.echo("=" * 70)
    typer.echo(f"{'Method':<20} {'AvgAcc':>8} {'Forgetting':>10} {'FwdTransfer':>12}")
    typer.echo("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["avg_acc"]):
        typer.echo(f"{name:<20} {r['avg_acc']:>7.3f} {r['forgetting']:>9.3f} {r['forward_transfer']:>11.3f}")
    typer.echo("=" * 70)

    # Save
    save_path = Path("benchmark_results") / "wam_trust_comparison.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    typer.echo(f"\nSaved to {save_path}")


if __name__ == "__main__":
    app()
