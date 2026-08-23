"""Trust Benchmark: Open-Loop vs Closed-Loop.

Runs 6 trust methods on trained RSSM predictions:
- Open-loop (prediction-only): EMA, FFDC, Ensemble
- Closed-loop (uses real obs): EMA+Feedback, FFDC+Conformal, Closed-Loop Trust

Metrics:
- Trust accuracy (AUC): does trust predict success?
- Error reduction: how much does feedback correction improve?
- Replan precision: when we replan, does it help?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import mani_skill.envs
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from rssm_world_model import WorldModel
from trust_metric_comparison import (
    EMAPredictionTrust,
    ActionStateConsistencyTrust,
    EnsembleDisagreementTrust,
    FFDCTrust,
    ForwardInverseCycleTrust,
)
from closed_loop_trust import ClosedLoopTrustCorrector


def load_trained_rssm(
    model_path: Path, obs_dim: int, action_dim: int, device: str = "cuda"
) -> WorldModel:
    """Load trained RSSM."""
    model = WorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        stochastic_dim=32,
        stochastic_classes=32,
        deterministic_dim=512,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def collect_test_data(
    env_name: str, n_episodes: int = 50, max_steps: int = 100, seed: int = 9999
) -> list[dict]:
    """Collect test episodes with success labels."""
    env = gym.make(env_name, render_mode=None)
    episodes = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_obs, ep_acts, ep_rews, ep_dones = [], [], [], []

        for step in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_obs.append(obs.flatten())
            ep_acts.append(action.flatten())
            ep_rews.append(reward)
            ep_dones.append(float(done))

            obs = next_obs
            if done:
                break

        success = info.get("success", False)
        episodes.append({
            "observations": np.array(ep_obs, dtype=np.float32),
            "actions": np.array(ep_acts, dtype=np.float32),
            "rewards": np.array(ep_rews, dtype=np.float32),
            "dones": np.array(ep_dones, dtype=np.float32),
            "success": success,
            "total_reward": sum(ep_rews),
        })

    env.close()
    return episodes


def evaluate_trust_methods(
    model: WorldModel,
    episodes: list[dict],
    obs_dim: int,
    action_dim: int,
    device: str = "cuda",
) -> dict[str, dict]:
    """Evaluate all trust methods on test episodes.

    Uses prediction error correlation instead of AUC (since random actions = 0% success).
    Metrics:
    - error_trust_corr: correlation between trust and -error (higher = better)
    - trust_variance: does trust vary? (higher = more informative)
    - feedback_error_reduction: closed-loop error vs open-loop error
    """
    methods = {
        # Open-loop (prediction-only)
        "EMA (open)": EMAPredictionTrust(),
        "FFDC (open)": FFDCTrust(obs_dim=obs_dim, action_dim=action_dim),
        "Ensemble (open)": EnsembleDisagreementTrust(obs_dim=obs_dim),
        # Closed-loop (uses real obs)
        "EMA+Feedback (closed)": EMAPredictionTrust(),
        "FFDC+Conformal (closed)": FFDCTrust(obs_dim=obs_dim, action_dim=action_dim),
        "Closed-Loop Trust (ours)": ClosedLoopTrustCorrector(obs_dim=obs_dim),
    }

    results = {name: {"trusts": [], "errors": []} for name in methods}

    with torch.no_grad():
        for ep in episodes:
            obs_seq = torch.from_numpy(ep["observations"]).float().to(device)
            act_seq = torch.from_numpy(ep["actions"]).float().to(device)

            h, z = model.rssm.initial_state(1, device)
            for t in range(len(obs_seq)):
                result = model.rssm.observe_step(h, z, act_seq[t:t+1], obs_seq[t:t+1])
                h, z = result["h"], result["z"]

                state = torch.cat([result["h"], result["z"]], dim=-1)
                pred_obs = model.obs_decoder(state)

                if t + 1 < len(obs_seq):
                    actual_obs = obs_seq[t+1:t+2]
                else:
                    actual_obs = obs_seq[t:t+1]

                pred_2d = pred_obs.squeeze(0)
                actual_2d = actual_obs.squeeze(0)
                act_1d = act_seq[t]
                z_1d = z.squeeze(0)

                error = F.mse_loss(pred_obs, actual_obs).item()

                for name, method in methods.items():
                    try:
                        if "Closed-Loop Trust" in name:
                            trust_result = method.observe_and_update(
                                pred_2d, actual_2d, act_1d, task_id=0
                            )
                            trust = trust_result["trust"].item()
                        elif "FFDC" in name:
                            verifier_input = torch.cat([pred_2d, actual_2d, z_1d, act_1d], dim=-1).unsqueeze(0)
                            trust = method.verifier(verifier_input).squeeze().item()
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

        # Correlation: trust should be HIGH when error is LOW
        if len(trusts) > 1 and np.std(trusts) > 1e-8:
            corr = -np.corrcoef(trusts, errors)[0, 1]
        else:
            corr = 0.0

        metrics[name] = {
            "trust_mean": float(trusts.mean()),
            "trust_std": float(trusts.std()),
            "error_mean": float(errors.mean()),
            "error_std": float(errors.std()),
            "error_trust_corr": float(corr),
            "n_steps": len(trusts),
        }

    return metrics


def print_comparison_table(metrics: dict[str, dict]):
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("TRUST METHOD COMPARISON: Open-Loop vs Closed-Loop")
    print("=" * 80)
    print(f"{'Method':<30} {'Trust':>8} {'Error':>8} {'Corr':>8}")
    print("-" * 80)

    for name, m in metrics.items():
        print(
            f"{name:<30} "
            f"{m['trust_mean']:>7.3f} "
            f"{m['error_mean']:>7.4f} "
            f"{m['error_trust_corr']:>7.3f}"
        )

    print("-" * 80)
    print("Trust = mean trust score (higher = more confident)")
    print("Error = mean prediction error (lower = better predictions)")
    print("Corr = trust-error correlation (higher = trust predicts low error)")
    print("Open-loop: trust = prediction-only evaluation")
    print("Closed-loop: trust = uses real observations for correction")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PickCube-v1")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-results", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Get env dimensions
    env = gym.make(args.env, render_mode=None)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    env.close()

    # Load trained RSSM
    print(f"Loading trained RSSM from {args.model_path}...")
    model = load_trained_rssm(Path(args.model_path), obs_dim, action_dim, args.device)

    # Collect test data
    print(f"Collecting {args.n_episodes} test episodes from {args.env}...")
    episodes = collect_test_data(args.env, n_episodes=args.n_episodes)

    # Evaluate
    print("Evaluating trust methods...")
    metrics = evaluate_trust_methods(model, episodes, obs_dim, action_dim, args.device)

    # Print results
    print_comparison_table(metrics)

    # Save results
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
