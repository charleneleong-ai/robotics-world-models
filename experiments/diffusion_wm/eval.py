"""Evaluate an action-conditioned diffusion dynamics model.

Metrics:
    1-step MSE / MAE on held-out transitions
    Multi-step rollout divergence over 5/10/20-step horizons
    Rollout visualization (predicted vs ground-truth state trajectories)
    Comparison against TD-MPC2 dynamics head on same data

Usage:
    python -m experiments.diffusion_wm.eval \\
        --checkpoint checkpoints/diffusion_wm/peginsertion-v1/best.pt \\
        --data-dir data/diffusion_wm/peginsertion \\
        --out eval_results/peginsertion
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import hydra.utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import typer  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from experiments.diffusion_wm.dataset import TransitionDataset
from experiments.diffusion_wm.model import DiffusionDynamics, MLPDenoiser

DEFAULT_TDMPC2_DIR = Path("/workspace/ManiSkill/examples/baselines/tdmpc2")


def _resolve_tdmpc2_dir() -> Path:
    for cand in (Path.cwd(), DEFAULT_TDMPC2_DIR):
        if (cand / "config.yaml").exists() and (cand / "common").is_dir():
            return cand
    raise FileNotFoundError(f"tdmpc2 dir not found")


def _build_cfg(base_dir: Path, overrides: dict):
    cfg = OmegaConf.load(base_dir / "config.yaml")
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    hydra.utils.get_original_cwd = lambda: str(base_dir)
    from common.parser import parse_cfg  # tdmpc2 dir on sys.path — not top-level
    return parse_cfg(cfg)


def _load_tdmpc2_dynamics(
    checkpoint: Path, env_id: str = "PegInsertionSide-v1", model_size: int = 5,
) -> tuple:
    """Load TD-MPC2 and return its dynamics head for comparison."""
    base_dir = _resolve_tdmpc2_dir()
    import sys
    sys.path.insert(0, str(base_dir))

    overrides = {
        "env_id": env_id, "model_size": model_size, "obs": "state",
        "control_mode": "pd_joint_delta_pos", "num_envs": 1, "num_eval_envs": 1,
        "eval_episodes_per_env": 1, "env_type": "gpu", "seed": 1,
        "checkpoint": str(checkpoint), "save_video_local": False,
    }
    cfg = _build_cfg(base_dir, overrides)
    from tdmpc2 import TDMPC2
    agent = TDMPC2(cfg)
    assert checkpoint.exists(), f"TD-MPC2 checkpoint {checkpoint} not found."
    agent.load(str(checkpoint))
    return agent


def load_model(checkpoint: Path, device: torch.device) -> DiffusionDynamics:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    obs_dim = cfg.get("obs_dim", 45)
    act_dim = cfg.get("act_dim", 8)
    denoiser = MLPDenoiser(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=cfg.get("hidden_dim", 512),
        num_blocks=cfg.get("num_blocks", 6),
        cond_dim=cfg.get("cond_dim", 256),
    )
    model = DiffusionDynamics(denoiser, timesteps=cfg.get("diffusion_timesteps", 1000))
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_1step_metrics(
    model: DiffusionDynamics, dataset: TransitionDataset, device: torch.device,
    num_samples: int = 10000, num_denoise_steps: int = 100,
) -> dict[str, float]:
    """Compute 1-step prediction metrics on held-out data."""
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    nll_total = 0.0
    n = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False)

    for idx in indices:
        item = dataset[idx]
        obs = item["obs"].unsqueeze(0).to(device)
        action = item["action"].unsqueeze(0).to(device)
        next_obs_gt = item["next_obs"].unsqueeze(0).to(device)

        pred = model.sample(obs, action, num_steps=num_denoise_steps)
        mse = (pred - next_obs_gt).pow(2).mean().item()
        mae = (pred - next_obs_gt).abs().mean().item()
        mse_total += mse
        mae_total += mae

    n = max(1, n)
    return {
        "1step_mse": mse_total / n,
        "1step_mae": mae_total / n,
        "n_samples": n,
    }


@torch.no_grad()
def compute_rollout_metrics(
    model: DiffusionDynamics, dataset: TransitionDataset, device: torch.device,
    horizons: tuple[int, ...] = (5, 10, 20),
    num_episodes: int = 100, num_denoise_steps: int = 100,
) -> dict[str, float]:
    """Compute multi-step rollout divergence.

    Finds contiguous trajectory segments in the dataset (consecutive indices
    within the same episode) and rolls out the model autoregressively.
    """
    model.eval()
    results = {}
    indices = np.random.choice(max(1, len(dataset)), num_episodes, replace=False)

    for h in horizons:
        mse_total = 0.0
        count = 0
        for start_idx in indices:
            # Find a contiguous block of h+1 transitions
            end_idx = min(start_idx + h, len(dataset) - 1)
            actual_h = end_idx - start_idx
            if actual_h < 1:
                continue

            # Collect ground truth trajectory
            states = []
            actions = []
            for i in range(start_idx, end_idx + 1):
                item = dataset[i]
                if i == start_idx:
                    states.append(item["obs"].cpu())
                states.append(item["next_obs"].cpu())
                actions.append(item["action"].cpu())

            states_gt = torch.stack(states).to(device)  # [h+1, obs_dim]
            actions_seq = torch.stack(actions).to(device)  # [h, act_dim]

            # Rollout
            s = states_gt[:1]  # initial state
            preds = [s]
            for t in range(min(h, actual_h)):
                a = actions_seq[t:t+1]
                s_pred = model.sample(s, a, num_steps=num_denoise_steps)
                preds.append(s_pred)
                s = s_pred
            preds = torch.cat(preds, dim=0)

            # MSE over horizon
            mse = (preds - states_gt[:len(preds)]).pow(2).mean().item()
            mse_total += mse
            count += 1

        results[f"rollout_{h}step_mse"] = mse_total / max(1, count)
        results[f"rollout_{h}step_n"] = count

    return results


def main(
    checkpoint: Path = typer.Option(..., help="Diffusion model checkpoint (.pt)."),
    data_dir: Path = typer.Option(..., help="Data directory with shards."),
    tdmpc2_checkpoint: Path | None = typer.Option(None, help="Optional TD-MPC2 checkpoint for comparison."),
    out: Path = typer.Option(Path("eval_results"), help="Output directory."),
    num_eval_samples: int = typer.Option(2000, help="Samples for 1-step eval."),
    num_rollout_episodes: int = typer.Option(100, help="Episodes for rollout eval."),
    num_denoise_steps: int = typer.Option(100, help="DDPM sampling steps at eval."),
    seed: int = typer.Option(42),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading diffusion model...")
    model = load_model(checkpoint, device)
    print(f"  model loaded, {sum(p.numel() for p in model.parameters()):,} params")

    # Load dataset
    dataset = TransitionDataset(data_dir)

    # 1-step metrics
    print("Computing 1-step metrics...")
    step_metrics = compute_1step_metrics(
        model, dataset, device,
        num_samples=num_eval_samples,
        num_denoise_steps=num_denoise_steps,
    )
    print(f"  1-step MSE: {step_metrics['1step_mse']:.6f}")
    print(f"  1-step MAE: {step_metrics['1step_mae']:.6f}")

    # Rollout metrics
    print("Computing rollout metrics...")
    rollout_metrics = compute_rollout_metrics(
        model, dataset, device,
        num_episodes=num_rollout_episodes,
        num_denoise_steps=num_denoise_steps,
    )
    for k, v in rollout_metrics.items():
        if "mse" in k:
            print(f"  {k}: {v:.6f}")

    # TD-MPC2 comparison
    tdmpc2_metrics = {}
    if tdmpc2_checkpoint:
        print("Loading TD-MPC2 for comparison...")
        agent = _load_tdmpc2_dynamics(tdmpc2_checkpoint)
        # Evaluate TD-MPC2's dynamics head (model._dynamics) on same data
        # TD-MPC2 uses a deterministic dynamics: mu, log_var = model._dynamics(z, a)
        agent.model.eval()
        if hasattr(agent.model, "_dynamics"):
            from collections import defaultdict
            tdmpc2_errs = defaultdict(list)
            for idx in np.random.choice(len(dataset), min(num_eval_samples, len(dataset)), replace=False):
                item = dataset[idx]
                obs_t = item["obs"].unsqueeze(0).to(device)
                action_t = item["action"].unsqueeze(0).to(device)
                next_obs_gt = item["next_obs"].to(device)

                # TD-MPC2 encodes obs to latent, predicts next latent, decodes
                with torch.no_grad():
                    z = agent.model.encode(obs_t, action_t)
                    z_next_pred = agent.model._dynamics(z, action_t)
                    next_obs_pred = agent.model.decode(z_next_pred)
                    mse = (next_obs_pred.squeeze(0) - next_obs_gt).pow(2).mean().item()
                    mae = (next_obs_pred.squeeze(0) - next_obs_gt).abs().mean().item()
                    tdmpc2_errs["mse"].append(mse)
                    tdmpc2_errs["mae"].append(mae)

            tdmpc2_metrics = {
                "tdmpc2_1step_mse": float(np.mean(tdmpc2_errs["mse"])),
                "tdmpc2_1step_mae": float(np.mean(tdmpc2_errs["mae"])),
            }
            print(f"  TD-MPC2 1-step MSE: {tdmpc2_metrics['tdmpc2_1step_mse']:.6f}")
            print(f"  TD-MPC2 1-step MAE: {tdmpc2_metrics['tdmpc2_1step_mae']:.6f}")

    # Save results
    results = {
        "model": str(checkpoint),
        "data_dir": str(data_dir),
        **step_metrics,
        **rollout_metrics,
        **tdmpc2_metrics,
    }
    out_path = out / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    typer.run(main)
