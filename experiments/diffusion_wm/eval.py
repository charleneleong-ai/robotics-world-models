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
from typing import Any

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
    candidates = (
        Path.cwd(),
        DEFAULT_TDMPC2_DIR,
        Path(__file__).resolve().parent.parent.parent / "benchmarks" / "ManiSkill" / "examples" / "baselines" / "tdmpc2",
    )
    for cand in candidates:
        if (cand / "config.yaml").exists() and (cand / "common").is_dir():
            return cand
    raise FileNotFoundError("tdmpc2 dir not found")


def _build_cfg(base_dir: Path, overrides: dict):
    cfg = OmegaConf.load(base_dir / "config.yaml")
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    hydra.utils.get_original_cwd = lambda: str(base_dir)
    from common.parser import parse_cfg  # tdmpc2 dir on sys.path — not top-level
    return parse_cfg(cfg)


def _load_tdmpc2_dynamics(
    checkpoint: str | Path, env_id: str = "PegInsertionSide-v1", model_size: int = 5,
    obs_dim: int = 43, act_dim: int = 8,
) -> tuple:
    """Load TD-MPC2 and return its dynamics head for comparison."""
    checkpoint = Path(checkpoint)
    assert checkpoint.exists(), f"TD-MPC2 checkpoint {checkpoint} not found."
    base_dir = _resolve_tdmpc2_dir()
    import sys
    sys.path.insert(0, str(base_dir))

    overrides = {
        "env_id": env_id, "model_size": model_size, "obs": "state",
        "control_mode": "pd_joint_delta_pos", "num_envs": 1, "num_eval_envs": 1,
        "eval_episodes_per_env": 1, "env_type": "gpu", "seed": 1,
        "checkpoint": str(checkpoint), "save_video_local": False,
        "multitask": False, "obs_shape": {"state": [obs_dim]},
        "action_dim": act_dim, "episode_length": 200,
    }
    cfg = _build_cfg(base_dir, overrides)
    from tdmpc2 import TDMPC2
    agent = TDMPC2(cfg)
    agent.load(str(checkpoint))
    return agent


def load_model(checkpoint: Path, device: torch.device) -> DiffusionDynamics:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sd = ckpt["model"]
    out_w = sd["denoiser.output_proj.1.weight"]
    in_w = sd["denoiser.input_proj.weight"]
    obs_dim = cfg.get("obs_dim") or out_w.shape[0]
    act_dim = cfg.get("act_dim") or (in_w.shape[1] - 2 * obs_dim)
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


def compute_tdmpc2_metrics(
    agent: object, dataset: TransitionDataset, device: torch.device,
    num_samples: int = 2000, num_episodes: int = 100,
    horizons: tuple[int, ...] = (5, 10, 20), seed: int = 42,
) -> dict[str, float]:
    """Latent-space dynamics error + rollout divergence for TD-MPC2.

    This ManiSkill baseline's WorldModel has no observation decoder, so the
    comparison happens in latent space: predict next(z_t, a_t) and compare
    against encode(next_obs). Rollouts track drift from the true latent path.
    """
    model = agent.model
    model.eval()
    rng = np.random.default_rng(seed)

    indices = rng.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    mse_total = 0.0
    mae_total = 0.0
    latent_dim = 0
    for idx in indices:
        item = dataset[idx]
        obs = item["obs"].unsqueeze(0).to(device)
        action = item["action"].unsqueeze(0).to(device)
        next_obs = item["next_obs"].unsqueeze(0).to(device)
        z = model.encode(obs, None)
        z_next_pred = model.next(z, action, None)
        z_next_true = model.encode(next_obs, None)
        latent_dim = z.size(-1)
        mse_total += (z_next_pred - z_next_true).pow(2).mean().item()
        mae_total += (z_next_pred - z_next_true).abs().mean().item()
    n = len(indices)
    metrics: dict[str, float] = {
        "tdmpc2_1step_latent_mse": mse_total / n,
        "tdmpc2_1step_latent_mae": mae_total / n,
        "tdmpc2_latent_dim": latent_dim,
    }

    ep_indices = rng.choice(len(dataset), num_episodes, replace=False)
    for h in horizons:
        mse_total = 0.0
        count = 0
        for start_idx in ep_indices:
            end_idx = min(start_idx + h, len(dataset) - 1)
            actual_h = end_idx - start_idx
            if actual_h < 1:
                continue
            states = []
            actions = []
            for i in range(start_idx, end_idx + 1):
                item = dataset[i]
                if i == start_idx:
                    states.append(item["obs"])
                states.append(item["next_obs"])
                actions.append(item["action"])
            states_t = torch.stack(states).to(device)
            actions_t = torch.stack(actions).to(device)

            z = model.encode(states_t[:1], None)
            for t in range(actual_h):
                z = model.next(z, actions_t[t:t + 1], None)
            z_true = model.encode(states_t[1:actual_h + 1], None)
            mse_total += (z - z_true).pow(2).mean().item()
            count += 1
        metrics[f"tdmpc2_rollout_{h}step_latent_mse"] = mse_total / max(1, count)
    return metrics


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
        tdmpc2_metrics = compute_tdmpc2_metrics(
            agent, dataset, device,
            num_samples=num_eval_samples,
            num_episodes=num_rollout_episodes,
            seed=seed,
        )
        print(f"  TD-MPC2 1-step latent MSE: {tdmpc2_metrics['tdmpc2_1step_latent_mse']:.6f}")
        print(f"  TD-MPC2 1-step latent MAE: {tdmpc2_metrics['tdmpc2_1step_latent_mae']:.6f}")

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

    log_media(model, dataset, device, out, step_metrics, rollout_metrics, tdmpc2_metrics, checkpoint=checkpoint)


def _fig_to_video(fig, path: Path, fps: int = 4) -> Path | None:
    """Render a Plotly figure as an MP4 video (one frame per subplot row)."""
    try:
        import imageio.v3 as iio
    except ImportError:
        print("imageio not installed — skipping video render")
        return None
    try:
        img_bytes = fig.to_image(format="png", width=1000, height=200 * max(1, len(fig.data) // 2))
        import numpy as np
        from PIL import Image
        import io
        frame = np.array(Image.open(io.BytesIO(img_bytes)))
        frames = np.stack([frame] * max(fps, 2))  # repeat for short video
        iio.imwrite(str(path), fps=fps, loop=0)
        print(f"Saved rollout video: {path}")
        return path
    except Exception as e:
        print(f"Video render failed (kaleido may be missing): {e}")
        return None


def log_media(
    model: DiffusionDynamics,
    dataset: TransitionDataset,
    device: torch.device,
    out: Path,
    step_metrics: dict[str, float],
    rollout_metrics: dict[str, float],
    tdmpc2_metrics: dict[str, float],
    val_split: float = 0.05,
    checkpoint: Path | None = None,
):
    """Save per-split (train/val) plotly charts to HTML and log to wandb when available.

    Split is contiguous (last ``val_split`` of the dataset is val) so rollout windows
    are real episodes — a random split would scatter transitions and break contiguity.
    """
    try:
        import wandb
    except ImportError:
        wandb = None  # type: ignore[assignment]
    from torch.utils.data import Subset
    from experiments.diffusion_wm.viz import denoising_grid, rollout_trajectories

    n_val = max(1, int(len(dataset) * val_split))
    train_ds = Subset(dataset, range(len(dataset) - n_val))
    val_ds = Subset(dataset, range(len(dataset) - n_val, len(dataset)))
    splits = {"train": train_ds, "val": val_ds}

    out.mkdir(parents=True, exist_ok=True)
    logged: dict[str, Any] = {}
    for split_name, split_ds in splits.items():
        # Fixed per-split media batch (deterministic seed)
        rng = torch.Generator().manual_seed(7)
        idx = torch.randint(0, len(split_ds), (4,), generator=rng)
        items = [split_ds[i.item()] for i in idx]
        obs = torch.stack([it["obs"] for it in items]).to(device)
        action = torch.stack([it["action"] for it in items]).to(device)
        next_obs = torch.stack([it["next_obs"] for it in items]).to(device)

        fig_grid = denoising_grid(model, obs, action, next_obs, num_steps=model.timesteps)
        fig_rollout = rollout_trajectories(model, split_ds, device, num_episodes=3, horizon=20)

        grid_html = out / f"{split_name}_denoising_grid.html"
        rollout_html = out / f"{split_name}_rollout_trajectories.html"
        fig_grid.write_html(grid_html)
        fig_rollout.write_html(rollout_html)
        print(f"Saved media: {grid_html}, {rollout_html}")

    if wandb is not None:
        logged[f"media/{split_name}_denoising_grid"] = wandb.Plotly(fig_grid)
        logged[f"media/{split_name}_rollout_trajectories"] = wandb.Plotly(fig_rollout)
        video_path = _fig_to_video(fig_rollout, out / f"{split_name}_rollout.mp4")
        if video_path is not None:
            logged[f"media/{split_name}_rollout_video"] = wandb.Video(str(video_path), fps=4)

    if wandb is None:
        print("wandb not installed — media logged as HTML only")
        return

    # Resume the training run when the checkpoint dir carries its wandb id,
    # so train + eval live in one run instead of two.
    run_id_file = checkpoint.parent / "wandb_run_id.txt" if checkpoint else None
    resume_kwargs: dict[str, str] = {}
    if run_id_file and run_id_file.exists():
        resume_kwargs = {"id": run_id_file.read_text().strip(), "resume": "must"}
        print(f"Resuming training run {resume_kwargs['id']} for eval media/metrics")
    elif run_id_file:
        print(f"WARNING: no {run_id_file} found — logging to a new run instead")

    wandb.init(
        project="wm-manip", entity="chaleong",
        **({"name": f"eval-{out.name}"} if not resume_kwargs else {}),
        config={
            "1step_mse": step_metrics["1step_mse"],
            "1step_mae": step_metrics["1step_mae"],
            **{k: v for k, v in rollout_metrics.items() if "mse" in k},
            **{k: v for k, v in tdmpc2_metrics.items() if "mse" in k},
        },
        **resume_kwargs,
    )
    run_id = wandb.run.id if wandb.run else "?"
    wandb.log({
        **logged,
        **{f"metrics/{k}": v for k, v in step_metrics.items()},
        **{f"metrics/{k}": v for k, v in rollout_metrics.items()},
        **{f"metrics/{k}": v for k, v in tdmpc2_metrics.items()},
    })
    wandb.finish()
    print(f"Logged media + metrics to wandb run {run_id}")


if __name__ == "__main__":
    typer.run(main)
