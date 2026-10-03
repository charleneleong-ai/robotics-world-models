"""Build and load World Action Models from training configs and checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .scaled_wam import ScaledDiffusionWAM
from .world_action_model import DiffusionWAM

WAM = DiffusionWAM | ScaledDiffusionWAM
MAX_MLP_HIDDEN_DIM = 512


def wam_class(hidden_dim: int) -> type[WAM]:
    return ScaledDiffusionWAM if hidden_dim > MAX_MLP_HIDDEN_DIM else DiffusionWAM


def load_wam(checkpoint_path: Path, device: torch.device | str) -> WAM:
    """Load a `train_wam` checkpoint; DiffusionWAM nests its weights under "denoiser", the scaled model is flat."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: dict[str, Any] = ckpt.get("config", {})
    weights = ckpt["model"]
    hidden_dim = cfg.get("hidden_dim", 512)
    model = wam_class(hidden_dim)(
        obs_dim=weights.get("obs_dim", cfg.get("obs_dim", 42)),
        act_dim=weights.get("act_dim", cfg.get("act_dim", 7)),
        hidden_dim=hidden_dim,
        num_blocks=cfg.get("num_blocks", 6),
        cond_dim=cfg.get("cond_dim", 256),
        timesteps=weights.get("timesteps", cfg.get("diffusion_timesteps", 1000)),
        action_horizon=weights.get("action_horizon", cfg.get("action_horizon", 1)),
        condition_on_action=cfg.get("condition_on_action", False),
    ).to(device)
    if isinstance(weights.get("denoiser"), dict):
        model.denoiser.load_state_dict(weights["denoiser"])
    else:
        model.load_state_dict(weights)
    return model.eval()
