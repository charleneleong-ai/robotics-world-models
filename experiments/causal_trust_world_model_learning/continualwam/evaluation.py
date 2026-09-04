"""Shared evaluation functions."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def eval_bc(
    policy: torch.nn.Module,
    demos: list[dict[str, np.ndarray]],
    device: str = "cuda",
) -> float:
    """Evaluate behavioral cloning policy."""
    policy.eval()
    all_obs = np.concatenate([d["obs"] for d in demos])
    all_acts = np.concatenate([d["acts"] for d in demos])
    obs_t = torch.tensor(all_obs, dtype=torch.float32).to(device)
    act_t = torch.tensor(all_acts, dtype=torch.float32).to(device)
    with torch.no_grad():
        error = F.mse_loss(policy(obs_t), act_t).item()
    return error
