"""Plotly figures for diffusion world-model diagnostics.

Two charts:
    denoising_grid      intermediate x_0 estimates vs ground truth per sample
    rollout_trajectories predicted vs ground-truth state trajectories

Each returns a plotly Figure ready for wandb.Plotly / write_html.
"""
from __future__ import annotations

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from experiments.diffusion_wm.dataset import TransitionDataset
from experiments.diffusion_wm.model import DiffusionDynamics

_GT_COLOR = "#444444"
_PRED_COLOR = "#1f77b4"


def denoising_grid(
    model: DiffusionDynamics,
    state: torch.Tensor,
    action: torch.Tensor,
    next_obs: torch.Tensor,
    milestones: tuple[int, ...] = (750, 500, 250, 0),
    num_steps: int = 1000,
    max_rows: int = 4,
) -> go.Figure:
    """Rows = samples, cols = milestone timesteps (noisiest -> clean) with GT overlaid."""
    n = min(state.size(0), max_rows)
    state, action, next_obs = state[:n], action[:n], next_obs[:n]
    estimates = model.denoise_with_progress(state, action, num_steps=num_steps, milestones=milestones)

    fig = make_subplots(
        rows=n, cols=len(milestones),
        subplot_titles=[f"x0 estimate @ t={t}" for t in milestones],
    )
    for r in range(n):
        for c, est in enumerate(estimates):
            fig.add_trace(
                go.Scatter(
                    y=[float(v) for v in est[r].cpu().flatten()],
                    mode="lines", line={"color": _PRED_COLOR},
                    showlegend=(r == 0 and c == 0), name="x0 estimate",
                    legendgroup="pred",
                ),
                row=r + 1, col=c + 1,
            )
            fig.add_trace(
                go.Scatter(
                    y=[float(v) for v in next_obs[r].cpu().flatten()],
                    mode="lines", line={"color": _GT_COLOR, "dash": "dash"},
                    showlegend=(r == 0 and c == 0), name="GT next obs",
                    legendgroup="GT",
                ),
                row=r + 1, col=c + 1,
            )
        fig.update_yaxes(title_text=f"sample {r}", row=r + 1, col=1)
    fig.update_layout(
        height=170 * n, width=900,
        title=f"Denoising progress — x0 estimates vs GT across timesteps ({num_steps} reverse steps)",
    )
    return fig


def rollout_trajectories(
    model: DiffusionDynamics,
    dataset: TransitionDataset,
    device: torch.device,
    num_episodes: int = 3,
    horizon: int = 20,
    num_denoise_steps: int = 100,
    dims: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    seed: int = 42,
) -> go.Figure:
    """Rows = episodes, cols = obs dims; dashed = GT, solid = predicted."""
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    start_indices = torch.randint(0, max(1, len(dataset) - horizon), (num_episodes,), generator=rng)

    fig = make_subplots(
        rows=num_episodes, cols=len(dims),
        subplot_titles=[f"obs dim {d}" for d in dims],
    )
    for ep, start_idx in enumerate(start_indices.tolist()):
        item = dataset[start_idx]
        states = [item["obs"]]
        actions = []
        for i in range(start_idx, start_idx + horizon):
            item = dataset[i]
            states.append(item["next_obs"])
            actions.append(item["action"])
        states_gt = torch.stack(states).to(device)
        actions_seq = torch.stack(actions).to(device)

        s = states_gt[:1]
        preds = [s]
        for h in range(horizon):
            s = model.sample(s, actions_seq[h:h + 1], num_steps=num_denoise_steps)
            preds.append(s)
        preds = torch.cat(preds).cpu()
        gt = states_gt.cpu()

        for col, d in enumerate(dims):
            fig.add_trace(
                go.Scatter(
                    y=[float(v) for v in gt[:, d]], mode="lines",
                    line={"color": _GT_COLOR, "dash": "dash"},
                    showlegend=(ep == 0 and col == 0), name="GT",
                    legendgroup="GT",
                ),
                row=ep + 1, col=col + 1,
            )
            fig.add_trace(
                go.Scatter(
                    y=[float(v) for v in preds[:, d]], mode="lines",
                    line={"color": _PRED_COLOR},
                    showlegend=(ep == 0 and col == 0), name="predicted",
                    legendgroup="pred",
                ),
                row=ep + 1, col=col + 1,
            )
    fig.update_layout(
        height=200 * num_episodes, width=1000,
        title=f"Autoregressive rollout — predicted vs GT ({num_denoise_steps} denoise steps)",
    )
    return fig
