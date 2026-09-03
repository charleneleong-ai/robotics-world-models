"""Shared W&B logging helpers for ContinualWAM experiments.

Import this module in all experiment scripts:
    from wandb_helpers import log_video, log_frame_grid, log_reward_chart, log_heatmap, log_bar_chart
"""
import io
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb


def log_video(run, key: str, frames: list, fps: int = 8):
    """Log a list of (H,W,3) uint8 numpy frames as an MP4 video to W&B."""
    if not frames:
        return
    arr = np.stack(frames)  # (T, H, W, 3)
    run.log({key: wandb.Video(arr, fps=fps, format="mp4")})


def log_frame_grid(run, key: str, frames: list, title: str, ncols: int = 5):
    """Log selected frames as a horizontal grid image."""
    if not frames:
        return
    n = min(len(frames), ncols)
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    selected = [frames[i] for i in indices]
    h, w = selected[0].shape[:2]
    gap = 2
    grid = np.zeros((h, w * n + (n - 1) * gap, 3), dtype=np.uint8)
    for i, f in enumerate(selected):
        x = i * (w + gap)
        grid[:, x:x + w] = f[:, :, :3]
    fig, ax = plt.subplots(figsize=(2 * n, 2))
    ax.imshow(grid)
    ax.set_xticks([i * (w + gap) + w // 2 for i in range(n)])
    ax.set_xticklabels([f"t={indices[i]}" for i in range(n)], fontsize=6)
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        fig.savefig(tf, format="png", dpi=150)
        tmpname = tf.name
    plt.close(fig)
    run.log({key: wandb.Image(tmpname)})
    os.unlink(tmpname)


def log_reward_chart(run, key: str, rewards: list, title: str, labels: list | None = None):
    """Log a bar chart of rewards."""
    fig, ax = plt.subplots(figsize=(max(4, len(rewards) * 0.8), 4))
    x = range(len(rewards))
    ax.bar(x, rewards, color="#4C72B0", edgecolor="white", linewidth=0.5)
    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        fig.savefig(tf, format="png", dpi=150)
        tmpname = tf.name
    plt.close(fig)
    run.log({key: wandb.Image(tmpname)})
    os.unlink(tmpname)


def log_heatmap(run, key: str, matrix: dict, title: str):
    """Log a backbone x trust heatmap."""
    backbones = list(matrix.keys())
    trusts = list(next(iter(matrix.values())).keys())
    data = [[matrix[b][t] for t in trusts] for b in backbones]
    fig, ax = plt.subplots(
        figsize=(1.2 * len(trusts) + 1.5, 0.5 * len(backbones) + 1)
    )
    vmin = min(min(row) for row in data)
    vmax = max(max(row) for row in data)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(trusts)))
    ax.set_xticklabels(trusts, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(backbones)))
    ax.set_yticklabels(backbones, fontsize=8)
    for i in range(len(backbones)):
        for j in range(len(trusts)):
            ax.text(
                j, i, f"{data[i][j]:.2f}", ha="center", va="center",
                fontsize=7, fontweight="bold",
            )
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Avg Reward")
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        fig.savefig(tf, format="png", dpi=150)
        tmpname = tf.name
    plt.close(fig)
    run.log({key: wandb.Image(tmpname)})
    os.unlink(tmpname)


def log_bar_chart(run, key: str, names: list, values: list, title: str, ylabel: str = "Score"):
    """Log a bar chart comparing methods."""
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.9), 4))
    colors = [
        "#55A868" if v == max(values) else "#C44E52" if v == min(values) else "#4C72B0"
        for v in values
    ]
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.4f}",
                ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        fig.savefig(tf, format="png", dpi=150)
        tmpname = tf.name
    plt.close(fig)
    run.log({key: wandb.Image(tmpname)})
    os.unlink(tmpname)


def collect_eval_frames(env, policy, max_steps, device):
    """Run one eval episode, return (frames, total_reward, n_steps)."""
    import torch
    frames = []
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float32).flatten()
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    total_r = 0.0
    for step in range(max_steps):
        frame = env.render()
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        if isinstance(frame, torch.Tensor):
            frame = frame.numpy()
        if frame.ndim == 4:
            frame = frame[0]
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frames.append(frame)
        with torch.no_grad():
            act = policy(obs_t)
        no, r, term, trunc, _ = env.step(act.cpu().numpy().flatten())
        total_r += float(r)
        obs_t = torch.from_numpy(np.asarray(no, dtype=np.float32).flatten()).float().unsqueeze(0).to(device)
        if term or trunc:
            break
    # Capture final frame
    frame = env.render()
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    if isinstance(frame, torch.Tensor):
        frame = frame.numpy()
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    frames.append(frame)
    return frames, total_r, len(frames)
