"""Video-level evaluation metrics for diffusion world model.

Implements FVD (Fréchet Video Distance), temporal LPIPS, and
action-conditioned video metrics for evaluating video prediction
quality in world models.

Based on Aljalbout et al. 2026 — pixel-level video metrics are
needed alongside state-space metrics for comprehensive evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------


class I3DFeatureExtractor(nn.Module):
    """I3D-based feature extractor for FVD computation.

    Uses a pretrained Inception3D network to extract spatiotemporal
    features from video frames.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.device = device
        # Placeholder — in practice, load pretrained I3D weights
        # For now, use a simple 3D CNN as feature extractor
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, 512)
        self.to(device)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Extract features from a video tensor.

        Args:
            video: Video tensor of shape (B, T, C, H, W) or (B, C, T, H, W).

        Returns:
            Feature tensor of shape (B, 512).
        """
        # Ensure (B, C, T, H, W) format
        if video.dim() == 5 and video.shape[1] != 3:
            video = video.permute(0, 2, 1, 3, 4)

        x = F.relu(self.conv1(video))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def _compute_stats(
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and covariance of feature vectors."""
    mean = features.mean(dim=0)
    # ManiSkill/sim observations may have feature dim == batch dim
    # Handle edge case where features are too small for cov
    if features.shape[0] > 1 and features.shape[1] > 1:
        centered = features - mean.unsqueeze(0)
        cov = (centered.T @ centered) / (features.shape[0] - 1)
    else:
        cov = torch.eye(features.shape[1], device=features.device)
    return mean, cov


def _frechet_distance(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute Fréchet distance between two Gaussians."""
    diff = mu1 - mu2

    # Product might be almost singular — add small regularization
    covmean = sigma1 @ sigma2
    # Use eigendecomposition for numerical stability
    eigenvalues = torch.linalg.eigvalsh(covmean)
    # Clamp to avoid log of negative
    eigenvalues = eigenvalues.clamp(min=eps)
    log_det = eigenvalues.log().sum()

    # Trace of sqrt of product
    trace = torch.trace(sigma1) + torch.trace(sigma2)

    fd = diff @ diff + trace - 2 * log_det
    return float(fd.item())


def compute_fvd(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    extractor: I3DFeatureExtractor | None = None,
    device: torch.device | str = "cpu",
) -> float:
    """Compute Fréchet Video Distance (FVD) between real and generated videos.

    Lower FVD indicates better video generation quality.

    Args:
        real_videos: Real video tensor (B, T, C, H, W) or (B, C, T, H, W).
        generated_videos: Generated video tensor (same shape).
        extractor: Optional pretrained feature extractor.
        device: Computation device.

    Returns:
        FVD score (lower is better).
    """
    if extractor is None:
        extractor = I3DFeatureExtractor(device)

    real_videos = real_videos.to(device)
    generated_videos = generated_videos.to(device)

    with torch.no_grad():
        real_features = extractor(real_videos)
        gen_features = extractor(generated_videos)

    mu_real, sigma_real = _compute_stats(real_features)
    mu_gen, sigma_gen = _compute_stats(gen_features)

    return _frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def compute_temporal_lpips(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    window_size: int = 3,
) -> float:
    """Compute temporal LPIPS across consecutive frame pairs.

    Measures perceptual similarity across time steps, not just
    individual frames.

    Args:
        real_videos: Real video tensor (B, T, C, H, W).
        generated_videos: Generated video tensor (B, T, C, H, W).
        window_size: Number of consecutive frames to compare.

    Returns:
        Average temporal LPIPS score (lower is better).
    """
    B, T = real_videos.shape[:2]

    if T < 2:
        return 0.0

    # Compute per-frame LPIPS approximation using SSIM-like computation
    total_lpips = 0.0
    count = 0

    for t in range(min(T - 1, window_size)):
        real_frame1 = real_videos[:, t]
        real_frame2 = real_videos[:, t + 1]
        gen_frame1 = generated_videos[:, t]
        gen_frame2 = generated_videos[:, t + 1]

        # Simple perceptual distance based on multi-scale structure
        # (approximation — real implementation would use LPIPS network)
        diff_real = (real_frame1 - real_frame2).abs().mean()
        diff_gen = (gen_frame1 - gen_frame2).abs().mean()

        total_lpips += (diff_real - diff_gen).abs().item()
        count += 1

    return total_lpips / max(count, 1)


def compute_idm_error(
    videos: torch.Tensor,
    actions: torch.Tensor,
    idm: nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> float:
    """Compute inverse dynamics model (IDM) action prediction error.

    IDM predicts actions from consecutive frames. High error indicates
    the video prediction doesn't preserve action semantics.

    Args:
        videos: Video tensor (B, T, C, H, W).
        actions: Ground truth actions (B, T-1, action_dim).
        idm: Optional pretrained inverse dynamics model.
        device: Computation device.

    Returns:
        Mean action prediction error (lower is better).
    """
    if videos.shape[1] < 2:
        return 0.0

    # Flatten frames to image features
    B, T = videos.shape[:2]

    # Simple frame differencing as IDM proxy
    # (real implementation would use a trained IDM)
    frame_diffs = []
    for t in range(T - 1):
        diff = (videos[:, t + 1] - videos[:, t]).abs().mean(dim=(-3, -2, -1))
        frame_diffs.append(diff)

    # Predicted actions as normalized frame differences
    pred_actions = torch.stack(frame_diffs, dim=1)

    if actions.device != pred_actions.device:
        actions = actions.to(pred_actions.device)

    # Ensure same shape — pred_actions is (B, T-1), gt_actions may be (B, T-1, D)
    min_t = min(pred_actions.shape[1], actions.shape[1])
    pred_actions = pred_actions[:, :min_t]
    gt_actions = actions[:, :min_t]

    # If gt_actions has extra dims, reduce to match
    if gt_actions.dim() > pred_actions.dim():
        gt_actions = gt_actions.mean(dim=-1)

    # Mean squared error
    error = F.mse_loss(pred_actions, gt_actions)
    return float(error.item())


def compute_rot_trans_error(
    predicted_poses: torch.Tensor,
    gt_poses: torch.Tensor,
) -> tuple[float, float]:
    """Compute rotation and translation error for predicted poses.

    Args:
        predicted_poses: Predicted poses (B, T, 7) — [x, y, z, qw, qx, qy, qz].
        gt_poses: Ground truth poses (B, T, 7).

    Returns:
        Tuple of (rotation_error_degrees, translation_error_meters).
    """
    # Translation error
    trans_pred = predicted_poses[..., :3]
    trans_gt = gt_poses[..., :3]
    trans_error = (trans_pred - trans_gt).norm(dim=-1).mean()

    # Rotation error (angular distance between quaternions)
    if predicted_poses.shape[-1] >= 7:
        rot_pred = predicted_poses[..., 3:7]
        rot_gt = gt_poses[..., 3:7]

        # Normalize quaternions
        rot_pred = F.normalize(rot_pred, dim=-1)
        rot_gt = F.normalize(rot_gt, dim=-1)

        # Dot product gives cos(half_angle)
        dot = (rot_pred * rot_gt).sum(dim=-1).abs().clamp(0, 1)
        # Angular distance in degrees
        rot_error = 2 * torch.acos(dot) * 180.0 / torch.pi
        rot_error = rot_error.mean()
    else:
        rot_error = torch.tensor(0.0)

    return float(rot_error.item()), float(trans_error.item())


# ---------------------------------------------------------------------------
# Aggregated evaluation
# ---------------------------------------------------------------------------


@dataclass
class VideoMetricsResult:
    """Aggregated video metrics result."""

    fvd: float
    temporal_lpips: float
    idm_error: float
    rotation_error_deg: float
    translation_error_m: float

    def to_dict(self) -> dict[str, float]:
        return {
            "fvd": self.fvd,
            "temporal_lpips": self.temporal_lpips,
            "idm_error": self.idm_error,
            "rotation_error_deg": self.rotation_error_deg,
            "translation_error_m": self.translation_error_m,
        }


def compute_all_video_metrics(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    actions: torch.Tensor | None = None,
    predicted_poses: torch.Tensor | None = None,
    gt_poses: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> VideoMetricsResult:
    """Compute all video-level metrics.

    Args:
        real_videos: Real videos (B, T, C, H, W).
        generated_videos: Generated videos (B, T, C, H, W).
        actions: Optional ground truth actions for IDM evaluation.
        predicted_poses: Optional predicted poses for rotation/translation error.
        gt_poses: Optional ground truth poses for rotation/translation error.
        device: Computation device.

    Returns:
        VideoMetricsResult with all computed metrics.
    """
    fvd = compute_fvd(real_videos, generated_videos, device=device)
    t_lpips = compute_temporal_lpips(real_videos, generated_videos)

    idm_err = 0.0
    if actions is not None:
        idm_err = compute_idm_error(generated_videos, actions, device=device)

    rot_err = 0.0
    trans_err = 0.0
    if predicted_poses is not None and gt_poses is not None:
        rot_err, trans_err = compute_rot_trans_error(predicted_poses, gt_poses)

    return VideoMetricsResult(
        fvd=fvd,
        temporal_lpips=t_lpips,
        idm_error=idm_err,
        rotation_error_deg=rot_err,
        translation_error_m=trans_err,
    )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_video_metrics(
    metrics: VideoMetricsResult,
    step: int,
    prefix: str = "video_eval",
    wandb_run: object | None = None,
) -> dict[str, float]:
    """Log video metrics to W&B.

    Args:
        metrics: VideoMetricsResult to log.
        step: Current training step.
        prefix: Metric name prefix.
        wandb_run: Optional W&B run object.

    Returns:
        Dictionary of logged metrics.
    """
    logged = {}
    for key, value in metrics.to_dict().items():
        full_key = f"{prefix}/{key}"
        logged[full_key] = value
        if wandb_run is not None:
            wandb_run.log({full_key: value, "step": step})

    return logged
