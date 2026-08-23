"""Diffusion world model for ManiSkill3 tasks.

Modules:
    - model: Conditional diffusion dynamics (MLP denoiser, DiT)
    - train: Training loop with W&B logging
    - dataset: Replay buffer dataset
    - eval: Evaluation metrics (MSE/MAE, action consistency, smoothness)
    - collect: Data collection from ManiSkill3 environments
    - domain_rand: Domain randomization for sim-to-real
    - video_metrics: Video-level evaluation (FVD, temporal LPIPS)
    - fidelity: Prediction calibration and divergence detection
    - system_id: System identification for parameter calibration
    - residual_dynamics: Residual dynamics for sim-to-real gap
    - transfer: Complete sim-to-real transfer pipeline
"""

from .model import MLPDenoiser, DiffusionDynamics, cosine_beta_schedule
from .domain_rand import (
    DomainRandomizationConfig,
    PhysicsRandomization,
    ObservationNoise,
    ActionNoise,
    apply_physics_randomization,
    apply_observation_noise,
    apply_action_noise,
)
from .video_metrics import (
    compute_fvd,
    compute_temporal_lpips,
    compute_all_video_metrics,
    VideoMetricsResult,
)
from .fidelity import (
    PredictionCalibration,
    DivergenceDetector,
    compute_trust_from_divergence,
)
from .system_id import (
    SystemIdentifier,
    ParameterEstimator,
    SystemIdentificationResult,
)
from .residual_dynamics import (
    ResidualDynamicsNet,
    HybridDynamicsModel,
    OnlineResidualAdapter,
    create_hybrid_model,
)
from .transfer import (
    SimToRealPipeline,
    TransferConfig,
    TransferResult,
    run_full_transfer,
)

__all__ = [
    # Model
    "MLPDenoiser",
    "DiffusionDynamics",
    "cosine_beta_schedule",
    # Domain randomization
    "DomainRandomizationConfig",
    "PhysicsRandomization",
    "ObservationNoise",
    "ActionNoise",
    "apply_physics_randomization",
    "apply_observation_noise",
    "apply_action_noise",
    # Video metrics
    "compute_fvd",
    "compute_temporal_lpips",
    "compute_all_video_metrics",
    "VideoMetricsResult",
    # Fidelity
    "PredictionCalibration",
    "DivergenceDetector",
    "compute_trust_from_divergence",
    # System ID
    "SystemIdentifier",
    "ParameterEstimator",
    "SystemIdentificationResult",
    # Residual dynamics
    "ResidualDynamicsNet",
    "HybridDynamicsModel",
    "OnlineResidualAdapter",
    "create_hybrid_model",
    # Transfer
    "SimToRealPipeline",
    "TransferConfig",
    "TransferResult",
    "run_full_transfer",
]
