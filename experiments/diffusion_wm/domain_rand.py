"""Domain randomization for sim-to-real transfer.

Applies physics parameter randomization, observation noise, and action
noise during data collection and training to build invariance to
sim-to-real discrepancies.

Based on Aljalbout et al. 2026 "The Reality Gap in Robotics" —
domain randomization is the highest-impact, lowest-effort sim-to-real
technique.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PhysicsRandomization:
    """Ranges for physics parameter randomization during data collection.

    Values are multipliers applied to the simulator's default parameters.
    e.g. friction=(0.5, 2.0) means friction is sampled from
    Uniform(default * 0.5, default * 2.0).
    """

    friction: tuple[float, float] = (0.5, 2.0)
    mass: tuple[float, float] = (0.8, 1.2)
    damping: tuple[float, float] = (0.5, 1.5)
    stiffness: tuple[float, float] = (0.8, 1.2)
    contact_stiffness: tuple[float, float] = (0.5, 2.0)
    contact_damping: tuple[float, float] = (0.5, 2.0)


@dataclass(frozen=True)
class ObservationNoise:
    """Gaussian noise added to observations during training.

    Noise is additive: obs_noisy = obs + N(0, noise_scale²) * obs_scale.
    """

    position_noise: float = 0.01
    velocity_noise: float = 0.02
    orientation_noise: float = 0.01
    rgb_noise: float = 0.03
    depth_noise: float = 0.02
    enabled: bool = True


@dataclass(frozen=True)
class ActionNoise:
    """Gaussian noise added to actions during training."""

    torque_noise: float = 0.05
    position_noise: float = 0.02
    enabled: bool = True


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Complete domain randomization configuration."""

    physics: PhysicsRandomization = field(default_factory=PhysicsRandomization)
    observation: ObservationNoise = field(default_factory=ObservationNoise)
    action: ActionNoise = field(default_factory=ActionNoise)
    seed: int | None = None


def _sample_uniform(range_tuple: tuple[float, float], rng: torch.Generator) -> float:
    """Sample from uniform distribution using PyTorch generator."""
    low, high = range_tuple
    return float(torch.empty(1).uniform_(low, high, generator=rng).item())


def apply_physics_randomization(
    env: Any,
    config: DomainRandomizationConfig,
    rng: torch.Generator | None = None,
) -> dict[str, float]:
    """Apply physics parameter randomization to a ManiSkill environment.

    This should be called BEFORE env.reset() to randomize the physics
    parameters for the next episode.

    Args:
        env: ManiSkill3 environment instance.
        config: Domain randomization configuration.
        rng: Optional seeded generator for reproducibility.

    Returns:
        Dictionary of sampled parameter values for logging.
    """
    if rng is None:
        rng = torch.Generator()
        rng.manual_seed(int(torch.empty((), dtype=torch.long).random_().item()))

    sampled: dict[str, float] = {}

    try:
        # ManiSkill3 exposes physics parameters through env.unwrapped
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

        # Randomize friction
        if hasattr(unwrapped, "friction"):
            factor = _sample_uniform(config.physics.friction, rng)
            unwrapped.friction *= factor
            sampled["friction_factor"] = factor

        # Randomize mass (via link mass properties)
        if hasattr(unwrapped, "link_mass"):
            factor = _sample_uniform(config.physics.mass, rng)
            unwrapped.link_mass *= factor
            sampled["mass_factor"] = factor

        # Randomize damping
        if hasattr(unwrapped, "damping"):
            factor = _sample_uniform(config.physics.damping, rng)
            unwrapped.damping *= factor
            sampled["damping_factor"] = factor

        # Randomize stiffness
        if hasattr(unwrapped, "stiffness"):
            factor = _sample_uniform(config.physics.stiffness, rng)
            unwrapped.stiffness *= factor
            sampled["stiffness_factor"] = factor

        # Randomize contact parameters
        if hasattr(unwrapped, "contact_stiffness"):
            factor = _sample_uniform(config.physics.contact_stiffness, rng)
            unwrapped.contact_stiffness *= factor
            sampled["contact_stiffness_factor"] = factor

        if hasattr(unwrapped, "contact_damping"):
            factor = _sample_uniform(config.physics.contact_damping, rng)
            unwrapped.contact_damping *= factor
            sampled["contact_damping_factor"] = factor

    except AttributeError:
        # Environment doesn't expose physics parameters directly
        # Fall through silently — randomization not applicable
        pass

    return sampled


def apply_observation_noise(
    obs: torch.Tensor,
    config: DomainRandomizationConfig,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise to observations during training.

    Args:
        obs: Observation tensor from environment.
        config: Domain randomization configuration.
        rng: Optional seeded generator for reproducibility.

    Returns:
        Noisy observation tensor.
    """
    if not config.observation.enabled:
        return obs

    if rng is None:
        rng = torch.Generator(device=obs.device)
        rng.manual_seed(int(torch.empty((), dtype=torch.long, device=obs.device).random_().item()))

    noisy = obs.clone()

    # Determine observation dimension and apply appropriate noise
    # Assumes flat observation vector: [pos, vel, quat, ...]
    obs_dim = obs.shape[-1] if obs.dim() > 1 else obs.shape[0]

    # Position noise (first 3 dims typically)
    if obs_dim >= 3:
        noise = torch.randn(3, generator=rng, device=obs.device) * config.observation.position_noise
        noisy[..., :3] += noise

    # Velocity noise (next 3 dims typically)
    if obs_dim >= 6:
        noise = torch.randn(3, generator=rng, device=obs.device) * config.observation.velocity_noise
        noisy[..., 3:6] += noise

    # Orientation noise (next 4 dims typically — quaternion)
    if obs_dim >= 10:
        noise = torch.randn(4, generator=rng, device=obs.device) * config.observation.orientation_noise
        noisy[..., 6:10] += noise

    return noisy


def apply_action_noise(
    action: torch.Tensor,
    config: DomainRandomizationConfig,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise to actions during training.

    Args:
        action: Action tensor from environment.
        config: Domain randomization configuration.
        rng: Optional seeded generator for reproducibility.

    Returns:
        Noisy action tensor.
    """
    if not config.action.enabled:
        return action

    if rng is None:
        rng = torch.Generator(device=action.device)
        rng.manual_seed(int(torch.empty((), dtype=torch.long, device=action.device).random_().item()))

    noisy = action.clone()

    # Apply torque noise to all action dimensions
    noise_scale = max(config.action.torque_noise, config.action.position_noise)
    noise = torch.randn_like(action, generator=rng) * noise_scale
    noisy += noise

    return noisy


def randomize_and_collect(
    env: Any,
    config: DomainRandomizationConfig,
    num_steps: int = 1000,
) -> dict[str, Any]:
    """Run one episode with domain randomization and collect transitions.

    Args:
        env: ManiSkill3 environment.
        config: Domain randomization configuration.
        num_steps: Number of steps to collect.

    Returns:
        Dictionary with transitions and randomized parameters.
    """
    rng = torch.Generator()
    rng.manual_seed(int(torch.empty((), dtype=torch.long).random_().item()))

    # Apply physics randomization before reset
    sampled_params = apply_physics_randomization(env, config, rng)

    obs, info = env.reset()
    transitions: list[dict[str, torch.Tensor]] = []

    for step in range(num_steps):
        # Add observation noise
        noisy_obs = apply_observation_noise(obs, config, rng)

        # Get action from policy (placeholder — in practice, use your policy)
        action = env.action_space.sample()
        action_tensor = torch.tensor(action, dtype=torch.float32)

        # Add action noise
        noisy_action = apply_action_noise(action_tensor, config, rng)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(noisy_action.numpy())

        transitions.append(
            {
                "obs": noisy_obs,
                "action": noisy_action,
                "next_obs": next_obs,
                "reward": torch.tensor(reward, dtype=torch.float32),
                "terminated": torch.tensor(terminated, dtype=torch.bool),
                **{f"physics_{k}": torch.tensor(v) for k, v in sampled_params.items()},
            }
        )

        obs = next_obs
        if terminated or truncated:
            break

    return {"transitions": transitions, "sampled_params": sampled_params}


# Default configs for ManiSkill3 tasks
TASK_DEFAULTS: dict[str, DomainRandomizationConfig] = {
    "PickCube-v1": DomainRandomizationConfig(
        physics=PhysicsRandomization(
            friction=(0.5, 2.0),
            mass=(0.8, 1.2),
            damping=(0.5, 1.5),
        ),
        observation=ObservationNoise(
            position_noise=0.01,
            velocity_noise=0.02,
            orientation_noise=0.01,
        ),
        action=ActionNoise(torque_noise=0.05, position_noise=0.02),
    ),
    "PegInsertionSide-v1": DomainRandomizationConfig(
        physics=PhysicsRandomization(
            friction=(0.3, 2.5),
            mass=(0.7, 1.3),
            damping=(0.4, 1.6),
            stiffness=(0.7, 1.3),
        ),
        observation=ObservationNoise(
            position_noise=0.005,
            velocity_noise=0.015,
            orientation_noise=0.008,
        ),
        action=ActionNoise(torque_noise=0.03, position_noise=0.015),
    ),
    "StackCube-v1": DomainRandomizationConfig(
        physics=PhysicsRandomization(
            friction=(0.4, 2.2),
            mass=(0.8, 1.2),
            damping=(0.5, 1.5),
        ),
        observation=ObservationNoise(
            position_noise=0.01,
            velocity_noise=0.02,
            orientation_noise=0.01,
        ),
        action=ActionNoise(torque_noise=0.04, position_noise=0.02),
    ),
    "PlugCharger-v1": DomainRandomizationConfig(
        physics=PhysicsRandomization(
            friction=(0.3, 2.5),
            mass=(0.7, 1.3),
            damping=(0.4, 1.6),
            stiffness=(0.7, 1.3),
            contact_stiffness=(0.3, 3.0),
            contact_damping=(0.3, 3.0),
        ),
        observation=ObservationNoise(
            position_noise=0.003,
            velocity_noise=0.01,
            orientation_noise=0.005,
        ),
        action=ActionNoise(torque_noise=0.02, position_noise=0.01),
    ),
}
