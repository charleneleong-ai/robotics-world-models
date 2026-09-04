"""Experiment configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SweepConfig:
    """Configuration for a backbone × trust sweep."""
    suite_dir: str
    n_tasks: int = 10
    max_demos: int = 5
    n_seeds: int = 5
    wm_epochs: int = 20
    bc_epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cuda"
    backbones: list[str] = field(default_factory=lambda: ["mlp", "rssm", "jepa", "dreamerv3", "diffusion", "transformer"])
    trust_methods: list[str] = field(default_factory=lambda: ["ema", "multi_step", "ensemble"])
    output_dir: str = "."

    @classmethod
    def from_yaml(cls, path: str | Path) -> SweepConfig:
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class DecoderTrustConfig:
    """Configuration for decoder-aware trust experiment."""
    suite_dir: str
    n_tasks: int = 10
    max_demos: int = 5
    n_seeds: int = 5
    wm_epochs: int = 50
    bc_epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cuda"
    output_dir: str = "."


@dataclass
class AuditConfig:
    """Configuration for audit experiments."""
    suite_dir: str
    n_tasks: int = 10
    max_demos: int = 5
    n_seeds: int = 3
    wm_epochs: int = 20
    bc_epochs: int = 50
    device: str = "cuda"
    output_dir: str = "."
