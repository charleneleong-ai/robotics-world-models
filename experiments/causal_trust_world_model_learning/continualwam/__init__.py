"""ContinualWAM — trust-aware continual learning for world models."""

from continualwam.backbones import BACKBONES, get_backbone
from continualwam.trust import TRUST_METHODS, make_trust
from continualwam.training import train_wm, train_bc, train_bc_trust, load_demos
from continualwam.evaluation import eval_bc

__all__ = [
    "BACKBONES", "get_backbone",
    "TRUST_METHODS", "make_trust",
    "train_wm", "train_bc", "train_bc_trust",
    "eval_bc", "load_demos",
]
