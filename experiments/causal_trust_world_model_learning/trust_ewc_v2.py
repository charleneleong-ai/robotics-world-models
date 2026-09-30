"""Per-sample trust weighting + trust-weighted online Fisher EWC."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

Batch = tuple[Tensor, Tensor, Tensor]
Demo = dict[str, np.ndarray]
Phases = list[list[Demo]]
EPS = 1e-8
BATCH_SIZE = 64


def per_sample_error(wm: nn.Module, obs: Tensor, act: Tensor, nxt: Tensor) -> Tensor:
    # full_backbone_sweep decorates predict_error with @torch.no_grad(); unwrap it to train.
    if hasattr(wm, "rssm"):
        return per_sample_error(wm.rssm, obs, act, nxt)
    fn = type(wm).predict_error
    err = getattr(fn, "__wrapped__", fn)(wm, obs, act, nxt)
    return err.reshape(err.shape[0], -1).mean(dim=1)


def transition_batches(demos: list[Demo], batch_size: int, device: torch.device) -> Iterator[Batch]:
    obs = np.concatenate([d["obs"][:-1] for d in demos])
    act = np.concatenate([d["actions"][:-1] for d in demos])
    nxt = np.concatenate([d["obs"][1:] for d in demos])
    perm = np.random.permutation(len(obs))
    for i in range(0, len(obs), batch_size):
        idx = perm[i : i + batch_size]
        yield tuple(torch.as_tensor(x[idx], device=device) for x in (obs, act, nxt))


class RunningTrust:
    """Per-sample trust from a z-score of error against running error statistics."""

    def __init__(self, decay: float = 0.99, temperature: float = 1.0) -> None:
        self.decay = decay
        self.temperature = temperature
        self.mean: Tensor | None = None
        self.var: Tensor | None = None

    def raw(self, err: Tensor) -> Tensor:
        if self.mean is None:
            return torch.full_like(err, 0.5)
        # Log scale: raw-error variance is dominated by early large errors and collapses z.
        z = (err.clamp_min(EPS).log() - self.mean) / (self.var + EPS).sqrt()
        return torch.sigmoid(-z / self.temperature)

    def weights(self, err: Tensor) -> Tensor:
        w = self.raw(err)
        return w / w.mean().clamp_min(EPS)

    def global_trust(self, err: Tensor, floor: float = 0.05) -> float:
        return self.raw(err).mean().clamp_min(floor).item()

    def update(self, err: Tensor) -> None:
        log_err = err.clamp_min(EPS).log()
        m, v = log_err.mean(), log_err.var(unbiased=False)
        if self.mean is None:
            self.mean, self.var = m, v
            return
        self.mean = self.decay * self.mean + (1 - self.decay) * m
        self.var = self.decay * self.var + (1 - self.decay) * v


class OnlineFisherEWC:
    """Diagonal empirical-Fisher EWC, accumulated across tasks (online EWC)."""

    def __init__(self, gamma: float = 1.0, max_samples: int = 512) -> None:
        self.gamma = gamma
        self.max_samples = max_samples
        self.fisher: dict[str, Tensor] = {}
        self.anchor: dict[str, Tensor] = {}

    def penalty(self, model: nn.Module) -> Tensor:
        terms = [
            (self.fisher[n] * (p - self.anchor[n]).pow(2)).sum()
            for n, p in model.named_parameters()
            if n in self.fisher
        ]
        return 0.5 * torch.stack(terms).sum() if terms else torch.zeros(())

    def consolidate(
        self, model: nn.Module, batches: Iterable[Batch], trust: RunningTrust | None
    ) -> None:
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        total_weight = 0.0
        for obs, act, nxt in batches:
            err = per_sample_error(model, obs, act, nxt)
            w = trust.weights(err.detach()) if trust else torch.ones_like(err)
            for e, wi in zip(err, w):
                model.zero_grad()
                e.backward(retain_graph=True)
                for n, p in params.items():
                    if p.grad is not None:
                        fisher[n] += wi * p.grad.detach().pow(2)
                total_weight += wi.item()
                if total_weight >= self.max_samples:
                    break
            if total_weight >= self.max_samples:
                break
        model.zero_grad()
        for n, p in params.items():
            new = fisher[n] / max(total_weight, EPS)
            self.fisher[n] = self.gamma * self.fisher.get(n, torch.zeros_like(new)) + new
            self.anchor[n] = p.detach().clone()


@dataclass(frozen=True)
class Arm:
    name: str
    sample_weighting: bool = False
    trust_fisher: bool = False
    adaptive_lambda: bool = False


ARMS = [
    Arm("online_ewc"),
    Arm("adaptive_lambda", adaptive_lambda=True),
    Arm("trust_sample", sample_weighting=True),
    Arm("trust_fisher", trust_fisher=True),
    Arm("trust_full", sample_weighting=True, trust_fisher=True),
]


def train_task(
    wm: nn.Module,
    optimizer: torch.optim.Optimizer,
    ewc: OnlineFisherEWC,
    trust: RunningTrust,
    batches: Iterable[Batch],
    lam: float,
    arm: Arm,
) -> list[float]:
    spreads: list[float] = []
    for obs, act, nxt in batches:
        err = per_sample_error(wm, obs, act, nxt)
        detached = err.detach()
        if len(detached) > 1:
            spreads.append(trust.raw(detached).std().item())
        w = trust.weights(detached) if arm.sample_weighting else torch.ones_like(err)
        # Reviewers' reduction of v1: trust * recon + λ·EWC  ≡  recon + (λ / trust̄)·EWC.
        lam_t = lam / trust.global_trust(detached) if arm.adaptive_lambda else lam
        trust.update(detached)
        loss = (w * err).mean() + lam_t * ewc.penalty(wm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return spreads


@torch.no_grad()
def eval_error(wm: nn.Module, batches: Iterable[Batch]) -> float:
    was_training = wm.training
    wm.eval()
    errs = [per_sample_error(wm, *b) for b in batches]
    wm.train(was_training)
    return torch.cat(errs).mean().item()


@dataclass(frozen=True)
class SequenceResult:
    errors: np.ndarray
    trust_spread: float

    @property
    def final_avg_error(self) -> float:
        return float(self.errors[-1].mean())

    @property
    def backward_transfer(self) -> float:
        n = self.errors.shape[0]
        return float(np.mean([self.errors[n - 1, j] - self.errors[j, j] for j in range(n - 1)]))

    @property
    def plasticity(self) -> float:
        return float(np.diag(self.errors).mean())


def run_sequence(
    wm: nn.Module,
    train_tasks: Phases,
    eval_tasks: Phases,
    arm: Arm,
    lam: float,
    epochs: int = 50,
    batch_size: int = BATCH_SIZE,
    lr: float = 1e-3,
) -> SequenceResult:
    device = next(wm.parameters()).device
    optimizer = torch.optim.Adam(wm.parameters(), lr=lr)
    ewc, trust = OnlineFisherEWC(), RunningTrust()
    n = len(train_tasks)
    errors = np.zeros((n, n))
    spreads: list[float] = []
    for i, demos in enumerate(train_tasks):
        for _ in range(epochs):
            batches = transition_batches(demos, batch_size, device)
            spreads += train_task(wm, optimizer, ewc, trust, batches, lam, arm)
        consolidation_batches = transition_batches(demos, batch_size, device)
        ewc.consolidate(wm, consolidation_batches, trust if arm.trust_fisher else None)
        for j in range(i + 1):
            errors[i, j] = eval_error(wm, transition_batches(eval_tasks[j], batch_size, device))
    return SequenceResult(errors, float(np.mean(spreads)))
