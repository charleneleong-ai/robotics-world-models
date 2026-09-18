#!/usr/bin/env python3
"""Does a better representation forget less?

The comparison probes asked which representation clones best on a single task. This asks
the question the continual-learning papers in this repository actually care about: train
one action head sequentially over the LIBERO-Spatial tasks and measure what it loses.

Protocol follows the task-order sensitivity design used elsewhere in this repo. Random
orderings, seeds nested inside an ordering and averaged before testing, arms paired on
identical orderings. Metrics are absolute: final error over all seen tasks, and absolute
forgetting (a task's error at the end of the sequence minus its error just after it was
trained). The first-task-normalised forgetting *rate* is deliberately not used -- an
earlier paper here reported a 40.8% reduction in it that turned out to be an artifact of
reliability weighting inflating the denominator.

Evaluation holds out whole demonstrations (`eval_demos` per task, default 2), and every
fitted statistic -- z-scoring and the PCA control alike -- sees training frames only, so
these numbers measure retention of behaviour rather than of training frames.

p-values are paired across orderings and uncorrected across arms.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import typer
import wandb
from scipy import stats
from sklearn.decomposition import PCA
from torch import Tensor, nn

ROOT = Path("/home/ubuntu/wan_latents")
CACHES = {"vae": "spatial_agentview_rgb", "dit_ctx1": "spatial_dit_ctx1", "dit_ctx8": "spatial_dit_ctx8",
          "dit_ctx16": "spatial_dit_ctx16", "siglip2": "spatial_siglip2", "siglip2_ctx8": "spatial_siglip2_ctx8",
          "openvla": "spatial_openvla", "openvla_ctx8": "spatial_openvla_ctx8"}
STATE = "state"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Task:
    """One LIBERO task's frames, split into train and held-out demonstrations.

    Not frozen: FeatureStore.project writes width-matched copies into `features`.
    """

    index: int
    action: np.ndarray
    eval_mask: np.ndarray
    features: dict[str, np.ndarray]

    def split(self, rep: str, norm: Normaliser) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = norm(self.features[rep])
        return (self.pick(x, train=True), self.pick(self.action, train=True),
                self.pick(x, train=False), self.pick(self.action, train=False))

    def pick(self, array: np.ndarray, train: bool) -> Tensor:
        return torch.tensor(array[~self.eval_mask if train else self.eval_mask], device=DEVICE)

    def train_frames(self, rep: str) -> np.ndarray:
        return self.features[rep][~self.eval_mask]

    @property
    def action_dim(self) -> int:
        return self.action.shape[1]


@dataclass(frozen=True)
class Normaliser:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def over(cls, blocks: list[np.ndarray]) -> Normaliser:
        stacked = np.concatenate(blocks)
        return cls(stacked.mean(0, keepdims=True), stacked.std(0, keepdims=True) + 1e-6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


class FeatureStore:
    """The cached representations on disk, plus proprioceptive state as the reference arm."""

    def __init__(self, root: Path, n_tasks: int, n_eval_demos: int, pca_dim: int = 0) -> None:
        self.sources = {k: root / v for k, v in CACHES.items() if (root / v / "task00.npz").exists()}
        if not self.sources:
            raise FileNotFoundError(f"no feature caches found under {root}")
        self.tasks = [self.read(i, n_eval_demos) for i in range(n_tasks)]
        self.derived: list[str] = []
        if pca_dim:
            self.project(pca_dim)
        self.norms = {r: Normaliser.over([t.train_frames(r) for t in self.tasks]) for r in self.representations}

    def read(self, index: int, n_eval_demos: int) -> Task:
        features: dict[str, np.ndarray] = {}
        action = state = demo = None
        for name, path in self.sources.items():
            z = np.load(path / f"task{index:02d}.npz")
            features[name] = z["latent"].astype(np.float32)
            if action is None:
                action, state, demo = z["action"].astype(np.float32), z["state"].astype(np.float32), z["demo"]
            assert len(features[name]) == len(demo), f"{name} task{index} has {len(features[name])} of {len(demo)}"
        features[STATE] = state
        held = set(np.unique(demo)[-n_eval_demos:].tolist())
        return Task(index, action, np.array([d in held for d in demo]), features)

    def project(self, dim: int) -> None:
        """Add a copy of every cached representation reduced to `dim` dimensions.

        Proprioceptive state is 21 numbers and the backbones are thousands, so any gap
        between them could be about input width rather than content -- pass the state's
        width here to remove that explanation. Fitting on training frames only keeps the
        held-out demonstrations out of the projection. State itself is left alone.
        """
        for rep in list(self.sources):
            pca = PCA(n_components=dim, random_state=0).fit(
                np.concatenate([t.train_frames(rep) for t in self.tasks]))
            name = f"{rep}_pca{dim}"
            for task in self.tasks:
                task.features[name] = pca.transform(task.features[rep]).astype(np.float32)
            self.derived.append(name)

    @property
    def representations(self) -> list[str]:
        return [STATE, *self.sources, *self.derived]

    def dim(self, rep: str) -> int:
        return self.tasks[0].features[rep].shape[1]


class ActionHead(nn.Module):
    """Small MLP mapping a frozen representation to an action, the only thing that learns."""

    def __init__(self, in_dim: int, out_dim: int, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class SequenceResult:
    """Outcome of training one head through one ordering of the task sequence.

    `errors[pos, step]` is the held-out error on the task at sequence position `pos`
    measured after the task at position `step` finished training, so only the upper
    triangle (pos <= step) is filled.
    """

    errors: np.ndarray

    @property
    def just_after(self) -> np.ndarray:
        """Each task's error the moment it finished training, before anything overwrote it."""
        return np.diag(self.errors)

    @property
    def final_error(self) -> float:
        return float(self.errors[:, -1].mean())

    @property
    def abs_forgetting(self) -> float:
        """The last task is excluded: nothing has been trained since, so it cannot have forgotten."""
        return float((self.errors[:-1, -1] - self.just_after[:-1]).mean())

    @property
    def first_task_error(self) -> float:
        return float(self.just_after[0])

    @property
    def plasticity(self) -> float:
        """Mean error on each task the moment it finished training -- lower is more plastic.

        This is a loss, so it runs opposite to its name. It is the guard against a regulariser
        that wins by refusing to move: a head frozen after task 1 forgets nothing and scores
        terribly here. Unlike `abs_forgetting` it includes the final task, which has been
        trained but not yet had the chance to decay.
        """
        return float(self.just_after.mean())

    @property
    def retention_curve(self) -> list[float]:
        """Mean error over the tasks seen so far, after each step."""
        return [float(self.errors[: s + 1, s].mean()) for s in range(len(self.errors))]


@dataclass
class TrainConfig:
    epochs: int = 20
    batch: int = 256
    lr: float = 1e-3
    ewc_lambda: float = 0.0


class ElasticPenalty:
    """Diagonal-Fisher EWC: one curvature estimate and one anchor per completed task.

    The textbook formulation rather than the online variant -- penalty terms accumulate
    per task instead of collapsing into a single running Fisher, which is what `plain EWC`
    means in the rest of this repository. Inert at strength 0, including the graph node,
    so the no-EWC arm reproduces the unregularised numbers exactly.
    """

    def __init__(self, strength: float) -> None:
        self.strength = strength
        self.terms: list[tuple[dict[str, Tensor], dict[str, Tensor]]] = []

    @property
    def active(self) -> bool:
        return self.strength > 0

    @property
    def applies(self) -> bool:
        """False until a task has been consolidated, so the loss expression stays untouched."""
        return bool(self.terms)

    def observe(self, net: ActionHead, x: Tensor, y: Tensor, batch: int) -> None:
        """Record where the parameters landed on the task just finished, and how much each matters.

        The batched empirical Fisher: the squared gradient of each minibatch-mean loss, not the
        mean of per-sample squared gradients. That is the usual EWC implementation but it shrinks
        the estimate by roughly the batch size, which is why the useful strengths here are 1e4 to
        1e8. Strengths are not portable across `batch`, `epochs`, or action dimensionality, so
        `batch` must stay tied to the training batch size.
        """
        if not self.active:
            return
        fisher = {n: torch.zeros_like(p) for n, p in net.named_parameters()}
        net.eval()
        for i in range(0, len(x), batch):
            net.zero_grad()
            F.mse_loss(net(x[i : i + batch]), y[i : i + batch]).backward()
            for n, param in net.named_parameters():
                if param.grad is not None:
                    fisher[n] += param.grad.detach() ** 2
        for n in fisher:
            fisher[n] /= max(1, math.ceil(len(x) / batch))
        self.terms.append((fisher, {n: p.detach().clone() for n, p in net.named_parameters()}))
        net.zero_grad()

    def __call__(self, net: ActionHead) -> Tensor:
        total = torch.zeros((), device=DEVICE)
        params = dict(net.named_parameters())
        for fisher, anchor in self.terms:
            for n, param in params.items():
                total = total + (fisher[n] * (param - anchor[n]) ** 2).sum()
        return 0.5 * self.strength * total


class SequenceRunner:
    """Trains one head over one ordering, evaluating every seen task after every step.

    Splits are built once per representation: nothing mutates them, and rebuilding them
    for all 18 ordering-seed cells would dominate the runtime on the wider backbones.
    """

    def __init__(self, store: FeatureStore, rep: str, cfg: TrainConfig) -> None:
        self.store, self.rep, self.cfg = store, rep, cfg
        norm = store.norms[rep]
        self.splits = {t.index: t.split(rep, norm) for t in store.tasks}

    def run(self, order: list[int], seed: int) -> SequenceResult:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        net = ActionHead(self.store.dim(self.rep), self.store.tasks[0].action_dim).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=self.cfg.lr)

        penalty = ElasticPenalty(self.cfg.ewc_lambda)
        errors = np.zeros((len(order), len(order)))
        for step, task in enumerate(order):
            xt, yt = self.splits[task][:2]
            self.fit(net, opt, xt, yt, penalty)
            penalty.observe(net, xt, yt, self.cfg.batch)
            net.eval()
            with torch.no_grad():
                for pos in range(step + 1):
                    _, _, xe, ye = self.splits[order[pos]]
                    errors[pos, step] = float(F.mse_loss(net(xe), ye))
        return SequenceResult(errors)

    def fit(self, net: ActionHead, opt: torch.optim.Optimizer, x: Tensor, y: Tensor,
            penalty: ElasticPenalty) -> None:
        net.train()
        for _ in range(self.cfg.epochs):
            perm = torch.randperm(len(x), device=DEVICE)
            for i in range(0, len(perm), self.cfg.batch):
                j = perm[i : i + self.cfg.batch]
                loss = F.mse_loss(net(x[j]), y[j])
                if penalty.applies:
                    loss = loss + penalty(net)
                opt.zero_grad()
                loss.backward()
                opt.step()


@dataclass
class ArmSummary:
    """One representation's results across every ordering and seed, shaped [ordering, seed]."""

    rep: str
    final: np.ndarray
    forgetting: np.ndarray
    first_task: np.ndarray
    plasticity: np.ndarray
    curve: np.ndarray

    @staticmethod
    def paired(metric: np.ndarray) -> np.ndarray:
        """Seed-averaged per ordering: seeds are nested, so orderings are the unit of analysis."""
        return metric.mean(axis=1)

    def against(self, reference: np.ndarray, mine: np.ndarray) -> float:
        if reference is mine:
            return float("nan")
        return float(stats.ttest_rel(self.paired(reference), self.paired(mine)).pvalue)

    def row(self, reference: ArmSummary) -> dict[str, float]:
        return {"final_error": float(self.final.mean()), "abs_forgetting": float(self.forgetting.mean()),
                "first_task_error": float(self.first_task.mean()), "plasticity": float(self.plasticity.mean()),
                "final_vs_ref_pct": 100 * (self.final.mean() / reference.final.mean() - 1),
                "p_final": self.against(reference.final, self.final),
                "forget_vs_ref": float(self.forgetting.mean() - reference.forgetting.mean()),
                "p_forget": self.against(reference.forgetting, self.forgetting)}


class ContinualExperiment:
    """Runs every representation through the same orderings and reports paired comparisons."""

    def __init__(self, store: FeatureStore, n_orderings: int, n_seeds: int, cfg: TrainConfig) -> None:
        self.store, self.n_seeds, self.cfg = store, n_seeds, cfg
        rng = random.Random(0)
        self.orderings = [rng.sample(range(len(store.tasks)), len(store.tasks)) for _ in range(n_orderings)]

    def arm(self, rep: str) -> ArmSummary:
        runner = SequenceRunner(self.store, rep, self.cfg)
        grid = [[runner.run(o, s) for s in range(self.n_seeds)] for o in self.orderings]

        def gather(attr: str) -> np.ndarray:
            return np.array([[getattr(r, attr) for r in row] for row in grid])

        return ArmSummary(rep, gather("final_error"), gather("abs_forgetting"),
                          gather("first_task_error"), gather("plasticity"), gather("retention_curve"))

    def run(self) -> list[ArmSummary]:
        arms = []
        for rep in self.store.representations:
            arms.append(self.arm(rep))
            typer.echo(f"{rep:<22} final {arms[-1].final.mean():.4f}  "
                       f"abs-forget {arms[-1].forgetting.mean():+.4f}")
        return arms


class Reporter:
    """Prints the paired table, streams curves to W&B, and writes the results JSON."""

    COLUMNS = ["rep", "final_error", "abs_forgetting", "plasticity", "first_task_error",
               "final_vs_ref_pct", "p_final", "forget_vs_ref", "p_forget"]

    def __init__(self, arms: list[ArmSummary], out: Path) -> None:
        self.arms, self.out = arms, out
        self.reference = next(a for a in arms if a.rep == STATE)
        self.summary = [{"rep": a.rep, **a.row(self.reference)} for a in arms]

    def emit(self) -> None:
        table = wandb.Table(columns=self.COLUMNS)
        for r in self.summary:
            table.add_data(*[r[c] if isinstance(r[c], str) else round(float(r[c]), 5) for c in self.COLUMNS])
            typer.echo(f"== {r['rep']:<22} final {r['final_error']:.4f} "
                       f"({r['final_vs_ref_pct']:+6.1f}% vs {self.reference.rep}, p={r['p_final']:.3f})   "
                       f"abs-forget {r['abs_forgetting']:+.4f} (p={r['p_forget']:.3f})   "
                       f"plasticity-err {r['plasticity']:.4f}")
            for k in ("final_error", "abs_forgetting", "first_task_error", "plasticity"):
                wandb.summary[f"{k}/{r['rep']}"] = r[k]
        wandb.log({"continual/summary": table})
        self.log_curves()
        self.write_json()

    def log_curves(self) -> None:
        """Mean error over tasks seen so far, after each step -- the shape of the forgetting."""
        wandb.define_metric("sequence/step")
        wandb.define_metric("sequence/*", step_metric="sequence/step")
        for step in range(self.arms[0].curve.shape[2]):
            wandb.log({"sequence/step": step + 1,
                       **{f"sequence/{a.rep}": float(a.curve[:, :, step].mean()) for a in self.arms}})

    def write_json(self) -> None:
        payload = {"summary": self.summary,
                   "per_ordering": {a.rep: {"final": ArmSummary.paired(a.final).tolist(),
                                            "abs_forgetting": ArmSummary.paired(a.forgetting).tolist(),
                                            "plasticity": ArmSummary.paired(a.plasticity).tolist()}
                                    for a in self.arms}}
        self.out.write_text(json.dumps(payload, indent=2))
        artifact = wandb.Artifact("continual-forgetting", type="results")
        artifact.add_file(str(self.out))
        wandb.log_artifact(artifact)


def main(n_orderings: int = 6, n_seeds: int = 3, n_tasks: int = 10, eval_demos: int = 2,
         epochs: int = 20, pca_dim: int = 0, ewc_lambda: float = 0.0,
         out: Path = ROOT / "continual_results.json") -> None:
    store = FeatureStore(ROOT, n_tasks, eval_demos, pca_dim)
    tag = f"-pca{pca_dim}" if pca_dim else ""
    tag += f"-ewc{ewc_lambda:g}" if ewc_lambda else ""
    run = wandb.init(project="video-wam", job_type="continual", name=f"continual-forgetting{tag}",
                     config={"reps": store.representations, "n_orderings": n_orderings, "n_seeds": n_seeds,
                             "n_tasks": n_tasks, "eval_demos": eval_demos, "epochs": epochs,
                             "pca_dim": pca_dim, "ewc_lambda": ewc_lambda,
                             "metric": "absolute", "paired_on": "ordering"})
    cfg = TrainConfig(epochs=epochs, ewc_lambda=ewc_lambda)
    arms = ContinualExperiment(store, n_orderings, n_seeds, cfg).run()
    typer.echo("")
    Reporter(arms, out).emit()
    typer.echo(f"\nW&B run: {run.url}")
    run.finish()


if __name__ == "__main__":
    typer.run(main)
