#!/usr/bin/env python3
"""Real 6-backbone x 4-trust-method sweep on ManiSkill, 5 seeds.

Combines three already-validated real components:
  - continualwam.backbones: the canonical 6 world-model backbones (MLP, RSSM,
    JEPA, DreamerV3, Diffusion, Transformer), each with a real train_loss/
    predict_error interface.
  - continualwam.trust: real trust scorers (None, EMA, MultiStep, Ensemble).
    FFDC is excluded -- its verifier network is randomly initialised and
    never trained here, so using it would produce meaningless random scores
    presented as a real trust signal.
  - ManiSkillBenchmark.collect_task_data: real gym.make(...)/env.step(...)
    rollouts (random policy) on real ManiSkill environments.

For each seed, task data for the 3 ManiSkill environments is collected once
and reused across all 6 backbones x 4 trust methods (data collection is
backbone/trust-independent). Each backbone is trained sequentially across
the 3 tasks with trust-weighted EWC consolidation between tasks, using the
same real Fisher-computation pattern already fixed and validated earlier
this session (compute a diagonal empirical Fisher from the completed task's
batches when a task finishes, anchor parameters, penalise drift on later
tasks scaled by that task's mean trust score).
"""
from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continualwam.backbones import BACKBONES, get_backbone
from continualwam.trust import NoTrust, EMATrust, MultiStepTrust, EnsembleTrust
from maniskill_benchmark import ManiSkillBenchmark

RESULTS_DIR = Path(__file__).parent / "results_maniskill_backbone_sweep"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASKS = ["PushCube-v1", "LiftPegUpright-v1", "StackCube-v1"]
TRUST_METHODS = ["none", "ema", "multi_step", "ensemble"]
NUM_SEEDS = 5
EPOCHS_PER_TASK = 10
EWC_LAMBDA = 1000.0


def make_trust_scorer(name: str, obs_dim: int, act_dim: int):
    if name == "none":
        return NoTrust()
    if name == "ema":
        return EMATrust()
    if name == "multi_step":
        return MultiStepTrust()
    if name == "ensemble":
        return EnsembleTrust(obs_dim=obs_dim, act_dim=act_dim).to(DEVICE)
    raise ValueError(name)


class BackboneSweepExperiment:
    """One (backbone, trust_method, seed) run: sequential training across the
    3 ManiSkill tasks with trust-weighted EWC consolidation, real Fisher
    computation between tasks.
    """

    def __init__(self, backbone_name: str, trust_name: str, obs_dim: int, act_dim: int):
        self.backbone_name = backbone_name
        self.trust_name = trust_name
        self.model = get_backbone(backbone_name, obs_dim, act_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.trust_scorer = make_trust_scorer(trust_name, obs_dim, act_dim)
        self.fisher_info: dict[int, dict[str, torch.Tensor]] = {}
        self.optimal_params: dict[int, dict[str, torch.Tensor]] = {}
        self.task_trust: dict[int, float] = {}

    def run_task(self, task_id: int, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> None:
        # train_loss() loops sequentially over the T dimension internally, so
        # a chunk length must stay short (matching the per-episode sequences
        # it's designed for) -- treating hundreds of concatenated transitions
        # as one giant sequence would make every epoch O(n_transitions).
        chunk_len = 16
        n_chunks = max(1, obs.shape[0] // chunk_len)
        trust_scores = []

        for _ in range(EPOCHS_PER_TASK):
            self.model.train()
            perm = torch.randperm(obs.shape[0])
            for c in range(n_chunks):
                idx = perm[c * chunk_len:(c + 1) * chunk_len]
                if len(idx) < 2:
                    continue
                self.optimizer.zero_grad()
                obs_seq, act_seq = obs[idx].unsqueeze(0), act[idx].unsqueeze(0)
                loss = self.model.train_loss(obs_seq, act_seq)

                penalty = torch.tensor(0.0, device=DEVICE)
                for tid, fisher in self.fisher_info.items():
                    trust_weight = self.task_trust.get(tid, 0.5)
                    for name, p in self.model.named_parameters():
                        if name in fisher:
                            optimal = self.optimal_params[tid][name]
                            penalty = penalty + trust_weight * EWC_LAMBDA * (fisher[name] * (p - optimal).pow(2)).sum()
                (loss + penalty).backward()
                self.optimizer.step()

            with torch.no_grad():
                eval_idx = torch.randperm(obs.shape[0])[:64]
                pe = self.model.predict_error(obs[eval_idx], act[eval_idx], next_obs[eval_idx])
            trust = self.trust_scorer.compute_trust(pe, obs=obs[eval_idx], act=act[eval_idx], next_obs=next_obs[eval_idx])
            trust_scores.append(trust.mean().item())

        self.task_trust[task_id] = float(np.mean(trust_scores))
        self._compute_fisher(task_id, obs, act, next_obs)

    def _compute_fisher(self, task_id: int, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> None:
        # predict_error() is @torch.no_grad() in the backbone classes (a real
        # eval-only utility), so Fisher must be estimated from train_loss()
        # instead, using a 2-step pseudo-sequence (same trick TransformerBackbone
        # uses internally): obs_seq = [obs, next_obs], act_seq = [act, 0].
        # Estimated as the squared gradient of the batch-mean loss (one
        # batched backward pass) rather than an average of per-sample squared
        # gradients -- a standard, much cheaper empirical-Fisher approximation.
        self.model.train()
        self.model.zero_grad()
        n_samples = min(50, obs.shape[0])
        idx = torch.randperm(obs.shape[0])[:n_samples]
        obs_seq = torch.stack([obs[idx], next_obs[idx]], dim=1)
        act_seq = torch.stack([act[idx], torch.zeros_like(act[idx])], dim=1)
        loss = self.model.train_loss(obs_seq, act_seq)
        loss.backward()
        fisher = {
            n: (p.grad.data.pow(2) if p.grad is not None else torch.zeros_like(p))
            for n, p in self.model.named_parameters()
        }
        self.fisher_info[task_id] = fisher
        self.optimal_params[task_id] = {n: p.detach().clone() for n, p in self.model.named_parameters()}

    def eval_task(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            return self.model.predict_error(obs, act, next_obs).mean().item()


def collect_seed_data(seed: int) -> list[dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    bench = ManiSkillBenchmark(num_tasks=len(TASKS), episodes_per_task=15, max_steps=60,
                                obs_dim=64, action_dim=10)
    task_tensors = []
    for env_name in TASKS:
        data = bench.collect_task_data(env_name, num_episodes=bench.episodes_per_task)
        obs = torch.tensor(np.asarray(data["observations"]), dtype=torch.float32, device=DEVICE)
        act = torch.tensor(np.asarray(data["actions"]), dtype=torch.float32, device=DEVICE)
        next_obs = torch.tensor(np.asarray(data["next_observations"]), dtype=torch.float32, device=DEVICE)
        task_tensors.append({"obs": obs, "act": act, "next_obs": next_obs})
    return task_tensors


def run_sweep() -> dict:
    obs_dim, act_dim = 64, 10
    results: dict[str, dict[str, list[list[float]]]] = {
        b: {t: [] for t in TRUST_METHODS} for b in BACKBONES
    }
    t0 = time.time()

    for seed in range(NUM_SEEDS):
        task_data = collect_seed_data(seed)
        for backbone_name in BACKBONES:
            for trust_name in TRUST_METHODS:
                torch.manual_seed(seed)
                exp = BackboneSweepExperiment(backbone_name, trust_name, obs_dim, act_dim)
                for task_id, td in enumerate(task_data):
                    exp.run_task(task_id, td["obs"], td["act"], td["next_obs"])
                per_task_errors = [exp.eval_task(td["obs"], td["act"], td["next_obs"]) for td in task_data]
                avg_error = float(np.mean(per_task_errors))
                results[backbone_name][trust_name].append(per_task_errors + [avg_error])
                print(f"seed={seed} {backbone_name:12s} {trust_name:10s} avg_error={avg_error:.4f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    aggregated = {}
    for b in BACKBONES:
        aggregated[b] = {}
        for t in TRUST_METHODS:
            runs = results[b][t]
            avgs = [r[-1] for r in runs]
            aggregated[b][t] = {
                "mean": float(np.mean(avgs)),
                "std": float(np.std(avgs)),
                "seeds": avgs,
            }
    return aggregated


def main() -> None:
    agg = run_sweep()
    with open(RESULTS_DIR / "aggregated.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
