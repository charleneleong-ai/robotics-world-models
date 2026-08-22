"""Unified Benchmark for Continual Learning on Robotics Environments.

Runs all 9 CL methods across KinDER and ManiSkill benchmarks.
Uses goal-direction binary classification task.

Usage:
    python3 unified_benchmark.py --benchmark kinder --phase 1
    python3 unified_benchmark.py --benchmark maniskill --phase 1
    python3 unified_benchmark.py --benchmark all --phase 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================== MODELS ========================

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        return self.net(x)


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


# ======================== DATA COLLECTION ========================

def _flatten(x, target_dim):
    if isinstance(x, dict):
        flat = np.concatenate([v.flatten() for v in x.values() if isinstance(v, np.ndarray)])
    else:
        flat = np.asarray(x, np.float32).flat[:]
    if len(flat) < target_dim:
        flat = np.pad(flat, (0, target_dim - len(flat)))
    return flat[:target_dim].astype(np.float32)


def collect_kinder(env_name, n_episodes=30, max_steps=100, obs_dim=100):
    """Collect transitions from KinDER environment."""
    try:
        import kinder
        kinder.register_all_environments()
        env = kinder.make(env_name)
    except Exception as e:
        print(f"  WARN: {env_name} failed: {e}")
        return None

    obs_list, act_list, next_list, rew_list = [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            obs_flat = _flatten(obs, obs_dim)
            action = env.action_space.sample()
            act_flat = _flatten(action, min(len(np.atleast_1d(action)), 10))

            next_obs, reward, term, trunc, _ = env.step(action)
            next_flat = _flatten(next_obs, obs_dim)

            obs_list.append(obs_flat)
            act_list.append(act_flat)
            next_list.append(next_flat)
            rew_list.append(float(reward))

            obs = next_obs
            if term or trunc:
                break
    env.close()

    obs_arr = np.array(obs_list, np.float32)
    next_arr = np.array(next_list, np.float32)
    
    # Dynamics-based labels: 1 = moving (large obs change), 0 = stationary
    obs_change = np.linalg.norm(next_arr - obs_arr, axis=1)
    threshold = np.median(obs_change)  # Median split for balanced classes
    labels = (obs_change > threshold).astype(np.int64)

    print(f"  {env_name}: {len(obs_arr)} samples, label dist: {np.bincount(labels, minlength=2).tolist()}")

    return {
        "obs": torch.tensor(obs_arr, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collect_maniskill(env_name, n_episodes=30, max_steps=100, obs_dim=100):
    """Collect transitions from ManiSkill environment."""
    try:
        import gymnasium as gym
        import mani_skill.envs
        env = gym.make(env_name, render_mode=None)
    except Exception as e:
        print(f"  WARN: {env_name} failed: {e}")
        return None

    obs_list, act_list, next_list, rew_list = [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            if isinstance(obs, dict):
                obs_flat = _flatten(obs, obs_dim)
            else:
                obs_flat = _flatten(obs, obs_dim)

            action = env.action_space.sample()
            act_flat = _flatten(action, min(len(np.atleast_1d(action)), 10))

            next_obs, reward, term, trunc, _ = env.step(action)
            next_flat = _flatten(next_obs, obs_dim)

            obs_list.append(obs_flat)
            act_list.append(act_flat)
            next_list.append(next_flat)
            rew_list.append(float(reward))

            obs = next_obs
            if term or trunc:
                break
    env.close()

    obs_arr = np.array(obs_list, np.float32)
    next_arr = np.array(next_list, np.float32)
    
    # Dynamics-based labels: 1 = moving (large obs change), 0 = stationary
    obs_change = np.linalg.norm(next_arr - obs_arr, axis=1)
    threshold = np.median(obs_change)  # Median split for balanced classes
    labels = (obs_change > threshold).astype(np.int64)

    print(f"  {env_name}: {len(obs_arr)} samples, label dist: {np.bincount(labels, minlength=2).tolist()}")

    return {
        "obs": torch.tensor(obs_arr, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ======================== CL METHODS ========================

class FineTuningCL:
    name = "Fine-tuning"
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def observe(self, batch):
        self.model.train()
        self.opt.zero_grad()
        obs, lbl = batch["obs"].to(self.device), batch["labels"].to(self.device)
        logits = self.model(obs)
        loss = F.cross_entropy(logits, lbl)
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits.argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid): pass


class EWCCL:
    name = "EWC"
    def __init__(self, model, device, lam=5000.0):
        self.model, self.device, self.lam = model, device, lam
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.fisher, self.star = {}, {}

    def observe(self, batch):
        self.model.train()
        self.opt.zero_grad()
        obs, lbl = batch["obs"].to(self.device), batch["labels"].to(self.device)
        logits = self.model(obs)
        loss = F.cross_entropy(logits, lbl)
        if self.fisher:
            pen = torch.tensor(0.0, device=self.device)
            for tid, f in self.fisher.items():
                for n, p in self.model.named_parameters():
                    if n in f:
                        pen += (f[n] * (p - self.star[tid][n]).pow(2)).sum()
            loss += self.lam * pen
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits.argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid):
        in_dim = self.model.net[0].in_features
        self.model.zero_grad()
        d = torch.randn(128, in_dim, device=self.device)
        loss = F.cross_entropy(self.model(d), torch.randint(0, 2, (128,), device=self.device))
        loss.backward()
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                fisher[n] = p.grad.data.pow(2).clone()
        self.fisher[tid] = fisher
        self.star[tid] = {n: p.data.clone() for n, p in self.model.named_parameters()}


class LwFCL:
    name = "LwF"
    def __init__(self, model, device, alpha=0.5, T=2.0):
        self.model, self.device = model, device
        self.alpha, self.T = alpha, T
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.prev = None

    def observe(self, batch):
        self.model.train()
        self.opt.zero_grad()
        obs, lbl = batch["obs"].to(self.device), batch["labels"].to(self.device)
        logits = self.model(obs)
        loss_new = F.cross_entropy(logits, lbl)
        if self.prev is not None:
            self.prev.eval()
            with torch.no_grad():
                prev_logits = self.prev(obs)
            loss_kd = F.kl_div(
                F.log_softmax(logits / self.T, -1),
                F.softmax(prev_logits / self.T, -1),
                reduction="batchmean",
            ) * (self.T ** 2)
            loss = (1 - self.alpha) * loss_new + self.alpha * loss_kd
        else:
            loss = loss_new
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits.argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid):
        import copy
        self.prev = copy.deepcopy(self.model)


class PackNetCL:
    name = "PackNet"
    def __init__(self, model, device, freeze_pct=0.3):
        self.model, self.device = model, device
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.freeze_pct = freeze_pct
        self.mask = {}

    def observe(self, batch):
        self.model.train()
        self.opt.zero_grad()
        obs, lbl = batch["obs"].to(self.device), batch["labels"].to(self.device)
        logits = self.model(obs)
        loss = F.cross_entropy(logits, lbl)
        loss.backward()
        for n, p in self.model.named_parameters():
            if n in self.mask and p.grad is not None:
                p.grad *= (1 - self.mask[n])
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits.argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid):
        import copy
        new_mask = {}
        for n, p in self.model.named_parameters():
            prev = self.mask.get(n, torch.zeros_like(p))
            mag = p.data.abs().clone()
            mag[prev.bool()] = float("inf")
            flat = mag.flatten()
            k = max(1, int(len(flat) * self.freeze_pct))
            thresh = flat.topk(k, largest=False).values[-1]
            freeze = (mag <= thresh).float()
            new_mask[n] = torch.clamp(prev + freeze, 0, 1)
        self.mask = new_mask


class ExperienceReplayCL:
    name = "Experience Replay"
    def __init__(self, model, device, buf_size=500):
        self.model, self.device = model, device
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.buf = deque(maxlen=buf_size)

    def observe(self, batch):
        self.model.train()
        self.opt.zero_grad()
        obs, lbl = batch["obs"].to(self.device), batch["labels"].to(self.device)
        for i in range(len(obs)):
            self.buf.append({"obs": obs[i].cpu(), "lbl": lbl[i].cpu()})
        if len(self.buf) > 128:
            s = list(self.buf)
            idx = np.random.choice(len(s), min(len(obs) * 2, len(s)), replace=False)
            ro = torch.stack([s[i]["obs"] for i in idx]).to(self.device)
            rl = torch.stack([s[i]["lbl"] for i in idx]).to(self.device)
            all_obs, all_lbl = torch.cat([obs, ro]), torch.cat([lbl, rl])
        else:
            all_obs, all_lbl = obs, lbl
        logits = self.model(all_obs)
        loss = F.cross_entropy(logits, all_lbl)
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits[:len(obs)].argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid): pass


class PrioritizedReplayCL:
    name = "Prioritized Replay"
    def __init__(self, model, device, buf_size=500, alpha=0.6):
        self.model, self.device, self.alpha = model, device, alpha
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.buf = deque(maxlen=buf_size)
        self.pris = deque(maxlen=buf_size)

    def observe(self, batch):
        self.model.train()
        self.opt.zero_grad()
        obs, lbl = batch["obs"].to(self.device), batch["labels"].to(self.device)
        with torch.no_grad():
            sl = F.cross_entropy(self.model(obs), lbl, reduction="none")
        for i in range(len(obs)):
            self.buf.append({"obs": obs[i].cpu(), "lbl": lbl[i].cpu()})
            self.pris.append(sl[i].item() + 1e-6)
        if len(self.buf) > 128:
            s = list(self.buf)
            p = np.array(list(self.pris)) ** self.alpha
            p /= p.sum()
            idx = np.random.choice(len(s), min(len(obs) * 2, len(s)), p=p, replace=True)
            ro = torch.stack([s[i]["obs"] for i in idx]).to(self.device)
            rl = torch.stack([s[i]["lbl"] for i in idx]).to(self.device)
            all_obs, all_lbl = torch.cat([obs, ro]), torch.cat([lbl, rl])
        else:
            all_obs, all_lbl = obs, lbl
        logits = self.model(all_obs)
        loss = F.cross_entropy(logits, all_lbl)
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits[:len(obs)].argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid): pass


class CuriousReplayCL:
    name = "Curious Replay"
    def __init__(self, model, world_model, device, buf_size=500, alpha=0.6):
        self.model, self.wm, self.device, self.alpha = model, world_model, device, alpha
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.buf = deque(maxlen=buf_size)
        self.pris = deque(maxlen=buf_size)

    def observe(self, batch):
        self.model.train()
        self.opt.zero_grad()
        obs = batch["obs"].to(self.device)
        lbl = batch["labels"].to(self.device)
        act_dim = self.wm.net[0].in_features - obs.shape[1]
        act = torch.randn(len(obs), act_dim, device=self.device)

        with torch.no_grad():
            err = F.mse_loss(self.wm(obs, act), obs, reduction="none").mean(dim=-1)

        for i in range(len(obs)):
            self.buf.append({"obs": obs[i].cpu(), "lbl": lbl[i].cpu()})
            self.pris.append(err[i].item() + 1e-6)

        if len(self.buf) > 128:
            s = list(self.buf)
            p = np.array(list(self.pris)) ** self.alpha
            p /= p.sum()
            idx = np.random.choice(len(s), min(len(obs) * 2, len(s)), p=p, replace=True)
            ro = torch.stack([s[i]["obs"] for i in idx]).to(self.device)
            rl = torch.stack([s[i]["lbl"] for i in idx]).to(self.device)
            all_obs, all_lbl = torch.cat([obs, ro]), torch.cat([lbl, rl])
        else:
            all_obs, all_lbl = obs, lbl

        logits = self.model(all_obs)
        loss = F.cross_entropy(logits, all_lbl)
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits[:len(obs)].argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid): pass


class WMTrustCL:
    name = "WM Trust CL"
    def __init__(self, model, wm, device, lam=5000.0):
        self.model, self.wm, self.device, self.lam = model, wm, device, lam
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.wm_opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        self.fisher, self.star, self.trust_map = {}, {}, {}

    def observe(self, batch):
        self.model.train()
        self.wm.train()
        obs = batch["obs"].to(self.device)
        lbl = batch["labels"].to(self.device)
        act_dim = self.wm.net[0].in_features - obs.shape[1]
        act = torch.randn(len(obs), act_dim, device=self.device)

        self.wm_opt.zero_grad()
        wm_loss = F.mse_loss(self.wm(obs, act), obs)
        wm_loss.backward()
        self.wm_opt.step()

        self.opt.zero_grad()
        logits = self.model(obs)
        ce = F.cross_entropy(logits, lbl, reduction="none")
        with torch.no_grad():
            err = F.mse_loss(self.wm(obs, act), obs, reduction="none").mean(dim=-1)
            trust = torch.exp(-err)
        loss = (trust * ce).mean()

        if self.fisher:
            pen = torch.tensor(0.0, device=self.device)
            for tid, f in self.fisher.items():
                t = self.trust_map.get(tid, 0.5)
                for n, p in self.model.named_parameters():
                    if n in f:
                        pen += self.lam * t * (f[n] * (p - self.star[tid][n]).pow(2)).sum()
            loss += pen
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits.argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid):
        in_dim = self.model.net[0].in_features
        self.model.zero_grad()
        d = torch.randn(128, in_dim, device=self.device)
        loss = F.cross_entropy(self.model(d), torch.randint(0, 2, (128,), device=self.device))
        loss.backward()
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                fisher[n] = p.grad.data.pow(2).clone()
        self.fisher[tid] = fisher
        self.star[tid] = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.trust_map[tid] = 0.5


class ContinualWAM:
    name = "ContinualWAM (ours)"
    def __init__(self, model, wm, device, buf_size=500, lam=5000.0):
        self.model, self.wm, self.device = model, wm, device
        self.buf = deque(maxlen=buf_size)
        self.lam = lam
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.wm_opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        self.fisher, self.star, self.trust_map = {}, {}, {}
        self.ema_trust = 0.5

    def observe(self, batch):
        self.model.train()
        self.wm.train()
        obs = batch["obs"].to(self.device)
        lbl = batch["labels"].to(self.device)
        act_dim = self.wm.net[0].in_features - obs.shape[1]
        act = torch.randn(len(obs), act_dim, device=self.device)

        self.wm_opt.zero_grad()
        wm_loss = F.mse_loss(self.wm(obs, act), obs)
        wm_loss.backward()
        self.wm_opt.step()

        self.opt.zero_grad()
        with torch.no_grad():
            err = F.mse_loss(self.wm(obs, act), obs, reduction="none").mean(dim=-1)
            trust = torch.exp(-err)
            self.ema_trust = 0.9 * self.ema_trust + 0.1 * trust.mean().item()

        for i in range(len(obs)):
            self.buf.append({"obs": obs[i].cpu(), "lbl": lbl[i].cpu(), "trust": trust[i].item()})

        if len(self.buf) > 128:
            s = list(self.buf)
            t = np.clip(np.array([x["trust"] for x in s]), 1e-6, None)
            p = t / t.sum()
            idx = np.random.choice(len(s), min(len(obs) * 2, len(s)), p=p, replace=True)
            ro = torch.stack([s[i]["obs"] for i in idx]).to(self.device)
            rl = torch.stack([s[i]["lbl"] for i in idx]).to(self.device)
            all_obs, all_lbl = torch.cat([obs, ro]), torch.cat([lbl, rl])
        else:
            all_obs, all_lbl = obs, lbl

        logits = self.model(all_obs)
        loss = F.cross_entropy(logits, all_lbl)

        if self.fisher:
            pen = torch.tensor(0.0, device=self.device)
            for tid, f in self.fisher.items():
                t = self.trust_map.get(tid, 0.5)
                adaptive_lam = self.lam * (1 - t)
                for n, p in self.model.named_parameters():
                    if n in f:
                        pen += adaptive_lam * (f[n] * (p - self.star[tid][n]).pow(2)).sum()
            loss += pen
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "acc": (logits[:len(obs)].argmax(-1) == lbl).float().mean().item()}

    def on_task_done(self, tid):
        in_dim = self.model.net[0].in_features
        self.model.zero_grad()
        d = torch.randn(128, in_dim, device=self.device)
        loss = F.cross_entropy(self.model(d), torch.randint(0, 2, (128,), device=self.device))
        loss.backward()
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                fisher[n] = p.grad.data.pow(2).clone()
        self.fisher[tid] = fisher
        self.star[tid] = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.trust_map[tid] = self.ema_trust


# ======================== BENCHMARK RUNNER ========================

def run_benchmark(
    benchmark_name: str,
    env_names: list[str],
    obs_dim: int = 100,
    n_episodes: int = 30,
    max_steps: int = 100,
    epochs_per_task: int = 15,
    batch_size: int = 64,
    output_dir: str = "benchmark_results",
):
    """Run all CL methods on a benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {benchmark_name.upper()}")
    print(f"Device: {device}, Environments: {len(env_names)}")
    print(f"{'='*80}\n")

    # Collect data
    print("Collecting data...")
    tasks = []
    collector = collect_kinder if benchmark_name == "kinder" else collect_maniskill

    for env_name in env_names:
        data = collector(env_name, n_episodes, max_steps, obs_dim)
        if data is not None:
            tasks.append(data)

    if not tasks:
        print("ERROR: No tasks collected!")
        return None

    print(f"\nCollected {len(tasks)} tasks")

    # Create methods
    obs_dim_actual = tasks[0]["obs"].shape[1]

    def mk_model():
        return MLP(obs_dim_actual, 256, 2).to(device)
    def mk_wm():
        return WorldModel(obs_dim_actual, 10, 256).to(device)

    methods = [
        FineTuningCL(mk_model(), device),
        EWCCL(mk_model(), device, lam=5000),
        LwFCL(mk_model(), device),
        PackNetCL(mk_model(), device, freeze_pct=0.3),
        ExperienceReplayCL(mk_model(), device, buf_size=500),
        PrioritizedReplayCL(mk_model(), device, buf_size=500),
        CuriousReplayCL(mk_model(), mk_wm(), device, buf_size=500),
        WMTrustCL(mk_model(), mk_wm(), device, lam=5000),
        ContinualWAM(mk_model(), mk_wm(), device, buf_size=500, lam=5000),
    ]

    # Run each method
    all_results = {}
    for learner in methods:
        print(f"\n{'='*60}")
        print(f"{learner.name}")
        print(f"{'='*60}")

        task_accs = []
        prev_accs_per_task = []

        for tid in range(len(tasks)):
            ds = tasks[tid]

            # Train
            indices = torch.randperm(len(ds["obs"]))
            for ep in range(epochs_per_task):
                for start in range(0, len(indices), batch_size):
                    bi = indices[start:start + batch_size]
                    batch = {
                        "obs": ds["obs"][bi],
                        "labels": ds["labels"][bi],
                    }
                    learner.observe(batch)

            # Evaluate on current task
            learner.model.eval()
            with torch.no_grad():
                logits = learner.model(ds["obs"].to(device))
                acc = (logits.argmax(-1) == ds["labels"].to(device)).float().mean().item()
            task_accs.append(acc)

            # Evaluate on all previous tasks
            prev_accs = []
            for ptid in range(tid):
                prev_ds = tasks[ptid]
                with torch.no_grad():
                    logits = learner.model(prev_ds["obs"].to(device))
                    prev_acc = (logits.argmax(-1) == prev_ds["labels"].to(device)).float().mean().item()
                prev_accs.append(prev_acc)
            prev_accs_per_task.append(prev_accs)

            prev_str = " ".join(f"T{p}={a:.3f}" for p, a in enumerate(prev_accs)) if prev_accs else "---"
            print(f"  T{tid}: train={acc:.4f}  prev=[{prev_str}]")

            learner.on_task_done(tid)

        # Compute metrics
        avg_acc = np.mean(task_accs)
        bwt = 0
        if len(task_accs) > 1:
            bwt = sum(task_accs[i] - task_accs[0] for i in range(1, len(task_accs))) / (len(task_accs) - 1)

        # Forward transfer: how well does T0 help T1, T1 help T2, etc.
        fwt = 0
        if len(task_accs) > 1:
            fwt_values = []
            for i in range(1, len(task_accs)):
                if prev_accs_per_task[i]:
                    fwt_values.append(np.mean(prev_accs_per_task[i]))
            if fwt_values:
                fwt = np.mean(fwt_values)

        all_results[learner.name] = {
            "task_accuracies": [float(a) for a in task_accs],
            "avg_accuracy": float(avg_acc),
            "bwt": float(bwt),
            "fwt": float(fwt),
        }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{benchmark_name}_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{benchmark_name.upper()} RESULTS")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'AvgAcc':>8} {'BWT':>8} {'FWT':>8}  Tasks")
    print("-" * 80)
    for name, res in all_results.items():
        tacc = " ".join(f"{a:.3f}" for a in res["task_accuracies"])
        print(f"{name:<30} {res['avg_accuracy']:>8.4f} {res['bwt']:>8.4f} {res['fwt']:>8.4f}  [{tacc}]")

    print(f"\nResults saved to {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["kinder", "maniskill", "all"], default="all")
    parser.add_argument("--phase", type=int, default=1, help="1=subset, 2=full")
    parser.add_argument("--obs-dim", type=int, default=100)
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    # Phase 1: Subset environments
    kinder_phase1 = [
        "kinder/Motion2D-p0-v0",
        "kinder/StickButton2D-b1-v0",
        "kinder/Obstruction2D-o1-v0",
        "kinder/KinematicShelf3D-o1-v0",
        "kinder/Obstruction3D-o0-v0",
        "kinder/Packing3D-p1-v0",
    ]

    maniskill_phase1 = [
        "PickCube-v1",
        "PushCube-v1",
        "LiftPegUpright-v1",
        "PlugCharger-v1",
    ]

    # Phase 2: Full environment sets
    kinder_phase2 = [
        # 2D environments
        "kinder/Motion2D-p0-v0",
        "kinder/Motion2D-p1-v0",
        "kinder/Motion2D-p2-v0",
        "kinder/Motion2D-p3-v0",
        "kinder/Motion2D-p4-v0",
        "kinder/Motion2D-p5-v0",
        "kinder/StickButton2D-b1-v0",
        "kinder/StickButton2D-b2-v0",
        "kinder/StickButton2D-b3-v0",
        "kinder/StickButton2D-b5-v0",
        "kinder/StickButton2D-b10-v0",
        "kinder/Obstruction2D-o0-v0",
        "kinder/Obstruction2D-o1-v0",
        "kinder/Obstruction2D-o2-v0",
        "kinder/Obstruction2D-o3-v0",
        "kinder/Obstruction2D-o4-v0",
        "kinder/ClutteredStorage2D-b1-v0",
        "kinder/ClutteredStorage2D-b3-v0",
        "kinder/ClutteredStorage2D-b7-v0",
        "kinder/ClutteredStorage2D-b15-v0",
        "kinder/ClutteredRetrieval2D-o1-v0",
        "kinder/ClutteredRetrieval2D-o10-v0",
        "kinder/ClutteredRetrieval2D-o25-v0",
        "kinder/DynObstruction2D-o0-v0",
        "kinder/DynObstruction2D-o1-v0",
        "kinder/DynObstruction2D-o2-v0",
        "kinder/DynObstruction2D-o3-v0",
        "kinder/DynPushPullHook2D-o0-v0",
        "kinder/DynPushPullHook2D-o1-v0",
        "kinder/DynPushPullHook2D-o5-v0",
        "kinder/DynPushT2D-t1-v0",
        "kinder/DynScoopPour2D-o10-v0",
        "kinder/DynScoopPour2D-o20-v0",
        "kinder/DynScoopPour2D-o30-v0",
        "kinder/DynScoopPour2D-o50-v0",
        "kinder/PushPullHook2D-v0",
        # 3D environments
        "kinder/BaseMotion3D-v0",
        "kinder/KinematicShelf3D-o1-v0",
        "kinder/KinematicShelf3D-o2-v0",
        "kinder/KinematicShelf3D-o3-v0",
        "kinder/KinematicShelf3D-o5-v0",
        "kinder/KinematicShelf3D-o10-v0",
        "kinder/Obstruction3D-o0-v0",
        "kinder/Obstruction3D-o1-v0",
        "kinder/Obstruction3D-o2-v0",
        "kinder/Obstruction3D-o3-v0",
        "kinder/Obstruction3D-o4-v0",
        "kinder/Packing3D-p1-v0",
        "kinder/Packing3D-p2-v0",
        "kinder/Packing3D-p3-v0",
        "kinder/PrplLab3D-o1-v0",
        "kinder/PrplLab3D-o2-v0",
        "kinder/Table3D-o1-v0",
        "kinder/Table3D-o2-v0",
        "kinder/Table3D-o3-v0",
        "kinder/Transport3D-o1-v0",
        "kinder/Transport3D-o2-v0",
    ]

    maniskill_phase2 = [
        "PickCube-v1",
        "PushCube-v1",
        "LiftPegUpright-v1",
        "PlugCharger-v1",
        "StackCube-v1",
        "PokeCube-v1",
        "PullCube-v1",
        "PullCubeTool-v1",
        "PlaceSphere-v1",
        "RollBall-v1",
        "PushT-v1",
        "MS-CartpoleBalance-v1",
        "MS-HopperHop-v1",
    ]

    kinder_envs = kinder_phase2 if args.phase == 2 else kinder_phase1
    maniskill_envs = maniskill_phase2 if args.phase == 2 else maniskill_phase1

    print(f"Phase {args.phase}: {'Full' if args.phase == 2 else 'Subset'} benchmark")

    if args.benchmark == "kinder":
        run_benchmark("kinder", kinder_envs, args.obs_dim, args.n_episodes, args.max_steps, args.epochs)
    elif args.benchmark == "maniskill":
        run_benchmark("maniskill", maniskill_envs, args.obs_dim, args.n_episodes, args.max_steps, args.epochs)
    else:
        run_benchmark("kinder", kinder_envs, args.obs_dim, args.n_episodes, args.max_steps, args.epochs)
        run_benchmark("maniskill", maniskill_envs, args.obs_dim, args.n_episodes, args.max_steps, args.epochs)


if __name__ == "__main__":
    main()
