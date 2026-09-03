"""Shared training functions."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

from continualwam.trust import TrustScorer


def load_demos(
    suite_dir: str, n_tasks: int = 10, max_demos: int = 5
) -> list[list[dict[str, np.ndarray]]]:
    """Load expert demonstrations from HDF5 files."""
    all_demos = []
    files = sorted(f for f in __import__("os").listdir(suite_dir) if f.endswith(".hdf5"))
    for fn in files[:n_tasks]:
        with h5py.File(__import__("os").path.join(suite_dir, fn), "r") as hf:
            task_demos = []
            count = 0
            for k in sorted(hf["data"].keys()):
                if not k.startswith("demo_") or count >= max_demos:
                    continue
                demo = hf["data"][k]
                parts = [np.array(demo["obs"][f]) for f in
                         ["ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"]]
                task_demos.append({
                    "obs": np.concatenate(parts, axis=-1),
                    "acts": np.array(demo["actions"]),
                })
                count += 1
            all_demos.append(task_demos)
    return all_demos


def train_wm(
    wm: nn.Module,
    demos: list[dict[str, np.ndarray]],
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    """Train world model on demonstrations."""
    opt = torch.optim.Adam(wm.parameters(), lr=lr)
    wm.train()
    for _ in range(epochs):
        for demo in demos:
            o = torch.tensor(demo["obs"], dtype=torch.float32).unsqueeze(0).to(device)
            a = torch.tensor(demo["acts"], dtype=torch.float32).unsqueeze(0).to(device)
            loss = wm.train_loss(o, a)
            opt.zero_grad()
            loss.backward()
            opt.step()


def train_bc(
    policy: nn.Module,
    demos: list[dict[str, np.ndarray]],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    """Train behavioral cloning policy."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    all_obs = np.concatenate([d["obs"] for d in demos])
    all_acts = np.concatenate([d["acts"] for d in demos])
    obs_t = torch.tensor(all_obs, dtype=torch.float32).to(device)
    act_t = torch.tensor(all_acts, dtype=torch.float32).to(device)
    policy.train()
    for _ in range(epochs):
        perm = torch.randperm(obs_t.size(0))
        for i in range(0, obs_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            loss = F.mse_loss(policy(obs_t[idx]), act_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


def train_bc_trust(
    policy: nn.Module,
    wm: nn.Module,
    trust_scorer: TrustScorer,
    demos: list[dict[str, np.ndarray]],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    """Train BC policy with trust-weighted loss."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    all_obs = np.concatenate([d["obs"] for d in demos])
    all_acts = np.concatenate([d["acts"] for d in demos])
    obs_t = torch.tensor(all_obs, dtype=torch.float32).to(device)
    act_t = torch.tensor(all_acts, dtype=torch.float32).to(device)
    wm.eval()
    policy.train()
    for _ in range(epochs):
        perm = torch.randperm(obs_t.size(0))
        for i in range(0, obs_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 2:
                continue
            pred = policy(obs_t[idx])
            with torch.no_grad():
                pe = wm.predict_error(obs_t[idx], act_t[idx], obs_t[idx])
                trust = trust_scorer.compute_trust(pe, obs=obs_t[idx], act=act_t[idx])
                w = trust.clamp(0.1, 1.0)
            loss = (F.mse_loss(pred, act_t[idx], reduction="none").mean(dim=-1) * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
