"""Shared pieces of the video-backbone audit: LIBERO frames in, cached features out, probes on top.

Kept free of wandb, diffusers and transformers so every environment the extractors run under
(~/wan_venv, the system python for OpenVLA, the repo venv for tests) can import it.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

LIBERO_ROOT = Path("/home/ubuntu/robotics_world_models/LIBERO")
CACHE_ROOT = Path("/home/ubuntu/wan_latents")
STATE_KEYS = ("ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Demo:
    index: int
    rgb: np.ndarray
    action: np.ndarray
    state: np.ndarray


class LiberoSuite:
    """The first `n_tasks` tasks of a suite, each read as its first `max_demos` demonstrations.

    Demonstrations are taken in sorted key order (demo_0, demo_1, demo_10, ...), which is the
    order every cache was written in, so `Demo.index` is what the held-out split keys on.
    """

    def __init__(self, suite: str, camera: str = "agentview_rgb", n_tasks: int = 10,
                 max_demos: int = 12, root: Path = LIBERO_ROOT) -> None:
        self.suite, self.camera, self.n_tasks, self.max_demos = suite, camera, n_tasks, max_demos
        self.dir = root / f"libero_{suite}"

    @property
    def task_files(self) -> list[Path]:
        return sorted(p for p in self.dir.iterdir() if p.suffix == ".hdf5")[: self.n_tasks]

    def demos(self, path: Path) -> Iterator[Demo]:
        """Each demonstration trimmed to the frames that have an image, an action and a state."""
        with h5py.File(path, "r") as h:
            keys = [k for k in sorted(h["data"].keys()) if k.startswith("demo_")][: self.max_demos]
            for index, key in enumerate(keys):
                demo = h["data"][key]
                rgb = np.array(demo[f"obs/{self.camera}"])
                action = np.array(demo["actions"], dtype=np.float32)
                state = np.concatenate([np.array(demo[f"obs/{k}"]) for k in STATE_KEYS], -1).astype(np.float32)
                n = min(len(rgb), len(action), len(state))
                yield Demo(index, rgb[:n], action[:n], state[:n])


def extract_suite(suite: LiberoSuite, encode: Callable[[np.ndarray], dict[str, np.ndarray]],
                  out_dirs: dict[str, Path], label: str,
                  log: Callable[[dict[str, float]], None]) -> int:
    """Encode every frame and write one `taskNN.npz` per task into each named cache.

    `encode` maps a demonstration's (T,H,W,3) frames to one (T,D) feature array per cache name,
    so a single forward pass can feed several caches. Returns the number of frames written.
    """
    for out in out_dirs.values():
        out.mkdir(parents=True, exist_ok=True)
    total = 0
    for ti, path in enumerate(suite.task_files):
        t0 = time.time()
        latents: dict[str, list[np.ndarray]] = {name: [] for name in out_dirs}
        actions, states, demo_ids = [], [], []
        for demo in suite.demos(path):
            for name, latent in encode(demo.rgb).items():
                latents[name].append(latent)
            actions.append(demo.action)
            states.append(demo.state)
            demo_ids.append(np.full(len(demo.action), demo.index, np.int16))
        common = {"action": np.concatenate(actions), "state": np.concatenate(states), "demo": np.concatenate(demo_ids)}
        for name, out in out_dirs.items():
            np.savez_compressed(out / f"task{ti:02d}.npz", latent=np.concatenate(latents[name]), **common)
        frames, seconds = len(common["action"]), time.time() - t0
        total += frames
        print(f"[{label}] task {ti}: {frames} frames, {seconds:.1f}s", flush=True)
        log({"task": ti, "frames": frames, "seconds": seconds})
    return total


def lagged_frames(n: int, lags: Sequence[int]) -> np.ndarray:
    """(n, len(lags)) frame indices: row t holds t - lag for each lag, clamped at the first frame."""
    return np.clip(np.arange(n)[:, None] - np.asarray(lags)[None, :], 0, None)


def held_out_mask(demo: np.ndarray, n_eval: int) -> np.ndarray:
    """True on every frame of the last `n_eval` demonstrations, so evaluation is whole trajectories."""
    held = np.unique(demo)[-n_eval:]
    return np.isin(demo, held)


@dataclass
class TaskArrays:
    features: dict[str, np.ndarray]
    action: np.ndarray
    state: np.ndarray
    demo: np.ndarray


def load_task(sources: dict[str, Path], index: int) -> TaskArrays:
    """One task from several caches, which must describe the same frames in the same order."""
    features: dict[str, np.ndarray] = {}
    first: np.lib.npyio.NpzFile | None = None
    for name, path in sources.items():
        z = np.load(path / f"task{index:02d}.npz")
        features[name] = z["latent"].astype(np.float32)
        if len(features[name]) != len(z["demo"]):
            raise ValueError(f"{name} task{index} has {len(features[name])} latents for {len(z['demo'])} frames")
        if first is None:
            first = z
        elif not (np.array_equal(first["demo"], z["demo"]) and first["action"].shape == z["action"].shape):
            raise ValueError(f"{name} task{index} is not aligned with {next(iter(sources))}")
    return TaskArrays(features, first["action"].astype(np.float32), first["state"].astype(np.float32), first["demo"])


class ActionHead(nn.Module):
    """Small MLP mapping a frozen representation to an action, the only thing that learns."""

    def __init__(self, in_dim: int, out_dim: int, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def fit_probe(xtr: np.ndarray, ytr: np.ndarray, xev: np.ndarray, yev: np.ndarray, seed: int,
              epochs: int = 30, batch: int = 256, lr: float = 1e-3) -> float:
    """Held-out MSE of an action head trained on z-scored training features (stats from training only)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    xt, yt = torch.tensor((xtr - mu) / sd, device=DEVICE), torch.tensor(ytr, device=DEVICE)
    xe, ye = torch.tensor((xev - mu) / sd, device=DEVICE), torch.tensor(yev, device=DEVICE)
    net = ActionHead(xt.shape[1], yt.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=DEVICE)
        for i in range(0, len(perm), batch):
            j = perm[i : i + batch]
            loss = F.mse_loss(net(xt[j]), yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return float(F.mse_loss(net(xe), ye))
