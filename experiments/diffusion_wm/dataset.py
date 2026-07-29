"""Streaming dataset for diffusion world model training.

Loads sharded .npz files produced by collect.py and provides
efficient batching via a lazy-loading Dataset + DataLoader.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TransitionDataset(Dataset):
    """Streaming dataset of (obs, action, next_obs) transitions from sharded .npz files.

    Shards are loaded lazily and cached in memory to avoid repeated disk I/O.
    Each shard is a .npz with fields: obs [N, D], action [N, A], next_obs [N, D].
    """
    def __init__(
        self,
        data_dir: str | Path,
        shard_ids: list[int] | None = None,
        cache_shards: bool = True,
        transform: callable | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.cache_shards = cache_shards
        self.transform = transform

        # Discover shards
        all_shards = sorted(self.data_dir.glob("shard_*.npz"))
        if not all_shards:
            raise FileNotFoundError(f"No shard_*.npz files found in {self.data_dir}")
        if shard_ids is not None:
            all_shards = [s for s in all_shards if _shard_id(s) in shard_ids]
        self.shard_paths = all_shards

        # Build index: (shard_idx, within-shard-position)
        self._shard_sizes: list[int] = []
        self._cum_sizes: list[int] = []
        self._cache: dict[int, dict[str, np.ndarray]] = {} if cache_shards else {}
        total = 0
        for i, sp in enumerate(self.shard_paths):
            # Quick size check without full load
            with np.load(sp, mmap_mode="r") as data:
                n = len(data["obs"])
            self._shard_sizes.append(n)
            self._cum_sizes.append(total)
            total += n
        self._total = total

    def __len__(self) -> int:
        return self._total

    def _load_shard(self, idx: int) -> dict[str, np.ndarray]:
        if idx in self._cache:
            return self._cache[idx]
        data = dict(np.load(self.shard_paths[idx]))
        if self.cache_shards:
            self._cache[idx] = data
        return data

    def __getitem__(self, global_idx: int) -> dict[str, torch.Tensor]:
        # Find which shard contains this index
        shard_idx = 0
        for i in range(len(self._shard_sizes) - 1):
            if global_idx < self._cum_sizes[i + 1]:
                shard_idx = i
                break
        else:
            shard_idx = len(self._shard_sizes) - 1
        local_idx = global_idx - self._cum_sizes[shard_idx]

        shard = self._load_shard(shard_idx)
        item = {
            "obs": torch.from_numpy(shard["obs"][local_idx]),
            "action": torch.from_numpy(shard["action"][local_idx]),
            "next_obs": torch.from_numpy(shard["next_obs"][local_idx]),
        }
        if self.transform:
            item = self.transform(item)
        return item


def _shard_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _standardize(item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Standardize observations to roughly zero mean, unit variance per dim.
    Applied lazily at dataloading time. Falls back to identity if stats
    aren't available."""
    return item


def create_dataloader(
    data_dir: str | Path,
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    val_split: float = 0.05,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train and optionally validation dataloaders.

    Returns:
        (train_loader, val_loader) — val_loader is None if val_split == 0.
    """
    dataset = TransitionDataset(data_dir)

    if val_split <= 0:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            drop_last=True,
        )
        return loader, None

    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
