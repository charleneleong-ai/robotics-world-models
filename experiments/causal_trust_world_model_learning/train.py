"""Train causal trust world model learning components.

Usage:
    python -m experiments.causal_trust_world_model_learning.train \\
        --data-dir data/causal_trust/peginsertion \\
        --run-id causal-trust-v1 \\
        --num-steps 100000
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("WANDB_SILENT", "true")

import numpy as np
import torch
import torch.nn as nn
import typer
import wandb

from experiments.causal_trust_world_model_learning.world_model_verifier import (
    WorldModelVerifier,
)
from experiments.causal_trust_world_model_learning.trust_scoring import TrustScorer
from experiments.causal_trust_world_model_learning.causal_attribution import (
    CausalAttributionEngine,
)


@dataclass
class Config:
    data_dir: Path
    run_id: str
    project: str = "causal-trust-wm"
    entity: str = "chaleong"
    num_steps: int = 100_000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: Path = Path("checkpoints/causal_trust_wm")
    resume: str | None = None

    # Trust scoring weights
    physics_weight: float = 0.3
    ood_weight: float = 0.25
    calibration_weight: float = 0.2
    confidence_weight: float = 0.15
    historical_weight: float = 0.1
    trust_threshold: float = 0.7

    # Causal attribution weights
    contact_weight: float = 0.33
    visual_weight: float = 0.33
    dynamic_weight: float = 0.34

    # Data
    num_workers: int = 4
    val_split: float = 0.05
    seed: int = 42

    # Dimensions (populated at runtime)
    obs_dim: int | None = None
    act_dim: int | None = None


class WarmupCosineLR:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        for i, pg in enumerate(self.optimizer.param_groups):
            if self._step < self.warmup_steps:
                frac = self._step / max(1, self.warmup_steps)
                pg["lr"] = self.base_lrs[i] * frac
            else:
                frac = (self._step - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps
                )
                pg["lr"] = self.base_lrs[i] * 0.5 * (
                    1 + torch.cos(torch.tensor(frac * torch.pi)).item()
                )

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict[str, Any]:
        return {"_step": self._step, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict[str, Any]):
        self._step = state["_step"]
        self.base_lrs = state.get("base_lrs", self.base_lrs)


def get_obs_act_dim(data_dir: Path) -> tuple[int, int]:
    meta_path = data_dir / "meta" / "collection.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["obs_dim"] > 0 and meta["act_dim"] > 0:
            return meta["obs_dim"], meta["act_dim"]
    shards = sorted(data_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shards or meta found in {data_dir}")
    with np.load(shards[0], mmap_mode="r") as data:
        return data["obs"].shape[1], data["action"].shape[1]


class TrustScoringModel(nn.Module):
    """Neural network for trust scoring."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Input: [obs, action, predicted_next]
        input_dim = obs_dim + act_dim + obs_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),  # 5 trust signals
            nn.Sigmoid(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        predicted_next: torch.Tensor,
    ) -> torch.Tensor:
        """Compute trust signals.

        Args:
            obs: Current state [batch, obs_dim]
            action: Action taken [batch, act_dim]
            predicted_next: Predicted next state [batch, obs_dim]

        Returns:
            Trust signals [batch, 5] (physics, ood, calibration, confidence, historical)
        """
        x = torch.cat([obs, action, predicted_next], dim=-1)
        return self.network(x)


class CausalAttributionModel(nn.Module):
    """Neural network for causal attribution."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Input: [obs, action, predicted_next, actual_next]
        input_dim = obs_dim + act_dim + obs_dim + obs_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3 failure mechanisms
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        predicted_next: torch.Tensor,
        actual_next: torch.Tensor,
    ) -> torch.Tensor:
        """Compute causal attribution.

        Args:
            obs: Current state [batch, obs_dim]
            action: Action taken [batch, act_dim]
            predicted_next: Predicted next state [batch, obs_dim]
            actual_next: Actual next state [batch, obs_dim]

        Returns:
            Failure mechanism probabilities [batch, 3] (contact, visual, dynamic)
        """
        x = torch.cat([obs, action, predicted_next, actual_next], dim=-1)
        return self.network(x)


def create_dataloader(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 4,
    val_split: float = 0.05,
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
    """Create train and validation dataloaders."""

    class ShardDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir: Path, split: str = "train"):
            shards = sorted(data_dir.glob("shard_*.npz"))
            if not shards:
                raise FileNotFoundError(f"No shards found in {data_dir}")

            self.data = {"obs": [], "action": [], "next_obs": []}
            for shard_path in shards:
                with np.load(shard_path) as data:
                    self.data["obs"].append(data["obs"])
                    self.data["action"].append(data["action"])
                    self.data["next_obs"].append(data["next_obs"])

            self.data["obs"] = np.concatenate(self.data["obs"])
            self.data["action"] = np.concatenate(self.data["action"])
            self.data["next_obs"] = np.concatenate(self.data["next_obs"])

            # Split
            n = len(self.data["obs"])
            val_size = int(n * val_split)
            if split == "train":
                start, end = val_size, n
            else:
                start, end = 0, val_size

            self.data = {
                k: v[start:end] for k, v in self.data.items()
            }

        def __len__(self):
            return len(self.data["obs"])

        def __getitem__(self, idx):
            return {
                "obs": torch.tensor(self.data["obs"][idx], dtype=torch.float32),
                "action": torch.tensor(self.data["action"][idx], dtype=torch.float32),
                "next_obs": torch.tensor(self.data["next_obs"][idx], dtype=torch.float32),
            }

    train_dataset = ShardDataset(data_dir, split="train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_split > 0:
        val_dataset = ShardDataset(data_dir, split="val")
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


def main(
    data_dir: Path = typer.Option(..., help="Directory with shard_*.npz"),
    run_id: str = typer.Option("causal-trust-v1", help="W&B run name / checkpoint dir"),
    num_steps: int = typer.Option(100_000, help="Total training steps"),
    batch_size: int = typer.Option(256, help="Batch size"),
    lr: float = typer.Option(1e-4, help="Peak learning rate"),
    trust_threshold: float = typer.Option(0.7, help="Trust threshold"),
    resume: str | None = typer.Option(None, help="Checkpoint path to resume from"),
    project: str = typer.Option("causal-trust-wm"),
    eval_interval: int = typer.Option(1000, help="Steps between val loss logging"),
    device: str = typer.Option("cuda"),
):
    cfg = Config(
        data_dir=data_dir,
        run_id=run_id,
        num_steps=num_steps,
        batch_size=batch_size,
        lr=lr,
        trust_threshold=trust_threshold,
        resume=resume,
        project=project,
        eval_interval=eval_interval,
    )
    train(cfg)


def setup_training(cfg: Config, device: torch.device):
    obs_dim, act_dim = get_obs_act_dim(cfg.data_dir)
    cfg.obs_dim, cfg.act_dim = obs_dim, act_dim
    print(f"Data: {cfg.data_dir} — obs_dim={obs_dim}, act_dim={act_dim}")

    # Create models
    trust_model = TrustScoringModel(obs_dim, act_dim).to(device)
    causal_model = CausalAttributionModel(obs_dim, act_dim).to(device)

    param_count = sum(p.numel() for p in trust_model.parameters())
    param_count += sum(p.numel() for p in causal_model.parameters())
    print(f"Models: {param_count:,} parameters total")

    # Combine parameters
    params = list(trust_model.parameters()) + list(causal_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = WarmupCosineLR(optimizer, cfg.warmup_steps, cfg.num_steps)

    train_loader, val_loader = create_dataloader(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )

    ckpt_dir = cfg.checkpoint_dir / cfg.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        trust_model.load_state_dict(ckpt["trust_model"])
        causal_model.load_state_dict(ckpt["causal_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step} ({cfg.resume})")

    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=cfg.run_id,
        config={
            "num_params": param_count,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "trust_threshold": cfg.trust_threshold,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "num_steps": cfg.num_steps,
            "data_dir": str(cfg.data_dir),
        },
    )
    (ckpt_dir / "wandb_run_id.txt").write_text(wandb.run.id)

    return trust_model, causal_model, optimizer, scheduler, train_loader, val_loader, start_step, ckpt_dir


def save_checkpoint(
    path: Path,
    step: int,
    trust_model: nn.Module,
    causal_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cfg: Config,
    val_loss: float | None = None,
):
    torch.save(
        {
            "step": step,
            "trust_model": trust_model.state_dict(),
            "causal_model": causal_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg.__dict__,
            **({"val_loss": val_loss} if val_loss is not None else {}),
        },
        path,
    )


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required for training"

    (
        trust_model,
        causal_model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        start_step,
        ckpt_dir,
    ) = setup_training(cfg, device)

    step = start_step
    best_val_loss = float("inf")
    data_iter = iter(train_loader)
    start_time = time.monotonic()

    # Loss functions
    trust_loss_fn = nn.MSELoss()
    causal_loss_fn = nn.CrossEntropyLoss()

    while step < cfg.num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        obs = batch["obs"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        next_obs = batch["next_obs"].to(device, non_blocking=True)

        # Generate "predicted next" (simulate world model prediction)
        # In practice, this would come from the actual world model
        # Use mean of action to affect all obs dimensions
        action_effect = action.mean(dim=-1, keepdim=True).expand_as(obs) * 0.1
        predicted_next = obs + action_effect + torch.randn_like(obs) * 0.01

        # Generate ground truth trust signals (simplified)
        # In practice, these would come from actual verification
        physics_consistency = torch.ones(obs.shape[0], 1, device=device) * 0.8
        ood_score = torch.ones(obs.shape[0], 1, device=device) * 0.9
        calibration_error = torch.ones(obs.shape[0], 1, device=device) * 0.05
        confidence = torch.ones(obs.shape[0], 1, device=device) * 0.85
        historical = torch.ones(obs.shape[0], 1, device=device) * 0.8

        ground_truth_trust = torch.cat(
            [physics_consistency, ood_score, calibration_error, confidence, historical],
            dim=-1,
        )

        # Generate ground truth causal attribution (simplified)
        # In practice, these would come from actual failure analysis
        ground_truth_causal = torch.zeros(obs.shape[0], 3, device=device)
        ground_truth_causal[:, 0] = 0.6  # contact
        ground_truth_causal[:, 1] = 0.2  # visual
        ground_truth_causal[:, 2] = 0.2  # dynamic

        # Forward pass
        pred_trust = trust_model(obs, action, predicted_next)
        pred_causal = causal_model(obs, action, predicted_next, next_obs)

        # Compute losses
        trust_loss = trust_loss_fn(pred_trust, ground_truth_trust)
        causal_loss = causal_loss_fn(pred_causal, ground_truth_causal)

        # Combined loss
        loss = trust_loss + causal_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            list(trust_model.parameters()) + list(causal_model.parameters()),
            cfg.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        if step % cfg.log_interval == 0:
            elapsed = time.monotonic() - start_time
            steps_per_sec = (step - start_step + 1) / max(1, elapsed)
            lr_now = scheduler.current_lr
            print(
                f"step {step:07d}/{cfg.num_steps} | "
                f"trust_loss={trust_loss.item():.6f} | "
                f"causal_loss={causal_loss.item():.6f} | "
                f"lr={lr_now:.2e} | "
                f"{steps_per_sec:.0f} steps/s"
            )
            wandb.log(
                {
                    "train/trust_loss": trust_loss.item(),
                    "train/causal_loss": causal_loss.item(),
                    "train/total_loss": loss.item(),
                    "train/grad_norm": (
                        grad_norm.item()
                        if isinstance(grad_norm, torch.Tensor)
                        else grad_norm
                    ),
                    "train/lr": lr_now,
                    "train/steps_per_sec": steps_per_sec,
                },
                step=step,
            )

        if val_loader is not None and step > 0 and step % cfg.eval_interval == 0:
            val_loss = validate(trust_model, causal_model, val_loader, device)
            wandb.log({"val/loss": val_loss}, step=step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    step,
                    trust_model,
                    causal_model,
                    optimizer,
                    scheduler,
                    cfg,
                    val_loss=val_loss,
                )
                print(f"  best model saved (val_loss={val_loss:.6f})")

        if step > 0 and step % cfg.save_interval == 0:
            save_checkpoint(
                ckpt_dir / f"step_{step:07d}.pt",
                step,
                trust_model,
                causal_model,
                optimizer,
                scheduler,
                cfg,
            )
            print(f"  checkpoint saved (step {step})")

        step += 1

    save_checkpoint(
        ckpt_dir / "final.pt",
        step,
        trust_model,
        causal_model,
        optimizer,
        scheduler,
        cfg,
    )
    print(f"\nTraining complete. Final model: {ckpt_dir / 'final.pt'}")
    wandb.finish()


@torch.no_grad()
def validate(
    trust_model: nn.Module,
    causal_model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    trust_model.eval()
    causal_model.eval()
    total_loss = 0.0
    num_batches = 0

    trust_loss_fn = nn.MSELoss()
    causal_loss_fn = nn.CrossEntropyLoss()

    for batch in val_loader:
        obs = batch["obs"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        next_obs = batch["next_obs"].to(device, non_blocking=True)

        # Generate predicted next
        action_effect = action.mean(dim=-1, keepdim=True).expand_as(obs) * 0.1
        predicted_next = obs + action_effect + torch.randn_like(obs) * 0.01

        # Ground truth
        physics_consistency = torch.ones(obs.shape[0], 1, device=device) * 0.8
        ood_score = torch.ones(obs.shape[0], 1, device=device) * 0.9
        calibration_error = torch.ones(obs.shape[0], 1, device=device) * 0.05
        confidence = torch.ones(obs.shape[0], 1, device=device) * 0.85
        historical = torch.ones(obs.shape[0], 1, device=device) * 0.8

        ground_truth_trust = torch.cat(
            [physics_consistency, ood_score, calibration_error, confidence, historical],
            dim=-1,
        )

        ground_truth_causal = torch.zeros(obs.shape[0], 3, device=device)
        ground_truth_causal[:, 0] = 0.6
        ground_truth_causal[:, 1] = 0.2
        ground_truth_causal[:, 2] = 0.2

        # Forward
        pred_trust = trust_model(obs, action, predicted_next)
        pred_causal = causal_model(obs, action, predicted_next, next_obs)

        # Loss
        trust_loss = trust_loss_fn(pred_trust, ground_truth_trust)
        causal_loss = causal_loss_fn(pred_causal, ground_truth_causal)
        loss = trust_loss + causal_loss

        total_loss += loss.item()
        num_batches += 1

    trust_model.train()
    causal_model.train()
    return total_loss / max(1, num_batches)


if __name__ == "__main__":
    typer.run(main)
