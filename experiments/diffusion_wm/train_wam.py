"""Train a World Action Model (joint state + action denoising).

Usage:
    PYTHONPATH=. .venv/bin/python -m experiments.diffusion_wm.train_wam \\
        --data-dir data/diffusion_wm/peginsertion \\
        --run-id wam-peginsertion-v1 \\
        --num-steps 500000
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

from experiments.diffusion_wm.dataset import create_dataloader
from experiments.diffusion_wm.world_action_model import DiffusionWAM


@dataclass
class Config:
    data_dir: Path
    run_id: str
    project: str = "wm-manip"
    entity: str = "chaleong"
    num_steps: int = 500_000
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5000
    grad_clip: float = 1.0
    log_interval: int = 100
    eval_interval: int = 5000
    save_interval: int = 25000
    checkpoint_dir: Path = Path("checkpoints/diffusion_wm")
    resume: str | None = None

    # Model
    hidden_dim: int = 512
    num_blocks: int = 6
    cond_dim: int = 256
    diffusion_timesteps: int = 1000
    inference_steps: int = 100
    action_horizon: int = 1
    obs_dim: int | None = None
    act_dim: int | None = None

    # Data
    num_workers: int = 4
    val_split: float = 0.05
    seed: int = 42


class WarmupCosineLR:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self) -> None:
        self._step += 1
        for i, pg in enumerate(self.optimizer.param_groups):
            if self._step < self.warmup_steps:
                frac = self._step / max(1, self.warmup_steps)
                pg["lr"] = self.base_lrs[i] * frac
            else:
                frac = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                pg["lr"] = self.base_lrs[i] * 0.5 * (1 + torch.cos(torch.tensor(frac * torch.pi)).item())

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict[str, Any]:
        return {"_step": self._step, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
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


def setup_training(cfg: Config, device: torch.device) -> tuple:
    obs_dim, act_dim = get_obs_act_dim(cfg.data_dir)
    cfg.obs_dim, cfg.act_dim = obs_dim, act_dim
    print(f"Data: {cfg.data_dir} — obs_dim={obs_dim}, act_dim={act_dim}")

    model = DiffusionWAM(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        cond_dim=cfg.cond_dim,
        timesteps=cfg.diffusion_timesteps,
        action_horizon=cfg.action_horizon,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = WarmupCosineLR(optimizer, cfg.warmup_steps, cfg.num_steps)

    train_loader, val_loader = create_dataloader(
        cfg.data_dir, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, val_split=cfg.val_split, seed=cfg.seed,
    )

    ckpt_dir = cfg.checkpoint_dir / cfg.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step} ({cfg.resume})")

    wandb.init(
        project=cfg.project, entity=cfg.entity, name=cfg.run_id,
        config={
            "num_params": param_count, "obs_dim": obs_dim, "act_dim": act_dim,
            "hidden_dim": cfg.hidden_dim, "num_blocks": cfg.num_blocks,
            "diffusion_timesteps": cfg.diffusion_timesteps, "batch_size": cfg.batch_size,
            "lr": cfg.lr, "weight_decay": cfg.weight_decay, "num_steps": cfg.num_steps,
            "data_dir": str(cfg.data_dir), "model_type": "DiffusionWAM",
        },
    )
    (ckpt_dir / "wandb_run_id.txt").write_text(wandb.run.id)

    return model, optimizer, scheduler, train_loader, val_loader, start_step, ckpt_dir


def save_checkpoint(path: Path, step: int, model: DiffusionWAM, optimizer: torch.optim.Optimizer,
                    scheduler: WarmupCosineLR, cfg: Config, val_loss: float | None = None) -> None:
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": cfg.__dict__,
        **({"val_loss": val_loss} if val_loss is not None else {}),
    }, path)


@torch.no_grad()
def validate(model: DiffusionWAM, val_loader: torch.utils.data.DataLoader,
             device: torch.device) -> dict[str, float]:
    model.eval()
    total_state_loss = 0.0
    total_action_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    for batch in val_loader:
        obs = batch["obs"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        next_obs = batch["next_obs"].to(device, non_blocking=True)
        loss, losses = model.training_loss(obs, next_obs, action)
        total_state_loss += losses["state_loss"]
        total_action_loss += losses["action_loss"]
        total_loss += losses["total_loss"]
        num_batches += 1
    model.train()
    n = max(1, num_batches)
    return {
        "state_loss": total_state_loss / n,
        "action_loss": total_action_loss / n,
        "total_loss": total_loss / n,
    }


def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required for training"

    model, optimizer, scheduler, train_loader, val_loader, start_step, ckpt_dir = setup_training(cfg, device)

    step = start_step
    best_val_loss = float("inf")
    data_iter = iter(train_loader)
    start_time = time.monotonic()

    while step < cfg.num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        obs = batch["obs"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        next_obs = batch["next_obs"].to(device, non_blocking=True)

        loss, losses = model.training_loss(obs, next_obs, action)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % cfg.log_interval == 0:
            elapsed = time.monotonic() - start_time
            steps_per_sec = (step - start_step + 1) / max(1, elapsed)
            lr_now = scheduler.current_lr
            print(
                f"step {step:07d}/{cfg.num_steps} | "
                f"total={losses['total_loss']:.6f} | "
                f"state={losses['state_loss']:.6f} | "
                f"action={losses['action_loss']:.6f} | "
                f"lr={lr_now:.2e} | {steps_per_sec:.0f} steps/s"
            )
            wandb.log({
                "train/total_loss": losses["total_loss"],
                "train/state_loss": losses["state_loss"],
                "train/action_loss": losses["action_loss"],
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "train/lr": lr_now,
                "train/steps_per_sec": steps_per_sec,
            }, step=step)

        if val_loader is not None and step > 0 and step % cfg.eval_interval == 0:
            val_metrics = validate(model, val_loader, device)
            wandb.log({
                "val/total_loss": val_metrics["total_loss"],
                "val/state_loss": val_metrics["state_loss"],
                "val/action_loss": val_metrics["action_loss"],
            }, step=step)
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                save_checkpoint(ckpt_dir / "best.pt", step, model, optimizer, scheduler, cfg, val_loss=best_val_loss)
                print(f"  best model saved (val_loss={best_val_loss:.6f})")

        if step > 0 and step % cfg.save_interval == 0:
            save_checkpoint(ckpt_dir / f"step_{step:07d}.pt", step, model, optimizer, scheduler, cfg)
            print(f"  checkpoint saved (step {step})")

        step += 1

    save_checkpoint(ckpt_dir / "final.pt", step, model, optimizer, scheduler, cfg)
    print(f"\nTraining complete. Final model: {ckpt_dir / 'final.pt'}")
    wandb.finish()


def main(
    data_dir: Path = typer.Option(..., help="Directory with shard_*.npz"),
    run_id: str = typer.Option("wam-v1", help="Run name / checkpoint dir"),
    num_steps: int = typer.Option(500_000, help="Total training steps"),
    batch_size: int = typer.Option(1024, help="Batch size"),
    lr: float = typer.Option(3e-4, help="Peak learning rate"),
    hidden_dim: int = typer.Option(512, help="Hidden dimension"),
    num_blocks: int = typer.Option(6, help="Residual blocks"),
    diffusion_timesteps: int = typer.Option(1000, help="Diffusion timesteps"),
    inference_steps: int = typer.Option(100, help="Sampling steps"),
    action_horizon: int = typer.Option(1, help="Action prediction horizon"),
    resume: str | None = typer.Option(None, help="Checkpoint to resume from"),
    checkpoint_dir: Path = typer.Option(Path("checkpoints/diffusion_wm"), help="Checkpoint output dir"),
    project: str = typer.Option("wm-manip"),
    eval_interval: int = typer.Option(5000),
    device: str = typer.Option("cuda"),
) -> None:
    cfg = Config(
        data_dir=data_dir, run_id=run_id, num_steps=num_steps, batch_size=batch_size,
        lr=lr, hidden_dim=hidden_dim, num_blocks=num_blocks,
        diffusion_timesteps=diffusion_timesteps, inference_steps=inference_steps,
        action_horizon=action_horizon, resume=resume, checkpoint_dir=checkpoint_dir,
        project=project, eval_interval=eval_interval,
    )
    train(cfg)


if __name__ == "__main__":
    typer.run(main)
