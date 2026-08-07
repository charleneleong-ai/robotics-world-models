"""Train an action-conditioned diffusion dynamics model.

Usage:
    python -m experiments.diffusion_wm.train \\
        --data-dir data/diffusion_wm/peginsertion \\
        --run-id peginsertion-diffusion-v1 \\
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
from experiments.diffusion_wm.model import DiffusionDynamics, MLPDenoiser


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

    # Data
    num_workers: int = 4
    val_split: float = 0.05
    seed: int = 42


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
                frac = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                pg["lr"] = self.base_lrs[i] * 0.5 * (1 + torch.cos(torch.tensor(frac * torch.pi)).item())

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


def main(
    data_dir: Path = typer.Option(..., help="Directory with shard_*.npz from collect.py"),
    run_id: str = typer.Option("diffusion-wm-v1", help="W&B run name / checkpoint dir"),
    num_steps: int = typer.Option(500_000, help="Total training steps"),
    batch_size: int = typer.Option(1024, help="Batch size"),
    lr: float = typer.Option(3e-4, help="Peak learning rate"),
    hidden_dim: int = typer.Option(512, help="MLP hidden dimension"),
    num_blocks: int = typer.Option(6, help="Number of residual blocks"),
    diffusion_timesteps: int = typer.Option(1000, help="Diffusion timesteps (train)"),
    inference_steps: int = typer.Option(100, help="Sampling steps (eval)"),
    resume: str | None = typer.Option(None, help="Checkpoint path to resume from"),
    project: str = typer.Option("wm-manip"),
    device: str = typer.Option("cuda"),
):
    cfg = Config(
        data_dir=data_dir,
        run_id=run_id,
        num_steps=num_steps,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        diffusion_timesteps=diffusion_timesteps,
        inference_steps=inference_steps,
        resume=resume,
        project=project,
    )
    train(cfg)


def setup_training(cfg, device):
    obs_dim, act_dim = get_obs_act_dim(cfg.data_dir)
    print(f"Data: {cfg.data_dir} — obs_dim={obs_dim}, act_dim={act_dim}")

    denoiser = MLPDenoiser(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=cfg.hidden_dim, num_blocks=cfg.num_blocks, cond_dim=cfg.cond_dim)
    model = DiffusionDynamics(denoiser, timesteps=cfg.diffusion_timesteps).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = WarmupCosineLR(optimizer, cfg.warmup_steps, cfg.num_steps)

    train_loader, val_loader = create_dataloader(cfg.data_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers, val_split=cfg.val_split, seed=cfg.seed)
    ckpt_dir = cfg.checkpoint_dir / cfg.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step} ({cfg.resume})")

    wandb.init(project=cfg.project, entity=cfg.entity, name=cfg.run_id, config={
        "num_params": param_count, "obs_dim": obs_dim, "act_dim": act_dim,
        "hidden_dim": cfg.hidden_dim, "num_blocks": cfg.num_blocks,
        "diffusion_timesteps": cfg.diffusion_timesteps, "batch_size": cfg.batch_size,
        "lr": cfg.lr, "weight_decay": cfg.weight_decay, "num_steps": cfg.num_steps,
        "data_dir": str(cfg.data_dir),
    })

    return model, optimizer, scheduler, train_loader, val_loader, start_step, ckpt_dir


def save_checkpoint(path, step, model, optimizer, scheduler, cfg, val_loss=None):
    torch.save({
        "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "config": cfg.__dict__,
        **({"val_loss": val_loss} if val_loss is not None else {}),
    }, path)


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required for training"

    model, optimizer, scheduler, train_loader, val_loader, start_step, ckpt_dir = setup_training(cfg, device)

    step = start_step
    best_val_loss = float("inf")
    val_loss: float | None = None
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

        loss = model(next_obs, obs, action)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % cfg.log_interval == 0:
            elapsed = time.monotonic() - start_time
            steps_per_sec = (step - start_step + 1) / max(1, elapsed)
            lr_now = scheduler.current_lr
            print(f"step {step:07d}/{cfg.num_steps} | loss={loss.item():.6f} | lr={lr_now:.2e} | {steps_per_sec:.0f} steps/s")
            wandb.log({
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "train/lr": lr_now,
                "train/steps_per_sec": steps_per_sec,
            }, step=step)

        if val_loader is not None and step > 0 and step % cfg.eval_interval == 0:
            val_loss = validate(model, val_loader, device)
            wandb.log({"val/loss": val_loss}, step=step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(ckpt_dir / "best.pt", step, model, optimizer, scheduler, cfg, val_loss=val_loss)
                print(f"  best model saved (val_loss={val_loss:.6f})")

        if step > 0 and step % cfg.save_interval == 0:
            save_checkpoint(ckpt_dir / f"step_{step:07d}.pt", step, model, optimizer, scheduler, cfg)
            print(f"  checkpoint saved (step {step})")

        step += 1

    save_checkpoint(ckpt_dir / "final.pt", step, model, optimizer, scheduler, cfg)
    print(f"\nTraining complete. Final model: {ckpt_dir / 'final.pt'}")
    wandb.finish()


@torch.no_grad()
def validate(model: DiffusionDynamics, val_loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in val_loader:
        obs = batch["obs"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        next_obs = batch["next_obs"].to(device, non_blocking=True)
        loss = model(next_obs, obs, action)
        total_loss += loss.item()
        num_batches += 1
    model.train()
    return total_loss / max(1, num_batches)


if __name__ == "__main__":
    typer.run(main)
