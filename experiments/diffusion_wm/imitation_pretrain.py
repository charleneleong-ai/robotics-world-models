"""Imitation pre-training (Behavior Cloning) on demonstration data.

Trains the WAM policy on expert demonstrations before self-driving loop.
Supports both state-only BC and joint state+action BC.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class BCPretrainConfig:
    """Configuration for BC pre-training."""
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    val_split: float = 0.1
    seed: int = 42
    # BC specific
    loss_type: str = "mse"  # "mse" or "huber"
    grad_clip: float = 1.0
    # Checkpointing
    save_best: bool = True
    save_every: int = 10


class BCPretrainer:
    """Behavior Cloning pre-trainer for WAM."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: BCPretrainConfig,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # LR scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Loss function
        if config.loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif config.loss_type == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")
        
        self.best_val_loss = float('inf')
        self.epoch = 0
    
    def _create_scheduler(self):
        """Create LR scheduler with warmup."""
        # We'll manually handle warmup
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
        )
    
    def prepare_dataloaders(
        self, 
        demo_dir: Path, 
        task: str,
        max_episodes: int = 100,
        max_steps: int = 200,
    ) -> tuple[DataLoader, DataLoader]:
        """Load demonstration data and create train/val dataloaders."""
        from experiments.diffusion_wm.collector import load_demonstration_data
        
        # Load demo data
        data_dir = load_demonstration_data(
            task=task,
            demo_dir=demo_dir,
            max_episodes=max_episodes,
            max_steps=max_steps,
        )
        
        # Collect all transitions
        all_obs = []
        all_actions = []
        all_next_states = []
        
        import glob
        for shard_path in sorted(data_dir.glob("shard_*.npz")):
            data = np.load(shard_path)
            all_obs.append(data["obs"])
            all_actions.append(data["action"])
            all_next_states.append(data["next_obs"])
        
        if not all_obs:
            raise ValueError("No demonstration data found")
        
        obs = np.concatenate(all_obs, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        next_states = np.concatenate(all_next_states, axis=0)
        
        # Create dataset
        dataset = TensorDataset(
            torch.from_numpy(obs).float(),
            torch.from_numpy(actions).float(),
            torch.from_numpy(next_states).float(),
        )
        
        # Split train/val
        val_size = int(len(dataset) * self.config.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_state_loss = 0.0
        total_action_loss = 0.0
        num_batches = 0
        
        for obs, action, next_state in train_loader:
            obs = obs.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass: compute loss on both state and action
            # Training loss: predict action + next_state from obs
            state_loss, action_loss = self._compute_loss(obs, next_state, action)
            loss = state_loss + action_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_state_loss += state_loss.item()
            total_action_loss += action_loss.item()
            num_batches += 1
        
        self.scheduler.step()
        self.epoch += 1
        
        return {
            "loss": total_loss / num_batches,
            "state_loss": total_state_loss / num_batches,
            "action_loss": total_action_loss / num_batches,
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_state_loss = 0.0
        total_action_loss = 0.0
        num_batches = 0
        
        for obs, action, next_state in val_loader:
            obs = obs.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            
            state_loss, action_loss = self._compute_loss(obs, next_state, action)
            loss = state_loss + action_loss
            
            total_loss += loss.item()
            total_state_loss += state_loss.item()
            total_action_loss += action_loss.item()
            num_batches += 1
        
        return {
            "val_loss": total_loss / num_batches,
            "state_loss": total_state_loss / num_batches,
            "action_loss": total_action_loss / num_batches,
        }
    
    def _compute_loss(
        self, 
        obs: torch.Tensor, 
        next_state: torch.Tensor, 
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute state and action prediction losses."""
        # This calls the model's training_loss or compute_loss method
        if hasattr(self.model, 'training_loss'):
            return self.model.training_loss(obs, next_state, action)
        elif hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(obs, next_state, action)
        else:
            # Fallback: use denoiser directly
            raise NotImplementedError("Model must have training_loss or compute_loss method")
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        checkpoint_dir: Path,
    ) -> list[dict]:
        """Full training loop."""
        history = []
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config.epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # Combine metrics
            metrics = {
                "epoch": self.epoch,
                **train_metrics,
                **val_metrics,
                "lr": self.optimizer.param_groups[0]['lr'],
            }
            history.append(metrics)
            
            # Log
            print(f"Epoch {self.epoch:3d}: "
                  f"train_loss={train_metrics['loss']:.4f}, "
                  f"val_loss={val_metrics['val_loss']:.4f}, "
                  f"lr={metrics['lr']:.2e}")
            
            # Save best model
            if self.config.save_best and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(checkpoint_dir / "best.pt")
                print(f"  ✓ Best model saved (val_loss={self.best_val_loss:.4f})")
            
            # Periodic save
            if self.config.save_every > 0 and self.epoch % self.config.save_every == 0:
                self.save_checkpoint(checkpoint_dir / f"epoch_{self.epoch:03d}.pt")
        
        return history
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config.__dict__,
            "best_val_loss": self.best_val_loss,
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt["epoch"]
        self.best_val_loss = ckpt.get("best_val_loss", float('inf'))
        print(f"Loaded checkpoint from {path} (epoch {self.epoch})")


def run_bc_pretrain(
    model: torch.nn.Module,
    demo_dir: Path,
    task: str,
    config: BCPretrainConfig,
    checkpoint_dir: Path,
    max_episodes: int = 100,
    max_steps: int = 200,
) -> tuple[torch.nn.Module, list[dict]]:
    """Convenience function to run BC pre-training."""
    pretrainer = BCPretrainer(model, config)
    train_loader, val_loader = pretrainer.prepare_dataloaders(
        Path(demo_dir), task, max_episodes, max_steps
    )
    history = pretrainer.train(pretrainer.train_loader, pretrainer.val_loader, Path(checkpoint_dir))
    return pretrainer.model, history
