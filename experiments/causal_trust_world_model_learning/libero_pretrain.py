#!/usr/bin/env python3
"""
Pretrain world model on LIBERO expert demonstrations, then fine-tune with trust.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from full_backbone_sweep import (
    BACKBONES, EMATrust, MultiStepAdaptiveTrust,
    FFDCMultiStep, EnsembleDisagreement
)

# ============================================================================
# DEMO LOADER
# ============================================================================

def load_libero_demos(suite: str, task_idx: int, max_demos: int = 10):
    """Load LIBERO expert demonstrations for a specific task."""
    import h5py
    
    suite_dir = Path("/home/ubuntu/robotics_world_models/LIBERO") / suite
    hdf5_files = sorted(suite_dir.glob("*.hdf5"))
    
    if task_idx >= len(hdf5_files):
        raise ValueError(f"Task {task_idx} not found in {suite} (has {len(hdf5_files)} tasks)")
    
    hdf5_path = hdf5_files[task_idx]
    task_name = hdf5_path.stem.replace("_demo", "")
    
    print(f"Loading demos from: {hdf5_path.name}")
    
    demos = []
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys())[:max_demos]
        
        for dk in demo_keys:
            demo = f["data"][dk]
            
            # Extract state features (no images)
            obs = demo["obs"]
            ee_pos = obs["ee_pos"][:]          # (T, 3)
            ee_ori = obs["ee_ori"][:]          # (T, 3)
            joint_states = obs["joint_states"][:]  # (T, 7)
            gripper_states = obs["gripper_states"][:]  # (T, 2)
            
            # Concatenate state features
            obs_seq = np.concatenate([
                ee_pos, ee_ori, joint_states, gripper_states
            ], axis=-1)  # (T, 15)
            
            actions = demo["actions"][:]  # (T, 7)
            
            demos.append({
                "obs": obs_seq.astype(np.float32),
                "actions": actions.astype(np.float32),
                "length": len(obs_seq)
            })
    
    print(f"Loaded {len(demos)} demos, avg length: {np.mean([d['length'] for d in demos]):.0f}")
    return demos, task_name


# ============================================================================
# WORLD MODEL PRETRAINING
# ============================================================================

def pretrain_world_model(backbone_name: str, demos: list, n_epochs: int = 50, device: torch.device = None):
    """Pretrain world model on expert demonstrations."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    obs_dim = demos[0]["obs"].shape[-1]
    act_dim = demos[0]["actions"].shape[-1]
    
    # Build backbone
    wm = BACKBONES[backbone_name](obs_dim, act_dim).to(device)
    
    optimizer = torch.optim.Adam(wm.parameters(), lr=1e-3)
    
    print(f"Pretraining {backbone_name} on {len(demos)} demos (device={device})...")
    
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for demo in demos:
            obs = torch.tensor(demo["obs"], device=device).unsqueeze(0)  # (1, T, obs_dim)
            actions = torch.tensor(demo["actions"], device=device).unsqueeze(0)  # (1, T, act_dim)
            
            # Use train_loss which handles sequences
            loss = wm.train_loss(obs, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")
    
    return wm, losses


# ============================================================================
# FINE-TUNE WITH TRUST
# ============================================================================

def finetune_with_trust(wm, trust_name: str, demos: list, n_episodes: int = 5, 
                        max_steps: int = 100, trust_threshold: float = 0.5, device: torch.device = None):
    """Fine-tune world model with trust-weighted consolidation."""
    if device is None:
        device = next(wm.parameters()).device
    
    obs_dim = demos[0]["obs"].shape[-1]
    act_dim = demos[0]["actions"].shape[-1]
    
    # Initialize trust scorer
    if trust_name == "ema":
        trust = EMATrust()
    elif trust_name == "multi_step":
        trust = MultiStepAdaptiveTrust()
    elif trust_name == "ffdc":
        trust = FFDCMultiStep(obs_dim, act_dim).to(device)
    elif trust_name == "ensemble":
        trust = EnsembleDisagreement(obs_dim).to(device)
    else:
        trust = None
    
    device = next(wm.parameters()).device
    
    # Store original parameters for EWC-style consolidation
    orig_params = {n: p.clone().detach() for n, p in wm.named_parameters()}
    
    rewards = []
    for ep in range(n_episodes):
        # Random demo
        demo = demos[np.random.randint(len(demos))]
        obs_seq = demo["obs"]
        act_seq = demo["actions"]
        
        ep_reward = 0.0
        for t in range(min(max_steps, len(obs_seq) - 1)):
            obs_t = torch.tensor(obs_seq[t:t+1], device=device)
            act_t = torch.tensor(act_seq[t:t+1], device=device)
            obs_next = torch.tensor(obs_seq[t+1:t+2], device=device)
            
            # Compute prediction error
            error = wm.predict_error(obs_t, act_t, obs_next)
            
            # Compute trust
            if trust is not None:
                if trust_name in ["ema", "multi_step"]:
                    trust_score = trust.compute_trust(error, task_id=0)
                elif trust_name == "ffdc":
                    trust_score = trust.compute_trust(obs_t, obs_next, act_t)
                elif trust_name == "ensemble":
                    trust_score = trust.compute_trust(obs_t)
                
                trust_score = trust_score.mean().item()
            else:
                trust_score = 1.0
            
            # Trust-weighted update
            if trust_score > trust_threshold:
                loss = error.mean()
                
                # EWC-style regularization
                ewc_loss = 0.0
                for n, p in wm.named_parameters():
                    if n in orig_params:
                        ewc_loss += F.mse_loss(p, orig_params[n])
                
                total_loss = loss + 0.1 * ewc_loss
                
                optimizer = torch.optim.Adam(wm.parameters(), lr=1e-4)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            ep_reward += -error.mean().item()
        
        rewards.append(ep_reward)
        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: reward={ep_reward:.3f}, avg_trust={trust_score:.3f}")
    
    return wm, rewards


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial", 
                        choices=["libero_spatial", "libero_object", "libero_goal"])
    parser.add_argument("--n-tasks", type=int, default=3)
    parser.add_argument("--backbones", nargs="+", default=["mlp", "rssm"])
    parser.add_argument("--trusts", nargs="+", default=["none", "ema", "multi_step"])
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-episodes", type=int, default=10)
    parser.add_argument("--output", default="libero_pretrain_results.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    
    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("LIBERO Expert Demo Pretraining + Trust Fine-tuning")
    print("="*60)
    print(f"Suite: {args.suite}")
    print(f"Tasks: {args.n_tasks}")
    print(f"Backbones: {args.backbones}")
    print(f"Trusts: {args.trusts}")
    print(f"Device: {device}")
    print()
    
    # Init wandb
    if HAS_WANDB and not args.no_wandb:
        wandb.init(
            project="continualwam",
            name=f"libero_{args.suite}_{device.type}",
            config=vars(args),
            reinit=True
        )
    
    results = []
    
    for bb_name in args.backbones:
        for trust_name in args.trusts:
            print(f"\n{'='*40}")
            print(f"Config: {bb_name} + {trust_name}")
            print(f"{'='*40}")
            
            all_rewards = []
            
            for task_idx in range(args.n_tasks):
                print(f"\n--- Task {task_idx} ---")
                
                # Load demos
                demos, task_name = load_libero_demos(args.suite, task_idx)
                
                # Pretrain
                wm, pretrain_losses = pretrain_world_model(
                    bb_name, demos, n_epochs=args.pretrain_epochs, device=device
                )
                
                # Fine-tune with trust
                wm, finetune_rewards = finetune_with_trust(
                    wm, trust_name, demos, 
                    n_episodes=args.finetune_episodes, device=device
                )
                
                all_rewards.append(np.mean(finetune_rewards))
                
                if HAS_WANDB and not args.no_wandb:
                    wandb.log({
                        f"{bb_name}_{trust_name}/task_{task_idx}_reward": np.mean(finetune_rewards),
                        f"{bb_name}_{trust_name}/task_{task_idx}_pretrain_loss": pretrain_losses[-1],
                        "suite": args.suite,
                        "device": str(device),
                    })
            
            avg_reward = np.mean(all_rewards)
            print(f"\n>>> Average reward: {avg_reward:.3f}")
            
            if HAS_WANDB and not args.no_wandb:
                wandb.log({
                    f"{bb_name}_{trust_name}/avg_reward": avg_reward,
                    f"{bb_name}_{trust_name}/all_task_rewards": all_rewards,
                })
            
            results.append({
                "benchmark": f"libero_{args.suite}",
                "backbone": bb_name,
                "trust": trust_name,
                "avg_reward": avg_reward,
                "task_rewards": all_rewards
            })
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {args.output}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary:")
    for r in results:
        print(f"  {r['backbone']:12} + {r['trust']:12}: {r['avg_reward']:.3f}")
    
    if HAS_WANDB and not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
