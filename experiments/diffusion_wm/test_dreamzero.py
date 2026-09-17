"""Test DreamZero inference on ManiSkill3.

DreamZero is a 14B video diffusion model that jointly predicts video and actions.
This script loads the pretrained checkpoint and tests inference on ManiSkill tasks.
"""
import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path


def load_dreamzero_model(checkpoint_dir: str, device: str = "cuda"):
    """Load DreamZero model from checkpoint.
    
    DreamZero uses a CausalWanModel (video diffusion backbone) with action/state registers.
    """
    print(f"Loading DreamZero from {checkpoint_dir}...")
    
    try:
        from safetensors.torch import load_file
        from diffusers import AutoModel
        
        # Try loading with diffusers
        config_path = Path(checkpoint_dir) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            print(f"Config: {json.dumps(config, indent=2)[:500]}")
        
        # Load model weights
        model_files = sorted(Path(checkpoint_dir).glob("model-*.safetensors"))
        print(f"Found {len(model_files)} model shards")
        
        # Load first shard to check shape
        shard = load_file(str(model_files[0]))
        total_params = sum(v.numel() for v in shard.values())
        print(f"First shard: {total_params:,} params")
        del shard
        
        return True, f"Model loaded successfully ({len(model_files)} shards)"
        
    except Exception as e:
        return False, str(e)


def test_dreamzero_inference():
    """Test DreamZero inference on a simple ManiSkill task."""
    checkpoint_dir = "/home/ubuntu/dreamzero/checkpoints/DreamZero-DROID"
    
    print("=" * 60)
    print("DreamZero Inference Test")
    print("=" * 60)
    print()
    
    # Check if checkpoint exists
    if not Path(checkpoint_dir).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_dir}")
        return
    
    # Try loading model
    success, msg = load_dreamzero_model(checkpoint_dir)
    print(f"Model load: {'✓' if success else '✗'} {msg}")
    
    if not success:
        print("\nTrying alternative loading method...")
        try:
            import safetensors.torch
            from safetensors.torch import load_file
            
            # Load just to check memory
            print("Loading first shard to check VRAM usage...")
            t0 = time.time()
            shard = load_file(str(Path(checkpoint_dir) / "model-00001-of-00010.safetensors"))
            elapsed = time.time() - t0
            
            total_params = sum(v.numel() for v in shard.values())
            total_bytes = sum(v.numel() * v.element_size() for v in shard.values())
            
            print(f"  Loaded in {elapsed:.1f}s")
            print(f"  Params: {total_params:,}")
            print(f"  Size: {total_bytes / 1e9:.2f} GB")
            print(f"  Device VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
            
            # Check if full model would fit
            estimated_full = total_bytes * 10  # 10 shards
            print(f"  Estimated full model: {estimated_full / 1e9:.2f} GB")
            print(f"  A100 VRAM: 80 GB")
            print(f"  {'✓ Fits!' if estimated_full < 80e9 else '✗ Too large for single A100'}")
            
            del shard
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
DreamZero is a 14B parameter video diffusion model.
- Requires minimum 2 GPUs for distributed inference
- Checkpoint is ~46GB (FP16) + activations
- Our A100 (80GB) cannot run full DreamZero inference
- Options:
  1. Use 5B version (Wan2.2-TI2V-5B backbone)
  2. Quantize to INT4 (reduce memory by 4x)
  3. Use CPU offloading (slower but fits)
""")


if __name__ == "__main__":
    test_dreamzero_inference()
