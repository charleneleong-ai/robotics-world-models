"""DreamZero single-GPU inference test on ManiSkill3.

Loads DreamZero (Wan2.2-TI2V-5B backbone) and runs inference on a ManiSkill task.
Requires: diffusers, transformers, accelerate, safetensors, groot (from dreamzero repo).
"""
import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/dreamzero")


def test_dreamzero_load():
    """Test if DreamZero model loads on single A100."""
    checkpoint_dir = "/home/ubuntu/dreamzero/checkpoints/DreamZero-DROID"
    
    print("=" * 60)
    print("DreamZero Model Load Test (single A100)")
    print("=" * 60)
    print()
    
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB")
    print()
    
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29599")
            dist.init_process_group(backend="nccl", rank=0, world_size=1)
            torch.cuda.set_device(0)
        
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        
        print("Loading GrootSimPolicy (this loads the full 14B model)...")
        t0 = time.time()
        
        policy = GrootSimPolicy.from_pretrained(checkpoint_dir)
        policy = policy.cuda()
        
        elapsed = time.time() - t0
        vram = torch.cuda.memory_allocated() / 1e9
        vram_max = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"\nModel loaded in {elapsed:.1f}s")
        print(f"VRAM used: {vram:.1f} GB (current), {vram_max:.1f} GB (peak)")
        print(f"Parameters: {sum(p.numel() for p in policy.parameters()):,}")
        
        # Test forward pass with dummy observation
        print("\nTesting inference with dummy observation...")
        
        # Create dummy observation (ManiSkill format)
        obs = {
            "agent_pos": torch.randn(1, 14).cuda(),  # joint positions
            "rgb": torch.randn(1, 3, 160, 320).cuda(),  # camera image
        }
        
        t1 = time.time()
        with torch.no_grad():
            action = policy.predict_action(obs)
        infer_time = time.time() - t1
        
        print(f"Inference: {infer_time:.3f}s")
        print(f"Action shape: {action.shape if hasattr(action, 'shape') else type(action)}")
        
        # VRAM after inference
        vram_after = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM after inference: {vram_after:.1f} GB")
        
        print("\n" + "=" * 60)
        print("RESULT: DreamZero loads and runs on single A100!")
        print("=" * 60)
        
        return True
        
    except torch.cuda.OutOfMemoryError:
        print("\nRESULT: DreamZero is too large for single A100 (OOM)")
        print("Options: multi-GPU, INT4 quantization, or CPU offloading")
        return False
        
    except Exception as e:
        print(f"\nRESULT: Failed - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dreamzero_load()
    sys.exit(0 if success else 1)
