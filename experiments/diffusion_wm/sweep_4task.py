#!/usr/bin/env python3
"""
4-Task Benchmark Sweep
Runs DiffusionWAM with demo bootstrap on all 4 ManiSkill3 tasks
"""

import subprocess
import time
from pathlib import Path

TASKS = [
    "PickCube-v1",
    "PushCube-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1",
]

ROUNDS = 5
EPISODES_PER_ROUND = 20

def run_sweep():
    """Run 4-task benchmark sweep"""
    
    print("=" * 60)
    print("4-Task Benchmark Sweep")
    print("=" * 60)
    
    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")
        
        # Create log directory
        log_dir = Path(f"logs/sweep_{task.replace('-', '_').lower()}")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Run sweep
        cmd = [
            "python", "-u", "experiments/diffusion_wm/loop.py",
            "--task", task,
            "--num-rounds", str(ROUNDS),
            "--episodes-per-round", str(EPISODES_PER_ROUND),
            "--demo-dir", f"/tmp/mp_demos/{task}/",
        ]
        
        log_file = log_dir / f"sweep_{int(time.time())}.log"
        print(f"Logging to: {log_file}")
        
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd="/home/ubuntu/robotics_world_models",
            )
            
            # Wait for completion
            process.wait()
            
            if process.returncode == 0:
                print(f"✓ {task} completed successfully")
            else:
                print(f"✗ {task} failed with return code {process.returncode}")
        
        # Brief pause between tasks
        time.sleep(5)
    
    print("\n" + "=" * 60)
    print("Sweep Complete!")
    print("=" * 60)
    print("\nResults saved to eval_results/")

if __name__ == "__main__":
    run_sweep()
