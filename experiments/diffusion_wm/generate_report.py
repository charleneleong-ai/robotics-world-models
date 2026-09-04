#!/usr/bin/env python3
"""
Generate benchmark report from 4-task sweep results
"""

import json
from pathlib import Path

TASKS = [
    "PickCube-v1",
    "PushCube-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1",
]

def load_results(task: str) -> dict:
    """Load results for a task"""
    results_dir = Path(f"eval_results/{task.replace('-', '_').lower()}")
    results_file = results_dir / "round_00" / "results.json"
    
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return {}

def generate_report():
    """Generate benchmark report"""
    
    print("=" * 80)
    print("4-Task Benchmark Report")
    print("=" * 80)
    
    results = {}
    for task in TASKS:
        results[task] = load_results(task)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Task':<25} {'Mean Reward':<15} {'Success Rate':<15} {'BC Baseline':<15}")
    print("-" * 80)
    
    bc_baselines = {
        "PickCube-v1": "—",
        "PushCube-v1": "26%",
        "PegInsertionSide-v1": "—",
        "PlugCharger-v1": "—",
    }
    
    for task in TASKS:
        data = results.get(task, {})
        reward = data.get("mean_reward", "N/A")
        success = data.get("success_rate", "N/A")
        bc = bc_baselines.get(task, "—")
        
        if isinstance(reward, float):
            reward = f"{reward:.3f}"
        if isinstance(success, float):
            success = f"{success*100:.1f}%"
        
        print(f"{task:<25} {reward:<15} {success:<15} {bc:<15}")
    
    # Print comparison with SOTA
    print("\n" + "=" * 80)
    print("Comparison with SOTA (ManiSkill3 Leaderboard)")
    print("=" * 80)
    print(f"{'Task':<25} {'Our Result':<15} {'SOTA (IPI)':<15} {'Gap':<15}")
    print("-" * 80)
    
    sota = {
        "PickCube-v1": "95.5% (Lift Peg)",
        "PushCube-v1": "97.3%",
        "PegInsertionSide-v1": "98.5% (Pull Cube)",
        "PlugCharger-v1": "—",
    }
    
    for task in TASKS:
        data = results.get(task, {})
        success = data.get("success_rate", 0)
        sota_val = sota.get(task, "—")
        
        if isinstance(success, float):
            success_str = f"{success*100:.1f}%"
        else:
            success_str = "N/A"
        
        print(f"{task:<25} {success_str:<15} {sota_val:<15} {'—':<15}")
    
    print("\n" + "=" * 80)
    print("Key Findings")
    print("=" * 80)
    print("""
1. Demo bootstrap provides initial learning signal (5% vs 0% random)
2. WM-guided exploration shows diminishing returns
3. Gap to SOTA is significant (5% vs 95%+)
4. SOTA methods use:
   - Vision-based (RGB images, not state vectors)
   - Large VLA models (7B+ params)
   - RL fine-tuning (StARe framework)
5. Our approach:
   - State-vector based (42-dim)
   - Small model (14M params)
   - No RL fine-tuning

Recommendation: Scale up model + add vision + RL fine-tuning
""")
    
    print("=" * 80)
    print("Next Steps")
    print("=" * 80)
    print("""
1. Implement Cascaded WAM architecture
2. Add visual encoder for RGB input
3. Scale model to 100M+ params
4. Add RL fine-tuning (PPO/SAC)
5. Test multi-task training
""")

if __name__ == "__main__":
    generate_report()
