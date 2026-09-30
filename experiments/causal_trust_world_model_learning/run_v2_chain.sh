#!/bin/bash
# Sequential re-run of every configuration with the real estimators.
cd /home/ubuntu/robotics_world_models/experiments/causal_trust_world_model_learning
for mode in n5 n9ext heldout alltasks; do
  echo "=== START $mode $(date -u +%H:%M:%SZ) ==="
  python3 -u rerun_v2.py "$mode" || echo "=== FAILED $mode ==="
  echo "=== END $mode $(date -u +%H:%M:%SZ) ==="
done
echo "=== CHAIN COMPLETE $(date -u +%H:%M:%SZ) ==="
