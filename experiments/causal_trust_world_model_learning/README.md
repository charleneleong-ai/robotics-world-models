# Causal Trust World Model Learning

Trust-aware learning, error recovery, and easier capability acquisition in simulation.

## Paper

- **Title**: Causal Verification-First World Models: Trust-Aware Learning, Error Recovery, 
  and Easier Robot Capability Acquisition in Simulation
- **Workshop**: NeurIPS 2026 Robotics_World_Modeling
- **Paper**: https://github.com/charleneleong-ai/neurips-workshops-2026/tree/main/papers/robotics-world-modeling

## Structure

| File | Purpose |
|------|---------|
| `world_model_verifier.py` | Verification backbone (trust scoring, calibration) |
| `vla_policy.py` | VLA action generation with candidate selection |
| `causal_attribution.py` | Causal diagnosis of verification failures |
| `trust_scoring.py` | Multi-signal trust computation |
| `recovery_strategies.py` | Error recovery (retry, reset, causal) |
| `train.py` | Training loop (following diffusion_wm pattern) |
| `eval.py` | Evaluation metrics and benchmarking |
| `test_model.py` | Unit tests |
| `benchmarks/` | LIBERO, ManiSkill, KinDER evaluation |
| `results/` | Experiment results and analysis |

## Key Contributions

1. **First hybrid verification-first architecture**: World model verification + VLA action generation
2. **First causal error diagnosis**: Mechanism-specific failure attribution
3. **First trust-aware candidate selection**: K candidates, select by trust score
4. **First causal recovery strategies**: Contact, visual, dynamic mechanisms
5. **Continual learning via verification**: Trust scoring enables learning new capabilities while preserving old knowledge

## Benchmarks

- **LIBERO**: Spatial, Object, Goal, Long tasks
- **ManiSkill**: PickCube, PegInsertionSide, PlugCharger
- **KinDER**: Physical reasoning challenges

## Setup

```bash
# Install dependencies
pip install -e .

# Run training
python -m experiments.causal_trust_world_model_learning.train --config configs/default.yaml

# Run evaluation
python -m experiments.causal_trust_world_model_learning.eval --checkpoint checkpoints/latest.pt
```

## Cross-References

- **Experiment code**: This directory
- **Paper**: https://github.com/charleneleong-ai/neurips-workshops-2026/tree/main/papers/robotics-world-modeling
- **Foundation**: `robotics_world_models/experiments/diffusion_wm/` (world model backbone)
