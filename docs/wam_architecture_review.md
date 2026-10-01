# World Action Model (WAM) Architecture Review

## Literature Survey

### What is a World Action Model?

From Wang et al. 2026 (arxiv:2605.12090):
> "World Action Models (WAMs): embodied foundation models that unify predictive state modeling with action generation, targeting a joint distribution over future states and actions rather than actions alone."

### WAM Taxonomy (Wang et al. 2026)

```
┌─────────────────────────────────────────────────────────────────┐
│                    World Action Models (WAMs)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │   Cascaded WAMs      │    │    Joint WAMs        │          │
│  │                      │    │                      │          │
│  │  World Model →       │    │  Joint P(s', a | s)  │          │
│  │  Action Decoder      │    │  Single model        │          │
│  │                      │    │  predicts both       │          │
│  └──────────────────────┘    └──────────────────────┘          │
│                                                                  │
│  Examples:                          Examples:                    │
│  - UniPi                            - DreamZero (14B)           │
│  - SayCan                           - Our DiffusionWAM (14M)    │
│  - RT-2                             - GATO                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### State-of-the-Art Comparison

| Model | Scale | Input | Output | Architecture | Key Innovation |
|-------|-------|-------|--------|--------------|----------------|
| **DreamZero** | 14B | Video frames | Video + actions | Video diffusion | Zero-shot transfer |
| **UniPi** | ~100M | Images | Actions | Cascaded | Video planning |
| **RT-2** | 55B | Images | Actions | VLA | Language grounding |
| **Our DiffusionWAM** | 14.4M | State vectors | State + action | Joint diffusion | CEM planning |

## Our Implementation

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      DiffusionWAM (14.4M params)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: current_obs (42-dim state vector)                      │
│          ┌─────────────────────────────────────────┐            │
│          │  ManiSkill3 observation:                 │            │
│          │  - Robot joint positions (7)             │            │
│          │  - Robot joint velocities (7)            │            │
│          │  - TCP pose (7)                          │            │
│          │  - Object pose (7)                       │            │
│          │  - Goal pose (7)                         │            │
│          │  - Additional features (7)               │            │
│          └─────────────────────────────────────────┘            │
│                          │                                       │
│                          ▼                                       │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              WAMDenoiser (Shared Backbone)                │  │
│   │                                                            │  │
│   │   ┌────────────────────────────────────────────────────┐  │  │
│   │   │  Input Projection                                  │  │  │
│   │   │  [obs, noisy_target] → hidden_dim (512)            │  │  │
│   │   └────────────────────────────────────────────────────┘  │  │
│   │                          │                                 │  │
│   │                          ▼                                 │  │
│   │   ┌────────────────────────────────────────────────────┐  │  │
│   │   │  Timestep Embedding + FiLM Modulation              │  │  │
│   │   │  t → sinusoidal → MLP → scale/bias per block       │  │  │
│   │   └────────────────────────────────────────────────────┘  │  │
│   │                          │                                 │  │
│   │                          ▼                                 │  │
│   │   ┌────────────────────────────────────────────────────┐  │  │
│   │   │  6x Transformer Blocks                             │  │  │
│   │   │  [LayerNorm → FiLM → GELU → Residual]              │  │  │
│   │   └────────────────────────────────────────────────────┘  │  │
│   │                          │                                 │  │
│   │              ┌───────────┴───────────┐                     │  │
│   │              ▼                       ▼                     │  │
│   │   ┌─────────────────┐    ┌─────────────────┐              │  │
│   │   │   State Head    │    │  Action Head    │              │  │
│   │   │                 │    │                 │              │  │
│   │   │  LayerNorm →    │    │  LayerNorm →    │              │  │
│   │   │  Linear(512→42) │    │  Linear(512→8)  │              │  │
│   │   │                 │    │                 │              │  │
│   │   │  Output:        │    │  Output:        │              │  │
│   │   │  predicted      │    │  predicted      │              │  │
│   │   │  state noise    │    │  action noise   │              │  │
│   │   └─────────────────┘    └─────────────────┘              │  │
│   │                                                            │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   Training: Joint MSE loss on both heads                       │
│   Inference: Denoise random noise → action (100 DDPM steps)    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### CEM Planner Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Cross-Entropy Method (CEM) Planner                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: current_state (42-dim)                                 │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Iteration 1..N (N=5):                                   │  │
│   │                                                            │  │
│   │  1. Sample K=100 action sequences                        │  │
│   │     actions ~ N(mean, std)  [K, horizon=8, act_dim=8]    │  │
│   │                                                            │  │
│   │  2. For each sequence:                                    │  │
│   │     ┌────────────────────────────────────────────────┐    │  │
│   │     │  Simulate through WAM:                         │    │  │
│   │     │  for t in 0..horizon:                          │    │  │
│   │     │    next_state = WAM.denoise_state(state, t)    │    │  │
│   │     │    state = next_state                          │    │  │
│   │     │                                                │    │  │
│   │     │  Score = reward + exploration_bonus(uncertainty)│   │  │
│   │     └────────────────────────────────────────────────┘    │  │
│   │                                                            │  │
│   │  3. Select top-K sequences by score                       │  │
│   │                                                            │  │
│   │  4. Refit distribution:                                    │  │
│   │     mean = topk_actions.mean(dim=0)                       │  │
│   │     std = topk_actions.std(dim=0).clamp(min=0.01)         │  │
│   │                                                            │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   Output: best_action (8-dim) — first action of best sequence  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Self-Driving Learning Loop

```
┌─────────────────────────────────────────────────────────────────┐
│              Self-Driving Learning Loop                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Round 0: Demo Bootstrap                                 │  │
│   │                                                            │  │
│   │  ManiSkill Expert Demos (H5) → Replay → Collect          │  │
│   │  20 episodes, 898 transitions                             │  │
│   │                                                            │  │
│   │  Train WAM on demo data                                   │  │
│   │  Loss: 0.06 (6x better than random 0.40)                 │  │
│   │                                                            │  │
│   │  Result: 5% success rate (vs 0% random)                   │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Round 1+: WM-Guided Exploration                         │  │
│   │                                                            │  │
│   │  ┌────────────────────────────────────────────────────┐   │  │
│   │  │  CEM Planning through trained WAM                  │   │  │
│   │  │  - Sample 100 action sequences                     │   │  │
│   │  │  - Simulate through world model                    │   │  │
│   │  │  - Select by reward + uncertainty                  │   │  │
│   │  └────────────────────────────────────────────────────┘   │  │
│   │                          │                                 │  │
│   │                          ▼                                 │  │
│   │  Collect new episodes with CEM planner                    │  │
│   │                          │                                 │  │
│   │                          ▼                                 │  │
│   │  Filter: keep top 50% by reward                           │  │
│   │                          │                                 │  │
│   │                          ▼                                 │  │
│   │  Retrain WAM on merged data                               │  │
│   │                                                            │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Evaluation Results

### PickCube-v1 Sweep (5 rounds, 20 episodes/round)

| Round | Source | Training Loss | Success Rate | Reward |
|-------|--------|---------------|--------------|--------|
| 0 | Demo bootstrap | 0.06 | 5% | 3.87 |
| 1 | WM-guided CEM | 0.08 | 5% | 2.79 |
| 2 | WM-guided CEM | 0.10 | 0% | 3.10 |
| 3 | WM-guided CEM | 0.09 | 5% | 2.37 |
| 4 | WM-guided CEM | 0.11 | 5% | 1.97 |

### Key Findings

1. ✅ **Demo bootstrap works**: 5% success rate (vs 0% random)
2. ✅ **WAM learns demo distribution**: Loss 0.06 (6x better than random)
3. ❌ **WM-guided doesn't improve**: Rewards decrease across rounds
4. ❌ **No learning progression**: Success rate flat at 5%

## Gap Analysis: Our Implementation vs SOTA

### What We're Doing Right

| Aspect | Status | Notes |
|--------|--------|-------|
| Joint state+action prediction | ✅ | Correct per WAM taxonomy |
| Diffusion-based denoising | ✅ | Matches DreamZero approach |
| Parallel heads sharing backbone | ✅ | Efficient architecture |
| CEM planning through world model | ✅ | Standard planning approach |

### What's Limiting Performance

| Aspect | Our Scale | SOTA Scale | Gap |
|--------|-----------|------------|-----|
| **Model size** | 14.4M | 14B (DreamZero) | 1000x |
| **Training data** | 20 demos | Millions of frames | 10,000x |
| **Input modality** | State vectors | RGB images | Different |
| **Generalization** | Task-specific | Cross-task | Limited |
| **Planning horizon** | 8 steps | 50+ steps | 6x |
| **Reward signal** | Uncertainty only | Learned reward | Missing |

### Recommendations for Improvement

1. **Scale up**: Increase model to 100M+ parameters
2. **Visual input**: Switch from state vectors to RGB images
3. **More data**: Collect 1000+ episodes per task
4. **Reward learning**: Train a reward model for planning
5. **Curriculum learning**: Start easy, increase difficulty
6. **Multi-task**: Train across multiple ManiSkill tasks

## References

1. Wang et al. (2026). "World Action Models: The Next Frontier in Embodied AI." arXiv:2605.12090
2. Ye et al. (2026). "World Action Models are Zero-shot Policies." arXiv:2602.15922 (DreamZero)
3. Zhang et al. (2026). "From World Models to World Action Models: A Concise Tutorial." arXiv:2607.00836
4. Aljalbout et al. (2026). "The Reality Gap in Robotics." arXiv:2510.20808
5. Tao et al. (2025). "ManiSkill3: GPU Parallelized Robotics Simulation." arXiv:2410.00425
