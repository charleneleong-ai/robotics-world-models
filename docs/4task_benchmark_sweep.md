# 4-Task Benchmark & Architecture Sweep

## Goal

Benchmark our DiffusionWAM against SOTA on ManiSkill3, and explore alternative WAM architectures (Cascaded, DreamZero-inspired, GATO-inspired) to identify performance improvements.

## ManiSkill3 Benchmark Context

From SOTA2 leaderboard (Dec 2025 - Jul 2026):

| Task | SOTA (IPI/OpenVLA-7B) | Baseline (BC) | Our Current |
|------|----------------------|---------------|-------------|
| Push Cube | 97.3% | 26% | — |
| Stack Cube | 94.3% | 0% | — |
| Pull Cube | 98.5% | — | — |
| Lift Peg Upright | 95.5% | — | 5% (PickCube) |
| Pick Cube | — | — | 5% |

### Key Insights

1. **SOTA is 80-96%**: Using VLA models (OpenVLA-7B, Pi0.5) with RL fine-tuning
2. **BC baseline is 0-26%**: Pure imitation learning struggles
3. **RL fine-tuning is critical**: StARe framework achieves 93-96% with reinforcement
4. **Vision-based**: All SOTA methods use RGB images, not state vectors

## Tasks to Benchmark

### 1. PickCube-v1 (Current)
- **Goal**: Pick up cube and hold at goal location
- **Action space**: pd_joint_delta_pos (8-dim)
- **Observation**: 42-dim state vector
- **Difficulty**: Easy
- **Demo data**: Available at `/tmp/mp_demos/PickCube-v1/`

### 2. PushCube-v1
- **Goal**: Push cube to goal position
- **Action space**: pd_joint_delta_pos (8-dim)
- **Observation**: 42-dim state vector
- **Difficulty**: Easy
- **Demo data**: Available at `/tmp/mp_demos/PushCube-v1/`

### 3. PegInsertionSide-v1
- **Goal**: Insert peg into hole
- **Action space**: pd_joint_delta_pos (8-dim)
- **Observation**: 42-dim state vector
- **Difficulty**: Medium
- **Demo data**: Available at `/tmp/mp_demos/PegInsertionSide-v1/`

### 4. PlugCharger-v1
- **Goal**: Plug charger into socket
- **Action space**: pd_joint_delta_pos (8-dim)
- **Observation**: 42-dim state vector
- **Difficulty**: Medium-Hard
- **Demo data**: Available at `/tmp/mp_demos/PlugCharger-v1/`

## Architecture Comparison

### Current: Joint DiffusionWAM (14.4M)

```
Input: [obs, noise] → Shared Backbone → Parallel Heads → (next_state, action)
```

- ✅ Matches WAM taxonomy (Joint WAM)
- ✅ Simple implementation
- ❌ 1000x smaller than DreamZero (14M vs 14B)
- ❌ State vectors only (no vision)

### Option A: Cascaded WAM

```
Input: obs → World Model → predicted_state → Action Decoder → action
```

**Pros:**
- World model can be trained independently
- Action decoder can be simpler (no diffusion needed)
- Matches UniPi/SayCan architecture

**Cons:**
- Error propagation (world model errors compound)
- Less efficient (two separate models)

### Option B: DreamZero-Inspired (14B scale)

```
Input: RGB frames → Video Diffusion → (next_frame, action)
```

**Pros:**
- State-of-the-art architecture
- Zero-shot transfer potential
- Visual reasoning

**Cons:**
- Requires 14B params (needs multi-GPU)
- Requires RGB observations (need to modify ManiSkill)
- Training data intensive (millions of frames)

### Option C: GATO-Inspired (Generalist)

```
Input: [obs + action history] → Transformer → next_obs + reward
```

**Pros:**
- Multi-task capable
- Discrete tokenization (can use language)
- Proven architecture

**Cons:**
- Requires discrete action space
- Less efficient for continuous control

### Option D: Hybrid Approach (Recommended)

```
Phase 1: Train world model on state vectors (current)
Phase 2: Add visual encoder for RGB input
Phase 3: Scale up to 100M+ params
Phase 4: Multi-task training
```

## Implementation Plan

### Phase 1: 4-Task Baseline (Current Architecture)

**Goal:** Establish baseline performance across all 4 tasks

**Steps:**
1. Run 5-round sweep on each task with demo bootstrap
2. Record success rates, rewards, training loss
3. Compare against BC baseline (0-26%)

**Duration:** ~2 hours on A100

### Phase 2: Architecture Exploration

**Goal:** Test Cascaded WAM vs Joint WAM

**Steps:**
1. Implement CascadedWAM (separate world model + action decoder)
2. Train on PickCube-v1 with same data
3. Compare training stability and success rate

**Duration:** ~4 hours

### Phase 3: Scale Up (If Phase 2 shows improvement)

**Goal:** Increase model capacity

**Steps:**
1. Scale DiffusionWAM from 14M to 100M params
2. Add visual encoder (ResNet/ViT) for RGB input
3. Test on PickCube-v1

**Duration:** ~8 hours

### Phase 4: Multi-Task Training (If Phase 3 shows improvement)

**Goal:** Train single model on all 4 tasks

**Steps:**
1. Combine demo data from all 4 tasks
2. Train single DiffusionWAM with task embedding
3. Evaluate zero-shot transfer

**Duration:** ~12 hours

## Success Metrics

| Metric | Target | SOTA Comparison |
|--------|--------|-----------------|
| PickCube success | >20% | 95.5% (Lift Peg) |
| PushCube success | >30% | 97.3% (Push Cube) |
| Training loss | <0.10 | — |
| Improvement over BC | >2x | BC: 0-26% |

## A100 Resources

- **GPU**: NVIDIA A100 80GB PCIe
- **Memory**: 80GB HBM2e
- **Compute**: 312 TFLOPS (FP32)
- **Current usage**: ~4GB (14M model)
- **Available**: ~76GB for scaling

## Expected Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | 2 hours | 4-task baseline results |
| Phase 2 | 4 hours | Cascaded WAM comparison |
| Phase 3 | 8 hours | Scaled model results |
| Phase 4 | 12 hours | Multi-task results |
| **Total** | **26 hours** | **Architecture recommendation** |

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| A100 unavailable | High | Check before starting each phase |
| Demo data missing | Medium | Download from ManiSkill first |
| Model too large | Low | Start with 100M, not 14B |
| Training diverges | Medium | Gradient clipping, lower LR |
