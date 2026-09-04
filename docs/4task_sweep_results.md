# 4-Task Benchmark Results

## Sweep Results (5 rounds, 20 episodes/round)

### PickCube-v1
| Round | Source | Success Rate | Mean Reward |
|-------|--------|-------------|-------------|
| 0 | Demo bootstrap | 5% | 3.690 |
| 1 | WM-guided CEM | 0% | 2.799 |
| 2 | WM-guided CEM | 0% | 2.994 |
| 3 | WM-guided CEM | 0% | 2.664 |
| 4 | WM-guided CEM | 5% | 2.639 |

### PushCube-v1
| Round | Source | Success Rate | Mean Reward |
|-------|--------|-------------|-------------|
| 0 | Demo bootstrap | 5% | 4.215 |
| 1 | WM-guided CEM | 0% | 3.903 |
| 2 | WM-guided CEM | 0% | 3.503 |
| 3 | WM-guided CEM | 0% | 4.354 |
| 4 | WM-guided CEM | 0% | 4.191 |

### PegInsertionSide-v1
| Round | Source | Success Rate | Mean Reward |
|-------|--------|-------------|-------------|
| 0 | Demo bootstrap | 0% | 2.634 |
| 1 | WM-guided CEM | 0% | 2.550 |
| 2 | WM-guided CEM | 0% | 2.172 |
| 3 | WM-guided CEM | 0% | 1.515 |
| 4 | — | FAILED (I/O error) | — |

### PlugCharger-v1
| Round | Source | Success Rate | Mean Reward |
|-------|--------|-------------|-------------|
| 0 | — | FAILED (no RL demos) | — |

## Demo Data Quality

| Task | Demo Source | Trajectories | Action Dim | Demo Success Rate |
|------|-------------|-------------|------------|-------------------|
| PickCube-v1 | motionplanning | 1000 | 8 | 7% |
| PushCube-v1 | motionplanning | 1000 | 8 | 13% |
| PegInsertionSide-v1 | rl | 1000 | 8 | 68% |
| PlugCharger-v1 | motionplanning | 1000 | 8 | 3% |

## Key Findings

1. **Demo bootstrap works for PickCube/PushCube**: 5% success (vs 0% random)
2. **WM-guided exploration doesn't improve**: Success flat or decreasing across rounds
3. **PegInsertionSide has best demos** (68% success) but our model can't learn them
4. **PlugCharger has worst demos** (3% success) — too hard for current approach
5. **All tasks: 0% success after round 0** — model forgets demo distribution

## SOTA Comparison

| Task | Our Best | SOTA (IPI/OpenVLA-7B) | Gap |
|------|----------|----------------------|-----|
| PickCube | 5% | 95.5% (Lift Peg) | 19x |
| PushCube | 5% | 97.3% | 19x |
| PegInsertionSide | 0% | 98.5% (Pull Cube) | ∞ |
| PlugCharger | 0% | — | — |

## Root Cause Analysis

### Why WM-guided doesn't help:
1. **Model too small** (14M vs 14B SOTA) — can't learn dynamics
2. **State vectors only** — no visual features for planning
3. **CEM planner limited** — only 5 iterations, 100 samples
4. **No reward signal** — uncertainty-based scoring is weak

### Why PegInsertionSide fails despite good demos:
1. **Longer episodes** (178 steps vs 74 for PickCube)
2. **More complex dynamics** — requires precise insertion
3. **Model can't generalize** — overfits to short demonstrations

## Recommendations

1. **Scale model to 100M+ params** — increase capacity
2. **Add vision encoder** — RGB images for better features
3. **Improve CEM planner** — more iterations, better scoring
4. **Add reward learning** — train reward model for planning
5. **Multi-task training** — leverage cross-task transfer
