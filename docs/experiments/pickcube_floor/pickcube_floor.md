# `pickcube_floor` — <one-line hypothesis here>

**Schedule:** [`configs/schedules/pickcube_floor.yaml`](../../../configs/schedules/pickcube_floor.yaml)
**Config:** `ppo`
**Chassis:** `<model_name>` · LoRA r=<rank> · max_seq=<n> · num_generations=<n>
**Iterations:** 0 iters · 5 rows logged
**Started:** <UTC timestamp> · **Finished:** <UTC timestamp> (<duration>)

## Hypothesis

<What mechanism is this sweep testing? What outcome would falsify the hypothesis?>

## Schedule

```yaml
task: pickcube_floor
env_id: PickCube-v1
obs: state
description: 'Establish a reproducible floor on PickCube-v1 and a like-for-like world-model-vs-model-free
  comparison on identical task/seeds/budget. Headline contribution is the world-model-vs-classical
  crossover characterization, with the contact-rich
  result on PegInsertionSide. Runs trained on a datacenter A100 80GB, logged to W&B
  project wm-manip.

  '
configs:
  ppo:
    launcher: benchmarks/ManiSkill/examples/baselines/ppo/ppo.py
    overrides:
      num_envs: 1024
      total_timesteps: 10000000
      update_epochs: 8
      num_minibatches: 32
    seeds:
    - 1
    - 2
    - 3
    - 4
    - 5
  tdmpc2:
    launcher: benchmarks/ManiSkill/examples/baselines/tdmpc2/train.py
    overrides:
      model_size: 5
      steps: 1000000
      num_envs: 32
      control_mode: pd_ee_delta_pos
      buffer_size: 1000000
    seeds:
    - 1
    - 2
    - 3
    - 4
    - 5
```

`common_overrides`: `(none)`

## Pre-launch comparisons

<Reference rows from prior sweeps that this one is anchored against. Pull from
`docs/ceiling-diagnosis-*.md` or the `results.jsonl` directly.>

| anchor | mean_total | f1 | no_halluc |
|---|---|---|---|
| <prior exp> | — | — | — |

## Results

| iter | exp | steps | runtime | mean_total | f1 | no_halluc | well_formed | pass_all |
|---|---|---|---|---|---|---|---|---|
| 1/0 | E0 | 9984000 | 28.7m | — | — | — | — | — |
| 2/0 | E1 | 9984000 | 33.0m | — | — | — | — | — |
| 3/0 | E2 | 9984000 | 31.2m | — | — | — | — | — |
| 4/0 | E3 | 9984000 | 28.8m | — | — | — | — | — |
| 5/0 | E4 | 9984000 | 32.1m | — | — | — | — | — |

## Verdict

<Did the hypothesis hold? Reference specific iter numbers. Use ✓/✗ to mark
sub-claims so future readers can scan.>

## Next move

<Pointer to the next sweep yaml + writeup, or a "ceiling reached, pivot to X"
note. Cross-link to the diagnosis doc for the cross-sweep narrative.>
