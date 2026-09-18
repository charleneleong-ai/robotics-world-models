# Video-backbone representations for LIBERO

Asks whether a pretrained visual or video backbone gives a better representation for robot
behaviour cloning than the proprioceptive state vector every earlier experiment in this
repository used. The LIBERO HDF5 files already carry two 128x128 RGB streams next to the
state; these scripts use them.

Nothing is generated. Each backbone is run forward once and its hidden states are read off
as features -- the "representation-only" branch of the world-action-model literature, where
the video objective matters at training time and test-time imagination is dropped.

## Backbones audited

| Name | Pretraining objective | Feature dim |
|---|---|---|
| `vae` | Wan2.2 video VAE -- reconstruction | 3072 |
| `dit_ctx{N}` | Wan2.2 video diffusion transformer -- denoising | 3072 |
| `siglip2` | SigLIP2-so400m -- image-text contrastive | 1152 |
| `openvla` | OpenVLA-7B vision backbone -- robot action prediction, in-domain | 2176 |

## Results

LIBERO-Spatial, 10 tasks x 3 seeds, held out by whole demonstration (2 of 12 per task),
identical MLP head on every representation, paired at the task level. `_ctx8` concatenates
frames t-7 and t, matching the two latent frames the DiT receives at an 8-frame window.

| Representation | Frames | Dims | Held-out MSE | vs state |
|---|---|---|---|---|
| state (proprioception) | -- | 21 | **0.0456** | -- |
| vae | 1 | 3072 | 0.1018 | +123% |
| dit_ctx1 | 1 | 3072 | 0.0868 | +90% |
| siglip2 | 1 | 1152 | 0.0826 | +81% |
| openvla | 1 | 2176 | 0.0771 | +69% |
| siglip2_ctx8 | 2 | 2304 | 0.0500 | +9.7% (p=0.37) |
| dit_ctx8 | 2 | 3072 | 0.0448 | -1.7% (p=0.87) |
| **openvla_ctx8** | 2 | 4352 | **0.0442** | -3.1% (p=0.75) |
| dit_ctx16 | 4 | 3072 | 0.0469 | +2.8% (p=0.83) |

The `state` baseline reproduces to four decimals across separate runs (0.0456), so figures
from different runs are comparable.

**Temporal context is the whole story, and it is not about video models.** Adding one extra
frame is worth 39-48% on every backbone tested: Wan DiT -48.4% (p=0.0002, 9/10 tasks),
OpenVLA -42.7% (p=0.0003, 9/10), SigLIP2 -39.5% (p=0.0002, 10/10). A generic image-text
encoder with two frames lands in the same place as a 5B video world model with two latent
frames. Whatever the video pretraining adds, it is not what closes the gap.

**In-domain robot pretraining buys about 10%.** OpenVLA, trained on robot demonstrations to
predict actions, beats SigLIP2 by 6.7% at one frame (p=0.28) and 11.6% at two (p=0.022,
9/10 tasks). Real but modest next to the 40% that one extra frame is worth.

**At best these tie proprioception rather than beating it.** The best representation
(`openvla_ctx8`) is 3.1% better than 21 numbers at p=0.75, and adding state on top does not
help. Matching privileged joint angles and end-effector pose from 128x128 pixels is a real
result, but it is parity bought with a large forward pass.

**Context saturates immediately.** 16 frames is indistinguishable from 8 (p=0.46). The VAE
compresses time 4x, so 8 frames is 2 latent frames and 16 is 4; the gain arrives exactly
where the model first receives more than one latent frame, and nothing after.

## State decodability (`state_decode.py`)

Linear ridge probes from each representation to the 21-dim state, penalty chosen on an
inner validation split. R^2, 1.0 being perfect recovery:

| Representation | ee_ori | ee_pos | ee_states | gripper | joints | all |
|---|---|---|---|---|---|---|
| vae | -0.20 | 0.96 | 0.19 | 0.69 | 0.82 | 0.670 |
| dit_ctx1 | 0.14 | 0.96 | 0.42 | 0.73 | 0.87 | 0.761 |
| dit_ctx4 | 0.11 | 0.94 | 0.39 | 0.71 | 0.85 | 0.743 |
| dit_ctx8 | 0.08 | 0.95 | 0.38 | 0.83 | 0.85 | 0.737 |
| dit_ctx16 | 0.09 | 0.95 | 0.38 | 0.87 | 0.83 | 0.723 |

This refutes the obvious explanation for the behaviour-cloning gap. Every representation,
the VAE included, recovers end-effector position at R^2 ~ 0.95, so "the video features
cannot see where the gripper is" is false. Decodability also runs *backwards* to cloning:
`dit_ctx1` recovers state best (0.761) and clones worst of the DiT variants. The one column
that tracks context is the gripper (0.73 -> 0.87), which is a temporally-defined event
rather than a static pose. Context is not improving the estimate of where the arm is; it is
improving the estimate of what the arm is doing.

## Environment

`diffusers` needs a newer `huggingface-hub` than the system `transformers` on this box can
take, so it lives in its own virtualenv. Do not `pip install` it system-wide -- that breaks
`transformers` for openvla and everything else here. OpenVLA pins `transformers==4.40.1`
and so must run under the *system* python, not the venv.

```bash
python3 -m venv --system-site-packages ~/wan_venv
~/wan_venv/bin/pip install "diffusers>=0.35" "huggingface-hub>=1.0" "transformers>=4.45" peft wandb
python3 -m huggingface_hub.commands.huggingface_cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --include "vae/*" "transformer/*" "model_index.json" "scheduler/*" --local-dir ~/wan22_ti2v_5b
```

The Wan text encoder is deliberately not downloaded (11 GB); features use null text
conditioning.

**Host durability.** This volume runs near capacity and is shared. During this work the
LIBERO suite directories, every feature cache under `~/wan_latents/`, and the whole 22 GB
Wan checkout were evicted between sessions. LIBERO is restorable by symlinking the copy in
the HuggingFace dataset cache; the backbone needs re-downloading. Treat anything large on
this host as temporary and keep results in W&B, not on disk.

## Running

```bash
V=~/wan_venv/bin/python
$V experiments/wan_extract.py spatial agentview_rgb        # Wan VAE latents
for c in 1 4 8 16; do $V experiments/dit_extract.py spatial $c; done
$V experiments/backbone_extract.py siglip2
python3  experiments/backbone_extract.py openvla           # system python
$V experiments/wan_compare.py                              # skips absent caches
$V experiments/state_decode.py
$V experiments/continual_probe.py --n-orderings 6 --n-seeds 3   # sequential tasks
$V experiments/continual_probe.py --n-orderings 6 --n-seeds 3 --pca-dim 21 \
   --out ~/wan_latents/continual_results_pca21.json                 # width control
for L in 0 10000 100000 1000000 10000000 100000000; do             # EWC lambda sweep
  $V experiments/continual_probe.py --n-orderings 6 --n-seeds 3 --ewc-lambda $L \
     --out ~/wan_latents/continual_ewc_$L.json
done
$V experiments/sweep_table.py --pattern 'continual_ewc_*.json'
for M in 0 20 100 500 2000; do                                     # replay memory sweep
  $V experiments/continual_probe.py --n-orderings 6 --n-seeds 3 --replay-size $M \
     --out ~/wan_latents/continual_er_$M.json
done
$V experiments/sweep_table.py --pattern 'continual_er_*.json' --axis memory
for M in 100 2000; do                                              # EWC on top of replay
  for L in 10000 100000 1000000; do
    $V experiments/continual_probe.py --n-orderings 6 --n-seeds 3 --replay-size $M --ewc-lambda $L \
       --out ~/wan_latents/continual_er${M}_ewc_$L.json
  done
done
$V experiments/combo_table.py
```

Caches land in `~/wan_latents/` (32-113 MB per representation, derived, not tracked). All
runs log to the `video-wam` W&B project.

## Caveats

Ten tasks at 3 seeds per cell, no multiple-comparison correction across representations; on
this project's own track record that is the regime where apparent per-configuration winners
turn out to be noise. For the DiT, block 15 of 30, a single noise level (t=100) and
mean-pooling over tokens were each chosen once and never varied. The `_ctx8` variants for
the image encoders use two frames while `dit_ctx8` sees two *latent* frames summarising
eight, so those are matched on latent frames rather than on raw frames seen. The sequential-task results below
use only the four image-encoder caches that survived eviction, so they say nothing about the
Wan representations.

## Sequential tasks: does a better representation forget less?

No. On a single task `openvla_ctx8` matched proprioceptive state (-3.1%, p=0.75). Trained
sequentially over the same ten tasks it is 24% worse at the end and forgets significantly
more. Every learned representation tested loses to 21 numbers of proprioception.

Six random orderings x 3 seeds, seeds averaged inside an ordering before testing, arms paired
on identical orderings, held out by whole demonstration. Metrics are absolute: final error is
the mean over all ten tasks after the last one is trained; absolute forgetting is a task's
error at the end minus its error immediately after it was trained. The first-task-normalised
forgetting *rate* is deliberately avoided -- the 40.8% reduction reported in `wm-pai` and since
retracted was an artifact of that denominator.

| Representation | Frames | Dims | Final error | vs state | Abs. forgetting | vs state |
|---|---|---|---|---|---|---|
| state | -- | 21 | 0.0910 | -- | +0.0599 | -- |
| siglip2 | 1 | 1152 | 0.1793 | +96.9% | +0.1103 | p<0.001 |
| openvla | 1 | 2176 | 0.1529 | +67.9% | +0.0880 | p=0.002 |
| siglip2_ctx8 | 2 | 2304 | 0.1375 | +51.1% | +0.0979 | p<0.001 |
| openvla_ctx8 | 2 | 4352 | 0.1129 | +24.0% | +0.0782 | p=0.014 |

Temporal context helps here as it did on the single task, and it helps retention specifically,
not just fit: absolute forgetting falls 11.3% for SigLIP2 (p=0.036, 5/6 orderings) and 11.1%
for OpenVLA (p=0.009, 6/6). It is not enough to close the gap -- the best representation still
forgets 23.5% more than state (p=0.015, 6/6).

The per-step curves in W&B show where the damage happens. Mean error over tasks seen so far
jumps at the first switch (state 0.045 to 0.079, `openvla_ctx8` 0.046 to 0.109) and is then
flat for the remaining eight tasks. The representations differ in how much they lose on the
first interference event, not in how they accumulate damage afterwards.

### Width is not the explanation

The obvious objection is that state is 21 numbers and the backbones are thousands, so the
comparison rewards a narrow input. Projecting each representation to 21 dimensions with PCA
fitted on training frames only makes every arm **worse**, not better:

| Representation | Final error | PCA-21 final | Abs. forgetting | PCA-21 |
|---|---|---|---|---|
| siglip2 | 0.1793 | 0.2153 | +0.1103 | +0.1194 |
| siglip2_ctx8 | 0.1375 | 0.1756 | +0.0979 | +0.1074 |
| openvla | 0.1529 | 0.1857 | +0.0880 | +0.1086 |
| openvla_ctx8 | 0.1129 | 0.1469 | +0.0782 | +0.0919 |

Width-matching costs `openvla_ctx8` 23.1% in final error (p=0.008, 6/6). Whatever proprioception
has that these representations lack, it is not low dimensionality. The plausible remaining
account is that the state vector is already the causally relevant variable, so tasks share a
coordinate system and updates transfer; the visual codes are task-shaped and overwrite.


### Does EWC close the gap?

No. Consolidation helps every arm by about the same amount, so the ordering survives.

Diagonal-Fisher EWC, the textbook multi-task form with one curvature estimate and one anchor
per completed task, swept over lambda and applied to *every* arm including proprioception.
Applying it only to the representations would have answered a different question.

Final error by lambda (lower is better; the lambda=0 row reproduces the unregularised numbers
above to four decimals, which is the check that the penalty is inert when switched off):

| lambda | state | siglip2 | siglip2_ctx8 | openvla | openvla_ctx8 |
|---|---|---|---|---|---|
| 0 | 0.0910 | 0.1793 | 0.1375 | 0.1529 | 0.1129 |
| 1e4 | 0.0722 | 0.1478 | 0.1313 | 0.1338 | 0.1085 |
| 1e5 | **0.0691** | **0.1347** | **0.0990** | **0.1107** | 0.0994 |
| 1e6 | 0.0822 | 0.1475 | 0.1103 | 0.1159 | **0.0822** |
| 1e7 | 0.1021 | 0.1990 | 0.1591 | 0.1594 | 0.1143 |
| 1e8 | 0.1101 | 0.2396 | 0.1961 | 0.1887 | 0.1538 |

At its own best lambda every representation improves by 24-28%, and the gap to state barely
moves: `openvla_ctx8` goes from +24.0% to +19.0% (0.0822 vs 0.0691, p=0.0007, 0 of 6 orderings
favouring the representation). EWC is a large effect that is close to representation-agnostic.

| Representation | best lambda | lambda=0 | tuned | change | vs tuned state |
|---|---|---|---|---|---|
| state | 1e5 | 0.0910 | 0.0691 | -24.1% | -- |
| siglip2 | 1e5 | 0.1793 | 0.1347 | -24.9% | +94.9% |
| openvla | 1e5 | 0.1529 | 0.1107 | -27.6% | +60.3% |
| siglip2_ctx8 | 1e5 | 0.1375 | 0.0990 | -28.0% | +43.3% |
| openvla_ctx8 | 1e6 | 0.1129 | 0.0822 | -27.2% | +19.0% |

**The tie at lambda=1e6 is a trap.** Read that row alone and `openvla_ctx8` exactly matches state
(0.0822 vs 0.0822, p=0.995). It matches only because state is past its own optimum there: state's
plasticity, the mean error on each task the moment it finished training, degrades from 0.0471 at
its best lambda to 0.0676 at 1e6, while `openvla_ctx8` gives up much less. Comparing two arms at
a single shared lambda where one is over-regularised manufactures parity. This is the same shape
of error as the retracted forgetting-rate claim -- a number that looks like a win because the
comparison is not matched -- so the tuned-per-arm row above is the one to quote.

Forgetting keeps falling monotonically in lambda for every arm while final error turns around,
which is the ordinary stability-plasticity trade-off and the reason absolute forgetting alone is
not a sufficient metric here.

The Fisher is estimated from batch-level squared gradients rather than per-sample ones, which
is the cheap standard approximation; gradients partly cancel within a batch, so the estimate
is coarser than a per-sample Fisher and the useful lambda range is correspondingly shifted.
It is the same estimator for every arm, so the comparison holds.

Lambda is chosen post hoc on the same six orderings the significance tests run over, so those
p-values are optimistically biased and uncorrected for selection across the grid. They support
`the tuned arms differ`, not `this lambda is the right one`. The headline does not lean on
them: the gap is significant at every lambda in the sweep except the single crossover point.



### Does replay close the gap?

Most of it, and the part it closes is the part this branch was asking about. Replay removes the
forgetting difference entirely; what survives is the single-task fit difference, which is not a
continual-learning problem at all.

Memory balanced across tasks seen so far, each task keeping a fixed permutation so a shrinking
quota evicts rather than resamples, with the new task's loss and an equally weighted loss on a
batch drawn from memory. Applied to every arm including proprioception, same as the EWC sweep.
Budget is counted in frames: 2000 frames is about a fifth of all training data.

| memory | state | siglip2 | siglip2_ctx8 | openvla | openvla_ctx8 |
|---|---|---|---|---|---|
| 0 | 0.0910 | 0.1793 | 0.1375 | 0.1529 | 0.1129 |
| 20 | 0.0689 | 0.1467 | 0.1091 | 0.1244 | 0.0868 |
| 100 | 0.0579 | 0.1177 | 0.0811 | 0.0992 | 0.0635 |
| 500 | 0.0454 | 0.0950 | 0.0600 | 0.0818 | 0.0496 |
| 2000 | **0.0402** | **0.0860** | **0.0506** | **0.0754** | **0.0445** |

**A hundred stored frames beat the best-tuned EWC on every arm**, by 10-23% with 6 of 6 orderings
agreeing: state -16.1% (p=0.0022), `siglip2_ctx8` -18.1% (p<0.001), `openvla_ctx8` -22.7%
(p=0.0006). That is about 1% of the training data against a method with a tuned hyperparameter,
and it reproduces on these representations what the earlier suites already found -- a small memory
with random replay beats the reliability signals.

**Replay costs no plasticity.** The mean error on each task the moment it finished training is flat
across the whole memory sweep, within 0.004 of its unregularised value for every arm. EWC bought
its retention by giving up plasticity, degrading state from 0.0471 to 0.0676 at the setting where
it appeared to tie. Replay buys retention with storage instead, and storage turns out to be much
the better currency here.

**The forgetting gap closes; the fit gap does not.** At memory 2000 the residual gap from
`openvla_ctx8` to state is +0.0042, and it decomposes almost entirely into the single-task term:

| component | gap | p |
|---|---|---|
| final error | +0.0042 | 0.0012 |
| plasticity (single-task fit) | +0.0037 | 0.0039 |
| absolute forgetting | +0.0007 | 0.379 |

From memory 500 upward the forgetting difference between the best representation and
proprioception is no longer distinguishable (p=0.42 at 500, p=0.38 at 2000, and the orderings
split rather than lining up). The representations are not worse at retaining; they are worse at
fitting, and that difference was already visible in the single-task probe. Given enough replay,
the sequential setting stops adding a penalty of its own.

At matched tuning the end-to-end gap narrows from +24.0% to +10.6% (0.0445 vs 0.0402, p=0.0012).
The same post-hoc-selection caveat as the EWC sweep applies, and here the selection is trivial
since every arm's best memory is the largest one swept.

Aggregate either sweep with `sweep_table.py`, which reads the swept value out of each filename.

### EWC on top of replay

The two are substitutes, not complements. At a memory that works, adding EWC only makes things
worse, and where EWC does help, buying more memory helps more.

EWC strength swept at a scarce memory (100 frames) and at the best one (2000), each cell tested
against the same memory with no EWC, paired on the shared orderings.

Final error at memory 2000, where replay is already doing its job:

| representation | replay alone | + EWC 1e4 | + EWC 1e5 | + EWC 1e6 |
|---|---|---|---|---|
| state | **0.0402** | 0.0452 (+12.3%) | 0.0549 (+36.4%) | 0.0689 (+71.3%) |
| siglip2_ctx8 | **0.0506** | 0.0513 (+1.4%) | 0.0586 (+15.8%) | 0.0805 (+59.0%) |
| openvla_ctx8 | **0.0445** | 0.0461 (+3.7%) | 0.0467 (+5.0%) | 0.0563 (+26.8%) |

Every arm is worse at every strength, and where it is significant no ordering favours the
addition. At memory 100 the picture inverts for the weaker arms -- `siglip2_ctx8` -10.3%
(p<0.001, 6/6), `openvla` -10.0% (p<0.001, 6/6), state -6.2% (p=0.040, 5/6) -- but the best
representation still gains nothing (`openvla_ctx8` -2.1%, p=0.22).

**The mechanism is visible in the two component metrics.** EWC reduces forgetting monotonically
in its strength, for every arm, at both memory sizes: at memory 100 it takes state's absolute
forgetting from 0.0193 down to 0.0026. It also costs plasticity monotonically, taking state's
from 0.0406 to 0.0693 over the same range. When memory is scarce, forgetting is still the
binding constraint and the trade can pay. When memory is ample, forgetting is already near zero
-- 0.0020 for state at memory 2000 -- so there is nothing left to buy and all the penalty does
is spend plasticity.

**More memory beats any amount of EWC.** Four hundred extra frames beat the best combination on
a small memory, 6 of 6 orderings, for both the reference and the best representation:

| | memory 100 + EWC | memory 500 alone | |
|---|---|---|---|
| state | 0.0543 | **0.0454** | -16.4%, p<0.001, 6/6 |
| openvla_ctx8 | 0.0644 | **0.0496** | -23.0%, p=0.0001, 6/6 |

So the three interventions order cleanly on this benchmark, and the ordering is the same one the
earlier suites in this repository reported: replay first by a wide margin, EWC a distant second
that only matters when replay is starved, and the two together never better than replay alone
with the memory it wants.

The sweep fixes memory and varies strength rather than covering the full grid, so an interaction
at some untested pair is not ruled out. The two memory sizes bracket the regime of interest, and
the trend within each is monotonic and consistent across all five arms. The grid is 90 uncorrected
two-sided tests over six orderings, so a handful below 0.05 are expected under the null; what
carries the conclusion is the monotone trend and the unanimous ordering counts, not any one cell.
Percentage changes in forgetting at memory 2000 are quoted as absolutes above because the
baselines there are near zero and ratios on them are meaningless.

Aggregate with `combo_table.py`.

Run with `continual_probe.py --n-orderings 6 --n-seeds 3` (add `--pca-dim 21` for the control).
Both runs are in the `video-wam` W&B project with per-step retention curves, a summary table,
and a `continual-forgetting` results artifact holding per-ordering values.

Six orderings is little power, and the p-values are uncorrected across the nine arms tested
against the same reference. This ran after the Wan caches and the 22 GB backbone were evicted
from the box, so the VAE and DiT rows are missing; the script picks them up automatically once
`~/wan_latents/spatial_dit_ctx8` and friends exist again.
