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
```

Caches land in `~/wan_latents/` (32-113 MB per representation, derived, not tracked). All
runs log to the `video-wam` W&B project.

## Caveats

Ten tasks at 3 seeds per cell, no multiple-comparison correction across representations; on
this project's own track record that is the regime where apparent per-configuration winners
turn out to be noise. For the DiT, block 15 of 30, a single noise level (t=100) and
mean-pooling over tokens were each chosen once and never varied. The `_ctx8` variants for
the image encoders use two frames while `dit_ctx8` sees two *latent* frames summarising
eight, so those are matched on latent frames rather than on raw frames seen. Single-task
imitation only -- whether a video representation *forgets* less across a task sequence is
untested, and is the question that would connect this to the continual-learning line.
