# Video-backbone representations for LIBERO

Probes whether a pretrained video world model gives a better representation for robot
behaviour cloning than the proprioceptive state vector every earlier experiment in this
repository used. The LIBERO HDF5 files already carry two 128x128 RGB streams next to the
state; these scripts use them.

Backbone is Wan2.2-TI2V-5B (video VAE + diffusion transformer). Nothing is generated: the
transformer is run forward once per window and its mid-block hidden states are read off as
features. This is the "representation-only" branch of the world-action-model literature,
where the video objective matters at training time and test-time imagination is dropped.

## Environment

`diffusers` needs a newer `huggingface-hub` than the system `transformers` on this box can
take, so it lives in its own virtualenv. Do not `pip install` it system-wide -- that breaks
`transformers` for openvla and everything else here.

```bash
python3 -m venv --system-site-packages ~/wan_venv
~/wan_venv/bin/pip install "diffusers>=0.35" "huggingface-hub>=1.0" "transformers>=4.45" peft wandb
python3 -m huggingface_hub.commands.huggingface_cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --include "vae/*" "transformer/*" "model_index.json" "scheduler/*" --local-dir ~/wan22_ti2v_5b
```

The text encoder is deliberately not downloaded (11 GB); features use null text conditioning.

## Running

```bash
~/wan_venv/bin/python experiments/wan_extract.py spatial agentview_rgb   # VAE latents
for c in 1 4 8 16; do ~/wan_venv/bin/python experiments/dit_extract.py spatial $c; done
~/wan_venv/bin/python experiments/wan_compare.py                          # paired comparison
```

Caches land in `~/wan_latents/` (~80 MB per representation, derived, not tracked). All runs
log to the `video-wam` W&B project.

## Results (LIBERO-Spatial, 10 tasks x 3 seeds, held out by whole demonstration)

| Representation | Dims | Held-out MSE | vs state |
|---|---|---|---|
| state (proprioception) | 21 | 0.0456 | -- |
| Wan VAE latent | 3072 | 0.1018 | +123% |
| DiT, 1 frame | 3072 | 0.0868 | +90% (p<0.0001, 0/10) |
| DiT, 4 frames | 3072 | 0.0736 | +61% (p=0.0001, 0/10) |
| DiT, 8 frames | 3072 | 0.0448 | -1.7% (p=0.87, 5/10) |
| DiT, 16 frames | 3072 | 0.0469 | +2.8% (p=0.83, 6/10) |
| DiT 16 + state | 3093 | 0.0455 | -0.2% (p=0.99, 5/10) |

Temporal context carries the effect: 8 frames beats the same features at 1 frame by 48%
(p=0.0002, 9/10 tasks) and beats the VAE tokenizer by 56% (10/10). It saturates there --
16 frames is indistinguishable from 8 (p=0.46). The VAE compresses time 4x, so 8 frames is
2 latent frames and 16 is 4; the jump happens exactly where the model first gets more than
one latent frame.

At its best the video representation *ties* privileged proprioception rather than beating
it, and adds nothing on top of it. Matching exact joint angles and end-effector pose from
128x128 pixels is a real result, but it is parity bought with a 5B forward pass.

## Caveats

Ten tasks at 3 seeds per cell, no multiple-comparison correction across the seven
representations; on this project's track record that is the regime where apparent
per-configuration winners turn out to be noise. Block 15 of 30, a single noise level
(t=100) and mean-pooling over tokens were each chosen once and never varied, so the
representation may be further from its ceiling than the context length is. Single-task
imitation only -- whether a video representation forgets less across a task sequence is a
different question and is not tested here.
