#!/usr/bin/env python3
"""Cache Wan2.2 video-DiT features for LIBERO demonstrations.

The VAE probe tested the tokenizer. This tests the pretrained *world model*: each
timestep is given a window of preceding frames, encoded as a clip so the latent is in
the distribution the transformer was trained on, then passed through the diffusion
transformer at a low noise level with null text conditioning. Features are the
mid-block hidden states, mean-pooled over tokens (3072 dims, matching the VAE latent
width so the two are directly comparable).

No denoising loop and no frame generation: the transformer is run once, forward, as a
representation. Context length is the variable of interest -- CONTEXT=1 makes the
temporal attention inert and acts as the control isolating what temporal context buys.

Usage: dit_extract.py <suite> <context_frames> [block]
"""
from __future__ import annotations

import os
import sys
import time

import h5py
import numpy as np
import torch
import wandb
from diffusers import AutoencoderKLWan, WanTransformer3DModel

sys.path.insert(0, "/home/ubuntu/robotics_world_models/experiments/causal_trust_world_model_learning")
from task_order_sensitivity import SUITE_DIRS  # noqa: E402

SUITE = sys.argv[1] if len(sys.argv) > 1 else "spatial"
CONTEXT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BLOCK = int(sys.argv[3]) if len(sys.argv) > 3 else 15
WAN, CAMERA = "/home/ubuntu/wan22_ti2v_5b", "agentview_rgb"
OUT_DIR = f"/home/ubuntu/wan_latents/{SUITE}_dit_ctx{CONTEXT}"
N_TASKS, MAX_DEMOS, BATCH, TIMESTEP = 10, 12, 16, 100
STATE_KEYS = ["ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"]
FEAT: dict[str, torch.Tensor] = {}


def windows(rgb: np.ndarray) -> np.ndarray:
    """(T,H,W,3) -> (T,CONTEXT,H,W,3); window t is the CONTEXT frames ending at t, edge-padded."""
    idx = np.arange(len(rgb))[:, None] - np.arange(CONTEXT - 1, -1, -1)[None, :]
    return rgb[np.clip(idx, 0, None)]


@torch.no_grad()
def encode(vae, dit, rgb: np.ndarray) -> np.ndarray:
    win = windows(rgb)
    out = []
    for i in range(0, len(win), BATCH):
        w = torch.from_numpy(win[i:i + BATCH]).cuda().half().div(127.5).sub(1.0)
        x = w.permute(0, 4, 1, 2, 3)                                  # (B,3,CONTEXT,H,W)
        lat = vae.encode(x).latent_dist.mode()                        # (B,48,T',8,8)
        b = lat.shape[0]
        dit(hidden_states=lat,
            timestep=torch.full((b,), TIMESTEP, dtype=torch.long, device="cuda"),
            encoder_hidden_states=torch.zeros(b, 1, 4096, dtype=torch.float16, device="cuda"))
        out.append(FEAT["h"].float().mean(1).cpu().numpy())           # mean-pool tokens -> (B,3072)
    return np.concatenate(out).astype(np.float16)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    run = wandb.init(project="video-wam", job_type="extract-dit", name=f"dit-{SUITE}-ctx{CONTEXT}",
                     config=dict(suite=SUITE, camera=CAMERA, context_frames=CONTEXT, block=BLOCK,
                                 timestep=TIMESTEP, backbone="Wan2.2-TI2V-5B-DiT", feature="midblock-meanpool",
                                 renders_frames=False, feature_dim=3072))
    vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae", torch_dtype=torch.float16).to("cuda").eval()
    dit = WanTransformer3DModel.from_pretrained(WAN, subfolder="transformer", torch_dtype=torch.float16).to("cuda").eval()
    dit.blocks[BLOCK].register_forward_hook(
        lambda _m, _i, o: FEAT.__setitem__("h", o[0] if isinstance(o, tuple) else o))

    d = SUITE_DIRS[SUITE]
    files = sorted(f for f in os.listdir(d) if f.endswith(".hdf5"))[:N_TASKS]
    total, t_start = 0, time.time()
    for ti, fn in enumerate(files):
        t0 = time.time()
        feats, acts, sts, dem = [], [], [], []
        with h5py.File(os.path.join(d, fn)) as h:
            for di, k in enumerate([k for k in sorted(h["data"].keys()) if k.startswith("demo_")][:MAX_DEMOS]):
                demo = h["data"][k]
                rgb = np.array(demo[f"obs/{CAMERA}"])
                a = np.array(demo["actions"], dtype=np.float32)
                s = np.concatenate([np.array(demo[f"obs/{x}"]) for x in STATE_KEYS], -1).astype(np.float32)
                n = min(len(rgb), len(a), len(s))
                feats.append(encode(vae, dit, rgb[:n])); acts.append(a[:n]); sts.append(s[:n])
                dem.append(np.full(n, di, np.int16)); total += n
        np.savez_compressed(os.path.join(OUT_DIR, f"task{ti:02d}.npz"),
                            latent=np.concatenate(feats), action=np.concatenate(acts),
                            state=np.concatenate(sts), demo=np.concatenate(dem))
        dt = time.time() - t0
        print(f"[dit ctx{CONTEXT}] task {ti}: {sum(len(a) for a in acts)} frames, {dt:.1f}s", flush=True)
        wandb.log({"task": ti, "seconds": dt})
    wandb.summary.update({"total_frames": total, "total_seconds": round(time.time() - t_start, 1)})
    print(f"== {total} frames -> {OUT_DIR}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
