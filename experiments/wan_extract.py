#!/usr/bin/env python3
"""Cache Wan2.2 video-VAE latents for LIBERO demonstrations.

The LIBERO HDF5 files carry two 128x128 RGB streams alongside the proprioceptive
state the earlier experiments used. This encodes each frame through the frozen
Wan2.2-TI2V-5B video VAE (spatial downsample 16x, so 128x128 -> 8x8x48 = 3072 dims)
and caches the result, giving a per-timestep representation aligned 1:1 with actions.

Usage: wan_extract.py <suite> [camera]   e.g. wan_extract.py spatial agentview_rgb
"""
from __future__ import annotations

import os
import sys
import time

import h5py
import numpy as np
import torch
import wandb
from diffusers import AutoencoderKLWan

LIBERO = "/home/ubuntu/robotics_world_models/LIBERO"
SUITE_DIRS = {s: f"{LIBERO}/libero_{s}" for s in ("spatial", "object", "goal")}

SUITE = sys.argv[1] if len(sys.argv) > 1 else "spatial"
CAMERA = sys.argv[2] if len(sys.argv) > 2 else "agentview_rgb"
WAN_PATH = "/home/ubuntu/wan22_ti2v_5b"
OUT_DIR = f"/home/ubuntu/wan_latents/{SUITE}_{CAMERA}"
N_TASKS, MAX_DEMOS, BATCH = 10, 12, 64
STATE_KEYS = ["ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"]


def load_vae() -> torch.nn.Module:
    vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae", torch_dtype=torch.float16)
    return vae.to("cuda").eval()


@torch.no_grad()
def encode_frames(vae, rgb: np.ndarray) -> np.ndarray:
    """(T,H,W,3) uint8 -> (T, 3072) float16, one latent per frame."""
    out = []
    for i in range(0, len(rgb), BATCH):
        chunk = torch.from_numpy(rgb[i:i + BATCH]).cuda().half().div(127.5).sub(1.0)
        x = chunk.permute(0, 3, 1, 2).unsqueeze(2)          # (B,3,1,H,W)
        lat = vae.encode(x).latent_dist.mode()              # (B,48,1,8,8)
        out.append(lat.flatten(1).cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    run = wandb.init(project="video-wam", job_type="extract", name=f"extract-{SUITE}-{CAMERA}",
                     config=dict(suite=SUITE, camera=CAMERA, backbone="Wan2.2-TI2V-5B-VAE",
                                 n_tasks=N_TASKS, max_demos=MAX_DEMOS, latent_dim=3072))
    vae = load_vae()
    d = SUITE_DIRS[SUITE]
    files = sorted(f for f in os.listdir(d) if f.endswith(".hdf5"))[:N_TASKS]
    total_frames, t_start = 0, time.time()

    for ti, fn in enumerate(files):
        t0 = time.time()
        lat_list, act_list, st_list, idx = [], [], [], []
        with h5py.File(os.path.join(d, fn)) as h:
            keys = [k for k in sorted(h["data"].keys()) if k.startswith("demo_")][:MAX_DEMOS]
            for di, k in enumerate(keys):
                demo = h["data"][k]
                rgb = np.array(demo["obs"][CAMERA])
                acts = np.array(demo["actions"], dtype=np.float32)
                state = np.concatenate([np.array(demo["obs"][s]) for s in STATE_KEYS], axis=-1).astype(np.float32)
                n = min(len(rgb), len(acts), len(state))
                lat_list.append(encode_frames(vae, rgb[:n]))
                act_list.append(acts[:n]); st_list.append(state[:n]); idx.append(np.full(n, di, np.int16))
                total_frames += n
        np.savez_compressed(os.path.join(OUT_DIR, f"task{ti:02d}.npz"),
                            latent=np.concatenate(lat_list), action=np.concatenate(act_list),
                            state=np.concatenate(st_list), demo=np.concatenate(idx))
        dt = time.time() - t0
        print(f"[{SUITE}] task {ti}: {len(keys)} demos, {sum(len(a) for a in act_list)} frames, {dt:.1f}s", flush=True)
        wandb.log({"task": ti, "frames": sum(len(a) for a in act_list), "seconds": dt})

    size_mb = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR)) / 1e6
    wandb.summary.update({"total_frames": total_frames, "cache_mb": round(size_mb, 1),
                          "total_seconds": round(time.time() - t_start, 1)})
    print(f"== {total_frames} frames -> {size_mb:.0f} MB in {OUT_DIR}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
