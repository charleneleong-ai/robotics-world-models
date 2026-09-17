#!/usr/bin/env python3
"""Cache features from image backbones trained on different objectives, for the audit.

The Wan probes compared a video reconstruction objective (VAE) against a video denoising
objective (DiT). These add two more pretraining objectives over the same LIBERO frames:

  siglip2   image-text contrastive -- semantic vision, no dynamics, no robot data
  openvla   robot action prediction -- in-domain, the natural upper bound for what
            pretraining on robot demonstrations buys

Both are single-image encoders, so each is cached twice: per frame (matched to `vae` and
`dit_ctx1`) and as the concatenation of frames t-7 and t (matched to `dit_ctx8`, which
receives two latent frames). Frames are encoded once and the context variant is assembled
by indexing, so the pairing costs nothing.

Usage: backbone_extract.py <siglip2|openvla> [suite]
  siglip2 runs under ~/wan_venv; openvla needs the system python (transformers 4.40.1).
"""
from __future__ import annotations

import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import wandb

LIBERO = "/home/ubuntu/robotics_world_models/LIBERO"
SUITE_DIRS = {s: f"{LIBERO}/libero_{s}" for s in ("spatial", "object", "goal")}

BACKBONE = sys.argv[1] if len(sys.argv) > 1 else "siglip2"
SUITE = sys.argv[2] if len(sys.argv) > 2 else "spatial"
CAMERA, N_TASKS, MAX_DEMOS, BATCH, CTX_GAP = "agentview_rgb", 10, 12, 32, 7
STATE_KEYS = ["ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"]
ROOT = "/home/ubuntu/wan_latents"


def build() -> tuple[torch.nn.Module, int]:
    if BACKBONE == "siglip2":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("google/siglip2-so400m-patch16-384", torch_dtype=torch.float16)
        return m.vision_model.cuda().eval(), 384
    if BACKBONE == "openvla":
        from transformers import AutoModelForVision2Seq
        m = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", torch_dtype=torch.float16,
                                                   trust_remote_code=True, low_cpu_mem_usage=True)
        return m.vision_backbone.cuda().eval(), 224
    raise ValueError(BACKBONE)


@torch.no_grad()
def encode(model, res: int, rgb: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, len(rgb), BATCH):
        x = torch.from_numpy(rgb[i:i + BATCH]).cuda().half().div(127.5).sub(1.0).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
        feats = model(torch.cat([x, x], 1)) if BACKBONE == "openvla" else model(pixel_values=x).last_hidden_state
        out.append(feats.float().mean(1).cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def main() -> None:
    plain, ctx = f"{ROOT}/{SUITE}_{BACKBONE}", f"{ROOT}/{SUITE}_{BACKBONE}_ctx8"
    os.makedirs(plain, exist_ok=True); os.makedirs(ctx, exist_ok=True)
    run = wandb.init(project="video-wam", job_type="extract-backbone", name=f"{BACKBONE}-{SUITE}",
                     config=dict(backbone=BACKBONE, suite=SUITE, camera=CAMERA, ctx_gap=CTX_GAP))
    model, res = build()
    d = SUITE_DIRS[SUITE]
    files = sorted(f for f in os.listdir(d) if f.endswith(".hdf5"))[:N_TASKS]
    total, t0 = 0, time.time()

    for ti, fn in enumerate(files):
        fp, fc, acts, sts, dem = [], [], [], [], []
        with h5py.File(os.path.join(d, fn)) as h:
            for di, k in enumerate([k for k in sorted(h["data"].keys()) if k.startswith("demo_")][:MAX_DEMOS]):
                demo = h["data"][k]
                rgb = np.array(demo[f"obs/{CAMERA}"])
                a = np.array(demo["actions"], dtype=np.float32)
                s = np.concatenate([np.array(demo[f"obs/{x}"]) for x in STATE_KEYS], -1).astype(np.float32)
                n = min(len(rgb), len(a), len(s))
                f_t = encode(model, res, rgb[:n])
                prev = f_t[np.clip(np.arange(n) - CTX_GAP, 0, None)]
                fp.append(f_t); fc.append(np.concatenate([prev, f_t], 1))
                acts.append(a[:n]); sts.append(s[:n]); dem.append(np.full(n, di, np.int16)); total += n
        common = dict(action=np.concatenate(acts), state=np.concatenate(sts), demo=np.concatenate(dem))
        np.savez_compressed(f"{plain}/task{ti:02d}.npz", latent=np.concatenate(fp), **common)
        np.savez_compressed(f"{ctx}/task{ti:02d}.npz", latent=np.concatenate(fc), **common)
        print(f"[{BACKBONE}] task {ti}: {sum(len(a) for a in acts)} frames, dim {fp[0].shape[1]}", flush=True)
        wandb.log({"task": ti})
    wandb.summary.update({"total_frames": total, "seconds": round(time.time() - t0, 1),
                          "feature_dim": int(fp[0].shape[1])})
    print(f"== {BACKBONE}: {total} frames -> {plain} and {ctx}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
