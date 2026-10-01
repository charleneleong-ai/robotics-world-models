#!/usr/bin/env python3
"""Cache Wan2.2 video-VAE latents for LIBERO demonstrations.

The LIBERO HDF5 files carry two 128x128 RGB streams alongside the proprioceptive
state the earlier experiments used. This encodes each frame through the frozen
Wan2.2-TI2V-5B video VAE (spatial downsample 16x, so 128x128 -> 8x8x48 = 3072 dims)
and caches the result, giving a per-timestep representation aligned 1:1 with actions.

Usage: wan_extract.py <suite> [camera]   e.g. wan_extract.py spatial agentview_rgb
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import typer
import wandb
from diffusers import AutoencoderKLWan

from probe_common import CACHE_ROOT, LiberoSuite, extract_suite

WAN_PATH = "/home/ubuntu/wan22_ti2v_5b"


class VaeFrameEncoder:
    """One Wan VAE latent per frame, each frame encoded as a single-frame clip."""

    def __init__(self, batch: int = 64) -> None:
        vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae", torch_dtype=torch.float16)
        self.vae, self.batch = vae.to("cuda").eval(), batch

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """(T,H,W,3) uint8 -> (T, 3072) float16."""
        out = []
        for i in range(0, len(rgb), self.batch):
            chunk = torch.from_numpy(rgb[i : i + self.batch]).cuda().half().div(127.5).sub(1.0)
            x = chunk.permute(0, 3, 1, 2).unsqueeze(2)          # (B,3,1,H,W)
            lat = self.vae.encode(x).latent_dist.mode()         # (B,48,1,8,8)
            out.append(lat.flatten(1).cpu().numpy())
        return np.concatenate(out).astype(np.float16)


def main(suite: str = typer.Argument("spatial"), camera: str = typer.Argument("agentview_rgb")) -> None:
    libero = LiberoSuite(suite, camera)
    out_dir = CACHE_ROOT / f"{suite}_{camera}"
    run = wandb.init(project="video-wam", job_type="extract", name=f"extract-{suite}-{camera}",
                     config=dict(suite=suite, camera=camera, backbone="Wan2.2-TI2V-5B-VAE",
                                 n_tasks=libero.n_tasks, max_demos=libero.max_demos, latent_dim=3072))
    encoder, t_start = VaeFrameEncoder(), time.time()
    total = extract_suite(libero, lambda rgb: {"vae": encoder(rgb)}, {"vae": out_dir}, suite, wandb.log)
    size_mb = sum(f.stat().st_size for f in Path(out_dir).iterdir()) / 1e6
    wandb.summary.update({"total_frames": total, "cache_mb": round(size_mb, 1),
                          "total_seconds": round(time.time() - t_start, 1)})
    print(f"== {total} frames -> {size_mb:.0f} MB in {out_dir}", flush=True)
    run.finish()


if __name__ == "__main__":
    typer.run(main)
