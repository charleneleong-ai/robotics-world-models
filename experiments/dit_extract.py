#!/usr/bin/env python3
"""Cache Wan2.2 video-DiT features for LIBERO demonstrations.

The VAE probe tested the tokenizer. This tests the pretrained *world model*: each
timestep is given a window of preceding frames, encoded as a clip so the latent is in
the distribution the transformer was trained on, then passed through the diffusion
transformer at a low noise level with null text conditioning. Features are the
mid-block hidden states, mean-pooled over tokens (3072 dims, matching the VAE latent
width so the two are directly comparable).

No denoising loop and no frame generation: the transformer is run once, forward, as a
representation. Context length is the variable of interest -- context 1 makes the
temporal attention inert and acts as the control isolating what temporal context buys.

Usage: dit_extract.py <suite> <context_frames> [block]
"""
from __future__ import annotations

import time

import numpy as np
import torch
import typer
import wandb
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from torch import Tensor, nn

from probe_common import CACHE_ROOT, LiberoSuite, extract_suite, lagged_frames

WAN = "/home/ubuntu/wan22_ti2v_5b"


class DitFeatureExtractor:
    """Mid-block DiT features for the `context` frames ending at each timestep, edge-padded."""

    def __init__(self, context: int, block: int, timestep: int = 100, batch: int = 16) -> None:
        self.context, self.timestep, self.batch = context, timestep, batch
        self.vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae", torch_dtype=torch.float16).to("cuda").eval()
        self.dit = WanTransformer3DModel.from_pretrained(
            WAN, subfolder="transformer", torch_dtype=torch.float16).to("cuda").eval()
        self.hidden: Tensor | None = None
        self.dit.blocks[block].register_forward_hook(self.capture)

    def capture(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor | tuple[Tensor, ...]) -> None:
        self.hidden = output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        win = rgb[lagged_frames(len(rgb), range(self.context - 1, -1, -1))]   # (T,context,H,W,3)
        out = []
        for i in range(0, len(win), self.batch):
            w = torch.from_numpy(win[i : i + self.batch]).cuda().half().div(127.5).sub(1.0)
            lat = self.vae.encode(w.permute(0, 4, 1, 2, 3)).latent_dist.mode()   # (B,48,T',8,8)
            b = lat.shape[0]
            self.dit(hidden_states=lat,
                     timestep=torch.full((b,), self.timestep, dtype=torch.long, device="cuda"),
                     encoder_hidden_states=torch.zeros(b, 1, 4096, dtype=torch.float16, device="cuda"))
            out.append(self.hidden.float().mean(1).cpu().numpy())               # mean-pool tokens -> (B,3072)
        return np.concatenate(out).astype(np.float16)


def main(suite: str = typer.Argument("spatial"), context: int = typer.Argument(8),
         block: int = typer.Argument(15)) -> None:
    libero = LiberoSuite(suite)
    out_dir = CACHE_ROOT / f"{suite}_dit_ctx{context}"
    extractor = DitFeatureExtractor(context, block)
    run = wandb.init(project="video-wam", job_type="extract-dit", name=f"dit-{suite}-ctx{context}",
                     config=dict(suite=suite, camera=libero.camera, context_frames=context, block=block,
                                 timestep=extractor.timestep, backbone="Wan2.2-TI2V-5B-DiT",
                                 feature="midblock-meanpool", renders_frames=False, feature_dim=3072))
    t_start = time.time()
    total = extract_suite(libero, lambda rgb: {"dit": extractor(rgb)}, {"dit": out_dir},
                          f"dit ctx{context}", wandb.log)
    wandb.summary.update({"total_frames": total, "total_seconds": round(time.time() - t_start, 1)})
    print(f"== {total} frames -> {out_dir}", flush=True)
    run.finish()


if __name__ == "__main__":
    typer.run(main)
