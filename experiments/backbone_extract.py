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

import time

import numpy as np
import torch
import torch.nn.functional as F
import typer
import wandb

from probe_common import CACHE_ROOT, LiberoSuite, extract_suite, lagged_frames

CTX_GAP = 7


class ImageBackbone:
    """A frozen single-image encoder, mean-pooled over tokens."""

    def __init__(self, name: str, batch: int = 32) -> None:
        self.name, self.batch = name, batch
        self.model, self.res = self.build(name)

    @staticmethod
    def build(name: str) -> tuple[torch.nn.Module, int]:
        # Imported per backbone: the two run under different transformers majors, and
        # AutoModelForVision2Seq exists only in the 4.x the OpenVLA checkpoint needs.
        if name == "siglip2":
            from transformers import AutoModel
            m = AutoModel.from_pretrained("google/siglip2-so400m-patch16-384", torch_dtype=torch.float16)
            return m.vision_model.cuda().eval(), 384
        if name == "openvla":
            from transformers import AutoModelForVision2Seq
            m = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", torch_dtype=torch.float16,
                                                       trust_remote_code=True, low_cpu_mem_usage=True)
            return m.vision_backbone.cuda().eval(), 224
        raise typer.BadParameter(f"unknown backbone {name}")

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(rgb), self.batch):
            x = torch.from_numpy(rgb[i : i + self.batch]).cuda().half().div(127.5).sub(1.0).permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
            feats = (self.model(torch.cat([x, x], 1)) if self.name == "openvla"
                     else self.model(pixel_values=x).last_hidden_state)
            out.append(feats.float().mean(1).cpu().numpy())
        return np.concatenate(out).astype(np.float16)

    def with_context(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Per-frame features, plus frame t-CTX_GAP concatenated before frame t."""
        per_frame = self(rgb)
        paired = per_frame[lagged_frames(len(per_frame), [CTX_GAP, 0])]
        return {"plain": per_frame, "ctx8": paired.reshape(len(per_frame), -1)}


def main(backbone: str = typer.Argument("siglip2"), suite: str = typer.Argument("spatial")) -> None:
    libero = LiberoSuite(suite)
    out_dirs = {"plain": CACHE_ROOT / f"{suite}_{backbone}", "ctx8": CACHE_ROOT / f"{suite}_{backbone}_ctx8"}
    run = wandb.init(project="video-wam", job_type="extract-backbone", name=f"{backbone}-{suite}",
                     config=dict(backbone=backbone, suite=suite, camera=libero.camera, ctx_gap=CTX_GAP))
    encoder, t0 = ImageBackbone(backbone), time.time()
    total = extract_suite(libero, encoder.with_context, out_dirs, backbone, wandb.log)
    feature_dim = int(np.load(out_dirs["plain"] / "task00.npz")["latent"].shape[1])
    wandb.summary.update({"total_frames": total, "seconds": round(time.time() - t0, 1), "feature_dim": feature_dim})
    print(f"== {backbone}: {total} frames -> {out_dirs['plain']} and {out_dirs['ctx8']}", flush=True)
    run.finish()


if __name__ == "__main__":
    typer.run(main)
