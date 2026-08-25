"""Ray Serve policy server for WAM inference.

Usage:
    PYTHONPATH=. .venv/bin/python -m experiments.diffusion_wm.serve \\
        --checkpoint checkpoints/diffusion_wm/wam-v1/best.pt \\
        --port 8000

The server exposes:
    POST /predict  — single obs → action
    POST /predict_batch — batch of obs → batch of actions
    GET  /health   — health check
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import typer
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="WAM Policy Server")


class ObsRequest(BaseModel):
    obs: list[float]


class BatchRequest(BaseModel):
    obs: list[list[float]]


class ActionResponse(BaseModel):
    action: list[float]
    latency_ms: float


class BatchResponse(BaseModel):
    actions: list[list[float]]
    latency_ms: float


class WAMPolicyServer:
    """Ray Serve deployment for WAM inference.

    Handles model loading, GPU allocation, and request routing.
    Model-agnostic: can swap backend for Transfusion/MoT later.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        self._load_model(Path(checkpoint_path))
        self.model.eval()

    def _load_model(self, checkpoint_path: Path) -> None:
        from experiments.diffusion_wm.world_action_model import DiffusionWAM

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        model_sd = ckpt["model"]
        # DiffusionWAM state_dict is nested: {"denoiser": ..., "timesteps": ..., "obs_dim": ..., ...}
        if "denoiser" in model_sd:
            self.model = DiffusionWAM(
                obs_dim=model_sd.get("obs_dim", cfg.get("obs_dim", 42)),
                act_dim=model_sd.get("act_dim", cfg.get("act_dim", 7)),
                hidden_dim=cfg.get("hidden_dim", 512),
                num_blocks=cfg.get("num_blocks", 6),
                cond_dim=cfg.get("cond_dim", 256),
                timesteps=model_sd.get("timesteps", cfg.get("diffusion_timesteps", 1000)),
                action_horizon=model_sd.get("action_horizon", cfg.get("action_horizon", 1)),
            ).to(self.device)
            self.model.load_state_dict(model_sd)
        else:
            # Flat state dict from train.py (DynamicsMLP)
            self.model = DiffusionWAM(
                obs_dim=cfg.get("obs_dim", 42),
                act_dim=cfg.get("act_dim", 7),
                hidden_dim=cfg.get("hidden_dim", 512),
                num_blocks=cfg.get("num_blocks", 6),
                cond_dim=cfg.get("cond_dim", 256),
                timesteps=cfg.get("diffusion_timesteps", 1000),
                action_horizon=cfg.get("action_horizon", 1),
            ).to(self.device)
            self.model.load_state_dict(model_sd)
        print(f"Loaded WAM from {checkpoint_path} (obs_dim={self.model.obs_dim}, act_dim={self.model.act_dim})")

    @torch.no_grad()
    def predict(self, obs: np.ndarray, num_steps: int = 100) -> np.ndarray:
        """Single observation → action."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.model.predict_action(obs_t, num_steps=num_steps)
        return action.cpu().numpy().squeeze(0)

    @torch.no_grad()
    def predict_batch(self, obs: np.ndarray, num_steps: int = 100) -> np.ndarray:
        """Batch of observations → batch of actions."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = self.model.predict_action(obs_t, num_steps=num_steps)
        return action.cpu().numpy()


# Global server instance (initialized on first request)
_server: WAMPolicyServer | None = None


def _get_server() -> WAMPolicyServer:
    global _server
    if _server is None:
        import os

        ckpt = os.environ.get("WAM_CHECKPOINT", "checkpoints/diffusion_wm/wam-v1/best.pt")
        device = os.environ.get("WAM_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        _server = WAMPolicyServer(ckpt, device)
    return _server


@app.post("/predict")
def predict(req: ObsRequest) -> ActionResponse:
    server = _get_server()
    obs = np.array(req.obs, dtype=np.float32)
    t0 = time.monotonic()
    action = server.predict(obs)
    latency_ms = (time.monotonic() - t0) * 1000
    return ActionResponse(action=action.tolist(), latency_ms=latency_ms)


@app.post("/predict_batch")
def predict_batch(req: BatchRequest) -> BatchResponse:
    server = _get_server()
    obs = np.array(req.obs, dtype=np.float32)
    t0 = time.monotonic()
    actions = server.predict_batch(obs)
    latency_ms = (time.monotonic() - t0) * 1000
    return BatchResponse(actions=actions.tolist(), latency_ms=latency_ms)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _server is not None}


def run_server(
    checkpoint: str = "checkpoints/diffusion_wm/wam-v1/best.pt",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Start the policy server."""
    import uvicorn

    import os
    os.environ["WAM_CHECKPOINT"] = checkpoint
    uvicorn.run(app, host=host, port=port)


def main(
    checkpoint: Path = typer.Option(..., help="WAM checkpoint path"),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
) -> None:
    run_server(str(checkpoint), host, port)


if __name__ == "__main__":
    typer.run(main)
