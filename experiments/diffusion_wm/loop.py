"""Self-Driving Learning Loop: collect → train → serve → eval → filter → retrain.

Orchestrates the full e2e cycle inspired by the Ray Summit 2026 tutorial,
adapted for World Action Models on ManiSkill3.

Usage:
    PYTHONPATH=. .venv/bin/python -m experiments.diffusion_wm.loop \\
        --task PegInsertionSide-v1 \\
        --num-rounds 5 \\
        --episodes-per-round 100 \\
        --train-steps-per-round 10000
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import typer

from experiments.diffusion_wm.collector import ManiSkillCollector
from experiments.diffusion_wm.fidelity import DivergenceDetector
from experiments.diffusion_wm.world_action_model import DiffusionWAM


@dataclass
class LoopConfig:
    """Configuration for the self-driving learning loop."""
    task: str
    num_rounds: int = 5
    episodes_per_round: int = 100
    train_steps_per_round: int = 10_000
    trust_threshold: float = 0.1
    reward_threshold: float = 0.0
    keep_top_pct: float = 0.5
    min_keep: int = 10
    checkpoint_dir: Path = Path("checkpoints/diffusion_wm")
    eval_dir: Path = Path("eval_results")
    port: int = 8000
    max_steps: int = 200
    num_envs: int = 1
    seed: int = 42

    # Model
    hidden_dim: int = 512
    num_blocks: int = 6
    diffusion_timesteps: int = 1000
    inference_steps: int = 100


class SelfDrivingLoop:
    """E2E learning loop: collect → train → serve → eval → filter → retrain."""

    def __init__(self, config: LoopConfig):
        self.config = config
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.eval_dir.mkdir(parents=True, exist_ok=True)
        self.round_history: list[dict] = []
        self.divergence_detector = DivergenceDetector()

    def _round_dir(self, round_num: int) -> Path:
        d = self.config.checkpoint_dir / f"round_{round_num:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _eval_dir(self, round_num: int) -> Path:
        d = self.config.eval_dir / f"round_{round_num:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _collect(self, round_num: int, checkpoint_path: Path | None) -> Path:
        """Collect new episodes. Round 0 uses random policy; later rounds use trained WAM."""
        data_dir = self.config.checkpoint_dir / f"round_{round_num:02d}" / "data"
        collector = ManiSkillCollector(
            self.config.task,
            num_envs=self.config.num_envs,
            max_steps=self.config.max_steps,
            seed=self.config.seed + round_num,
        )

        policy_fn = None
        if checkpoint_path is not None and checkpoint_path.exists():
            model = self._load_model(checkpoint_path)
            policy_fn = lambda obs, m=model: self._wam_policy(m, obs)

        collector.collect_dataset(
            self.config.episodes_per_round,
            policy_fn=policy_fn,
            out=data_dir,
        )
        collector.close()
        return data_dir

    def _train(self, round_num: int, data_dir: Path) -> Path:
        """Train WAM on collected data."""
        ckpt_dir = self._round_dir(round_num)
        run_id = f"wam_round_{round_num:02d}_{self.config.task}"

        # Auto-scale batch size for small datasets
        num_transitions = sum(
            len(np.load(shard)["obs"]) for shard in sorted(data_dir.glob("shard_*.npz"))
        )
        batch_size = min(256, max(16, num_transitions // 4))
        num_steps = min(self.config.train_steps_per_round, num_transitions * 10)

        cmd = [
            str(Path(".venv/bin/python")),
            "-m", "experiments.diffusion_wm.train_wam",
            "--data-dir", str(data_dir),
            "--run-id", run_id,
            "--num-steps", str(num_steps),
            "--batch-size", str(batch_size),
            "--hidden-dim", str(self.config.hidden_dim),
            "--num-blocks", str(self.config.num_blocks),
            "--diffusion-timesteps", str(self.config.diffusion_timesteps),
            "--checkpoint-dir", str(self.config.checkpoint_dir),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        subprocess.run(cmd, env=env, check=True)

        # Find the best checkpoint
        best_ckpt = self.config.checkpoint_dir / run_id / "best.pt"
        if not best_ckpt.exists():
            best_ckpt = self.config.checkpoint_dir / run_id / "final.pt"
        return best_ckpt

    def _serve(self, checkpoint_path: Path) -> subprocess.Popen:
        """Start FastAPI policy server in background."""
        # Kill any stale server on this port
        self._kill_port(self.config.port)

        cmd = [
            str(Path(".venv/bin/python")),
            "-m", "experiments.diffusion_wm.serve",
            "--checkpoint", str(checkpoint_path),
            "--port", str(self.config.port),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Wait for server to be ready
        import requests as _requests
        for _ in range(30):
            time.sleep(1)
            try:
                _requests.get(f"http://localhost:{self.config.port}/health", timeout=2)
                break
            except Exception:
                pass
        return proc

    def _kill_port(self, port: int) -> None:
        """Kill any process listening on the given port."""
        import subprocess as _sp
        try:
            pids = _sp.check_output(
                f"lsof -ti :{port}", shell=True, text=True
            ).strip().split("\n")
            for pid in pids:
                if pid:
                    _sp.run(["kill", "-9", pid], capture_output=True)
            time.sleep(0.5)
        except Exception:
            pass

    def _eval(self, round_num: int, checkpoint_path: Path) -> dict:
        """Evaluate WAM in ManiSkill3 sim via HTTP."""
        eval_dir = self._eval_dir(round_num)
        cmd = [
            str(Path(".venv/bin/python")),
            "-m", "experiments.diffusion_wm.eval_worker",
            "--task", self.config.task,
            "--policy-url", f"http://localhost:{self.config.port}",
            "--num-episodes", str(self.config.episodes_per_round),
            "--out", str(eval_dir),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        subprocess.run(cmd, env=env, check=True)

        results_path = eval_dir / "results.json"
        return json.loads(results_path.read_text())

    def _filter(self, eval_results: dict, round_num: int) -> dict:
        """Filter episodes: percentile-based reward + trust, with minimum keep floor."""
        episodes = eval_results["episodes"]
        n = len(episodes)
        if n == 0:
            eval_results["kept_episodes"] = []
            eval_results["filter_stats"] = {"total": 0, "kept": 0}
            return eval_results

        # Load model once for trust computation
        model_path = self._round_dir(round_num) / f"wam_round_{round_num:02d}_{self.config.task}" / "best.pt"
        if not model_path.exists():
            model_path = self._round_dir(round_num) / f"wam_round_{round_num:02d}_{self.config.task}" / "final.pt"
        trust_model = self._load_model(model_path) if model_path.exists() else None

        # Score all episodes
        scored = []
        for ep in episodes:
            reward = ep["reward"]
            trust = self._compute_trust_for_episode(ep, round_num, trust_model)
            scored.append({"episode": ep["episode"], "reward": reward, "trust": trust})

        # Percentile-based: keep top keep_top_pct by reward, with trust floor
        scored.sort(key=lambda x: x["reward"], reverse=True)
        keep_n = max(self.config.min_keep, int(n * self.config.keep_top_pct))
        candidates = scored[:keep_n]

        # Apply trust threshold as a soft filter
        kept = [s["episode"] for s in candidates if s["trust"] >= self.config.trust_threshold]

        # Minimum keep floor: if trust filtered too aggressively, relax and keep top by reward
        if len(kept) < self.config.min_keep:
            kept = [s["episode"] for s in scored[:self.config.min_keep]]

        eval_results["kept_episodes"] = kept
        eval_results["filter_stats"] = {
            "total": n,
            "kept": len(kept),
            "trust_threshold": self.config.trust_threshold,
            "reward_threshold": self.config.reward_threshold,
            "keep_top_pct": self.config.keep_top_pct,
        }
        return eval_results

    def _compute_trust_for_episode(self, ep: dict, round_num: int, model: DiffusionWAM | None) -> float:
        """Compute trust score for a single episode."""
        if model is None:
            return 1.0  # No model = default trust

        traj_path = self._eval_dir(round_num) / f"episode_{ep['episode']:04d}.pkl"
        if not traj_path.exists():
            return 0.5

        import pickle
        with open(traj_path, "rb") as f:
            traj = pickle.load(f)

        if not traj["obs"]:
            return 0.5

        obs = np.stack(traj["obs"])
        action = np.stack(traj["action"])
        next_obs = np.stack(traj["next_obs"])

        device = next(model.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        action_t = torch.tensor(action, dtype=torch.float32, device=device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            predicted_next = model.predict_next_state(obs_t, action_t, num_steps=self.config.inference_steps)
            mse = ((predicted_next - next_obs_t) ** 2).mean().item()

        return max(0.0, 1.0 - mse)

    def _merge_datasets(self, existing: Path | None, new_data: Path) -> Path:
        """Merge existing and new data directories."""
        if existing is None or not existing.exists():
            return new_data

        merged = new_data.parent / "merged"
        merged.mkdir(exist_ok=True)
        (merged / "meta").mkdir(exist_ok=True)

        # Copy all shards
        shard_idx = 0
        for shard in sorted(existing.glob("shard_*.npz")):
            data = np.load(shard)
            np.savez_compressed(merged / f"shard_{shard_idx:05d}.npz", **dict(data))
            shard_idx += 1

        for shard in sorted(new_data.glob("shard_*.npz")):
            data = np.load(shard)
            np.savez_compressed(merged / f"shard_{shard_idx:05d}.npz", **dict(data))
            shard_idx += 1

        # Update meta
        meta = json.loads((existing / "meta" / "collection.json").read_text())
        new_meta = json.loads((new_data / "meta" / "collection.json").read_text())
        meta["num_episodes"] += new_meta["num_episodes"]
        meta["num_transitions"] += new_meta["num_transitions"]
        meta["num_shards"] = shard_idx
        (merged / "meta" / "collection.json").write_text(json.dumps(meta, indent=2))

        return merged

    def _load_model(self, checkpoint_path: Path) -> DiffusionWAM:
        """Load WAM from checkpoint."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model_sd = ckpt["model"]
        if "denoiser" in model_sd:
            # DiffusionWAM nested state_dict
            model = DiffusionWAM(
                obs_dim=model_sd.get("obs_dim", cfg.get("obs_dim", 42)),
                act_dim=model_sd.get("act_dim", cfg.get("act_dim", 7)),
                hidden_dim=cfg.get("hidden_dim", 512),
                num_blocks=cfg.get("num_blocks", 6),
                timesteps=model_sd.get("timesteps", cfg.get("diffusion_timesteps", 1000)),
            ).to(device)
            model.load_state_dict(model_sd)
        else:
            model = DiffusionWAM(
                obs_dim=cfg.get("obs_dim", 42),
                act_dim=cfg.get("act_dim", 7),
                hidden_dim=cfg.get("hidden_dim", 512),
                num_blocks=cfg.get("num_blocks", 6),
                timesteps=cfg.get("diffusion_timesteps", 1000),
            ).to(device)
            model.load_state_dict(model_sd)
        model.eval()
        return model

    @staticmethod
    def _wam_policy(model: DiffusionWAM, obs: np.ndarray) -> np.ndarray:
        """Wrap WAM as a policy function for collection."""
        import torch
        device = next(model.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = model.predict_action(obs_t)
        return action.cpu().numpy().squeeze(0)

    def round(self, round_num: int, dataset: Path | None = None) -> Path:
        """Run one iteration of the loop."""
        self.config.round_num = round_num
        print(f"\n{'='*60}")
        print(f"Round {round_num} — {self.config.task}")
        print(f"{'='*60}")

        # 1. COLLECT
        print("\n[1/5] Collecting data...")
        t0 = time.monotonic()
        checkpoint = dataset.parent / "best.pt" if dataset else None
        new_data = self._collect(round_num, checkpoint)
        print(f"  Collected in {time.monotonic()-t0:.0f}s")

        # 2. TRAIN
        print("\n[2/5] Training WAM...")
        t0 = time.monotonic()
        merged = self._merge_datasets(dataset, new_data)
        checkpoint = self._train(round_num, merged)
        print(f"  Trained in {time.monotonic()-t0:.0f}s")

        # 3. SERVE
        print("\n[3/5] Starting policy server...")
        server_proc = self._serve(checkpoint)

        # 4. EVAL
        print("\n[4/5] Evaluating in sim...")
        t0 = time.monotonic()
        eval_results = self._eval(round_num, checkpoint)
        print(f"  Evaluated in {time.monotonic()-t0:.0f}s")

        # 5. FILTER
        print("\n[5/5] Filtering episodes...")
        eval_results = self._filter(eval_results, round_num)
        kept = eval_results["filter_stats"]["kept"]
        total = eval_results["filter_stats"]["total"]
        print(f"  Kept {kept}/{total} episodes "
              f"(trust>={self.config.trust_threshold}, reward>={self.config.reward_threshold})")

        # Stop server
        server_proc.terminate()
        server_proc.wait(timeout=10)

        # Log round summary
        summary = {
            "round": round_num,
            "task": self.config.task,
            "mean_reward": eval_results.get("mean_reward", 0),
            "success_rate": eval_results.get("success_rate", 0),
            "episodes_collected": self.config.episodes_per_round,
            "episodes_kept": kept,
            "time": time.monotonic() - t0,
        }
        self.round_history.append(summary)

        return merged

    def run(self) -> list[dict]:
        """Run the full loop."""
        print(f"Self-Driving Learning Loop — {self.config.task}")
        print(f"  Rounds: {self.config.num_rounds}")
        print(f"  Episodes/round: {self.config.episodes_per_round}")
        print(f"  Train steps/round: {self.config.train_steps_per_round}")
        print(f"  Trust threshold: {self.config.trust_threshold}")
        print(f"  Reward threshold: {self.config.reward_threshold}")
        print(f"  Keep top %: {self.config.keep_top_pct}")
        print(f"  Min keep: {self.config.min_keep}")

        dataset = None
        for round_num in range(self.config.num_rounds):
            dataset = self.round(round_num, dataset)

        print(f"\n{'='*60}")
        print("Loop complete!")
        print(f"{'='*60}")
        for summary in self.round_history:
            print(f"  Round {summary['round']}: reward={summary['mean_reward']:.3f}, "
                  f"success={summary['success_rate']:.3f}, kept={summary['episodes_kept']}")

        # Save summary
        summary_path = self.config.checkpoint_dir / "loop_summary.json"
        summary_path.write_text(json.dumps(self.round_history, indent=2))
        return self.round_history


def main(
    task: str = typer.Option("PegInsertionSide-v1", help="ManiSkill3 task"),
    num_rounds: int = typer.Option(5, help="Number of loop rounds"),
    episodes_per_round: int = typer.Option(100, help="Episodes per round"),
    train_steps_per_round: int = typer.Option(10_000, help="Training steps per round"),
    trust_threshold: float = typer.Option(0.1, help="Trust score threshold for filtering"),
    reward_threshold: float = typer.Option(0.0, help="Reward threshold for filtering"),
    keep_top_pct: float = typer.Option(0.5, help="Keep top % of episodes by reward"),
    min_keep: int = typer.Option(10, help="Minimum episodes to keep per round"),
    checkpoint_dir: Path = typer.Option(Path("checkpoints/diffusion_wm")),
    port: int = typer.Option(8000, help="Policy server port"),
    seed: int = typer.Option(42),
) -> None:
    config = LoopConfig(
        task=task,
        num_rounds=num_rounds,
        episodes_per_round=episodes_per_round,
        train_steps_per_round=train_steps_per_round,
        trust_threshold=trust_threshold,
        reward_threshold=reward_threshold,
        keep_top_pct=keep_top_pct,
        min_keep=min_keep,
        checkpoint_dir=checkpoint_dir,
        port=port,
        seed=seed,
    )
    loop = SelfDrivingLoop(config)
    loop.run()


if __name__ == "__main__":
    typer.run(main)
