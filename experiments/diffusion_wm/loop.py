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
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
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
    trust_threshold: float = 0.7
    reward_threshold: float = 0.5
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

    def _filter(self, eval_results: dict) -> dict:
        """Filter episodes by trust score AND reward."""
        kept = []
        for ep in eval_results["episodes"]:
            # Trust: check if episode trajectories diverge from WM predictions
            traj_path = self._eval_dir(self.config.round_num) / f"episode_{ep['episode']:04d}.pkl"
            if traj_path.exists():
                import pickle
                with open(traj_path, "rb") as f:
                    traj = pickle.load(f)

                trust_score = self._compute_trust(traj)
                reward = ep["reward"]

                if trust_score >= self.config.trust_threshold and reward >= self.config.reward_threshold:
                    kept.append(ep["episode"])

        eval_results["kept_episodes"] = kept
        eval_results["filter_stats"] = {
            "total": len(eval_results["episodes"]),
            "kept": len(kept),
            "trust_threshold": self.config.trust_threshold,
            "reward_threshold": self.config.reward_threshold,
        }
        return eval_results

    def _compute_trust(self, trajectory: dict) -> float:
        """Compute trust score for a trajectory using WM predictions."""
        if not trajectory["obs"]:
            return 0.0

        obs = np.stack(trajectory["obs"])
        action = np.stack(trajectory["action"])
        next_obs = np.stack(trajectory["next_obs"])

        # Compute prediction error as a proxy for trust
        obs_t = torch.tensor(obs, dtype=torch.float32)
        action_t = torch.tensor(action, dtype=torch.float32)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32)

        # Load current model for trust computation
        model_path = self._round_dir(self.config.round_num) / f"wam_round_{self.config.round_num:02d}_{self.config.task}" / "best.pt"
        if not model_path.exists():
            return 0.5  # Default trust if no model

        model = self._load_model(model_path)
        with torch.no_grad():
            predicted_next = model.predict_next_state(obs_t, action_t, num_steps=self.config.inference_steps)
            mse = ((predicted_next - next_obs_t) ** 2).mean().item()

        # Trust: 1.0 for perfect prediction, decays with error
        trust = max(0.0, 1.0 - mse)
        return trust

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
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = DiffusionWAM(
            obs_dim=cfg.get("obs_dim", 34),
            act_dim=cfg.get("act_dim", 7),
            hidden_dim=cfg.get("hidden_dim", 512),
            num_blocks=cfg.get("num_blocks", 6),
            timesteps=cfg.get("diffusion_timesteps", 1000),
        ).to(device)
        model.load_state_dict(ckpt["model"])
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
        eval_results = self._filter(eval_results)
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
    trust_threshold: float = typer.Option(0.7, help="Trust score threshold for filtering"),
    reward_threshold: float = typer.Option(0.5, help="Reward threshold for filtering"),
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
        checkpoint_dir=checkpoint_dir,
        port=port,
        seed=seed,
    )
    loop = SelfDrivingLoop(config)
    loop.run()


if __name__ == "__main__":
    typer.run(main)
