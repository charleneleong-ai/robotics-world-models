"""Multi-Benchmark Sweep for ContinualWAM.

Runs backbone x trust scoring experiments across:
1. ManiSkill  (PushCube, LiftPeg, StackCube)  — manipulation
2. LIBERO     (spatial, object, goal suites)   — language-conditioned
3. KinDER     (57 physics reasoning tasks)     — physical reasoning

Each benchmark uses the same WorldModelBackbone interface from full_backbone_sweep.
"""

from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("DISPLAY", "")

import argparse
import json
import time

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from wandb_helpers import collect_eval_frames, log_video, log_frame_grid, log_reward_chart

# Import shared backbone/trust classes from the sweep module
from full_backbone_sweep import (
    BACKBONES,
    TRUST_NAMES,
    WorldModelBackbone,
    Policy,
    EWC,
    make_trust,
    collect_episode,
    collect_buffer,
)


# ============================================================================
# LIBERO BENCHMARK
# ============================================================================

def run_libero_suite(
    bb_name: str,
    tr_name: str,
    suite_name: str = "libero_spatial",
    n_ep: int = 10,
    max_steps: int = 200,
    dev: str = "cuda",
) -> dict:
    """Run one (backbone, trust) on a LIBERO suite (10 tasks)."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    device = torch.device(dev if torch.cuda.is_available() else "cpu")

    bd = benchmark.get_benchmark_dict()
    suite = bd[suite_name]()
    n_tasks = suite.n_tasks
    task_names = [suite.get_task(i).name for i in range(n_tasks)]

    all_rewards = []
    run = wandb.init(project="continualwam", name=f"{suite_name}-{bb_name}-{tr_name}",
        tags=[bb_name, tr_name, suite_name],
        config={"backbone": bb_name, "trust": tr_name, "suite": suite_name}, reinit=True)

    for ti in range(n_tasks):
        task = suite.get_task(ti)
        bddl_path = suite.get_task_bddl_file_path(ti)
        env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=64, camera_widths=64)

        # Get obs/act dims — LIBERO uses OffScreenRenderEnv (no gym API)
        obs_dict = env.reset()
        obs = _flatten_obs(obs_dict)
        obs_dim = obs.shape[0]
        act_dim = env.env.action_dim  # LIBERO wraps robosuite, action_dim on inner env
        # Keep env open — it will be reused across phases

        # Create models per task (different obs_dim possible)
        backbone = BACKBONES[bb_name](obs_dim, act_dim).to(device)
        trust = make_trust(tr_name, obs_dim, act_dim)
        policy = Policy(obs_dim, act_dim).to(device)
        ewc = EWC(policy)

        # Phase 1: Collect data & train world model
        bb_opt = torch.optim.Adam(backbone.parameters(), lr=3e-4)
        for _ in range(20):
            obs_list, act_list = [], []
            for ep in range(n_ep):
                obs_dict = env.reset()
                obs = _flatten_obs(obs_dict)
                ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                for _ in range(max_steps):
                    with torch.no_grad():
                        at = policy(ot)
                    result = env.step(at.detach().cpu().numpy().flatten())
                    no_dict, r, term = result[0], result[1], result[2]
                    trunc = False  # LIBERO doesn't have truncation
                    no = _flatten_obs(no_dict)
                    obs_list.append(ot.squeeze(0))
                    act_list.append(at.squeeze(0))
                    ot = torch.from_numpy(no).float().unsqueeze(0).to(device)
                    if term or trunc:
                        break

            if len(obs_list) > 2:
                all_obs = torch.stack(obs_list)
                all_act = torch.stack(act_list)
                T = min(32, len(obs_list))
                n_seqs = len(obs_list) // T
                if n_seqs > 0:
                    obs_seqs = all_obs[:n_seqs * T].view(n_seqs, T, -1)
                    act_seqs = all_act[:n_seqs * T].view(n_seqs, T, -1)
                    loss = backbone.train_loss(obs_seqs, act_seqs)
                    bb_opt.zero_grad()
                    loss.backward()
                    bb_opt.step()

        ewc.consolidate()

        # Phase 2: Train policy with trust
        pol_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        for ep in range(n_ep):
            obs_dict = env.reset()
            obs = _flatten_obs(obs_dict)
            ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            for _ in range(max_steps):
                at = policy(ot)
                result = env.step(at.detach().cpu().numpy().flatten())
                no_dict, r, term = result[0], result[1], result[2]
                trunc = False
                no = _flatten_obs(no_dict)
                not_ = torch.from_numpy(no).float().unsqueeze(0).to(device)

                tw = 1.0
                if trust is not None:
                    with torch.no_grad():
                        err = backbone.predict_error(ot, at, not_)
                        if tr_name in ("ema", "multi_step"):
                            tw = float(trust.compute_trust(err, ti).mean())
                        elif tr_name == "ffdc":
                            tw = float(trust.compute_trust(ot, not_, at).mean())
                        elif tr_name == "ensemble":
                            tw = float(trust.compute_trust(ot).mean())

                target = torch.randn(1, act_dim, device=device)
                loss = F.mse_loss(at, target) * tw + ewc.penalty()
                pol_opt.zero_grad()
                loss.backward()
                pol_opt.step()
                ot = not_
                if term or trunc:
                    break

        # Phase 3: Evaluate with rendering
        eval_rews, eval_frames = [], []
        for ep in range(5):
            obs_dict = env.reset()
            obs = _flatten_obs(obs_dict)
            ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            total_r, ep_frames = 0, []
            for step in range(max_steps):
                if ep == 0 and step % max(1, max_steps // 8) == 0:
                    try:
                        f = env.sim.render(64, 64, camera_name="agentview")
                        if isinstance(f, np.ndarray) and f.ndim == 3: ep_frames.append(f)
                    except Exception: pass
                with torch.no_grad():
                    at = policy(ot)
                result = env.step(at.detach().cpu().numpy().flatten())
                no_dict, r, term = result[0], result[1], result[2]
                trunc = False
                total_r += r
                ot = torch.from_numpy(_flatten_obs(no_dict)).float().unsqueeze(0).to(device)
                if term or trunc: break
            eval_rews.append(total_r)
            if ep == 0: eval_frames = ep_frames
        avg_r = float(np.mean(eval_rews))
        all_rewards.append(avg_r)
        env.close()
        print(f"  Task {ti}: {task.name[:50]}... -> {avg_r:.3f}")

    final = {"benchmark": f"libero_{suite_name}", "backbone": bb_name, "trust": tr_name,
             "task_rewards": all_rewards, "avg_reward": float(np.mean(all_rewards))}
    log_reward_chart(run, "chart/learning_curve", all_rewards, f"{suite_name} {bb_name}+{tr_name}")
    run.log({"avg_reward": final["avg_reward"]})
    run.finish()
    return final


def _flatten_obs(obs) -> np.ndarray:
    """Flatten dict observation to 1D float32, excluding image keys."""
    if isinstance(obs, dict):
        parts = []
        for k, v in obs.items():
            if 'image' in k:
                continue
            arr = np.asarray(v, dtype=np.float32).flatten()
            parts.append(arr)
        return np.concatenate(parts)
    return np.asarray(obs, dtype=np.float32).flatten()


# ============================================================================
# MANISKILL BENCHMARK (from full_backbone_sweep)
# ============================================================================

def run_maniskill(
    bb_name: str,
    tr_name: str,
    tasks: list[str] | None = None,
    n_ep: int = 10,
    max_steps: int = 50,
    dev: str = "cuda",
) -> dict:
    """Run one (backbone, trust) on ManiSkill tasks."""
    from full_backbone_sweep import run_one
    if tasks is None:
        tasks = ["PushCube-v1", "LiftPegUpright-v1", "StackCube-v1"]
    run = wandb.init(project="continualwam", name=f"maniskill-{bb_name}-{tr_name}",
        tags=[bb_name, tr_name, "maniskill"],
        config={"backbone": bb_name, "trust": tr_name}, reinit=True)
    r = run_one(bb_name, tr_name, tasks, n_ep=n_ep, max_steps=max_steps, dev=dev)
    r["benchmark"] = "maniskill"
    log_reward_chart(run, "chart/learning_curve", r["task_rewards"], f"ManiSkill {bb_name}+{tr_name}")
    run.log({"avg_reward": r["avg_reward"]})
    run.finish()
    return r


# ============================================================================
# KINDER BENCHMARK
# ============================================================================

def run_kinder(
    bb_name: str,
    tr_name: str,
    n_tasks: int = 10,
    n_ep: int = 10,
    max_steps: int = 50,
    dev: str = "cuda",
) -> dict:
    """Run one (backbone, trust) on KinDER physics tasks."""
    try:
        import kinder
        kinder.register_all_environments()
        all_ids = sorted(kinder.get_all_env_ids())
        # Filter to 2D envs only (3D need EGL rendering)
        env_ids = [eid for eid in all_ids if "2D" in eid][:n_tasks]
        if not env_ids:
            print("  KinDER: no envs found after registration")
            return {"benchmark": "kinder", "backbone": bb_name, "trust": tr_name,
                    "task_rewards": [], "avg_reward": 0.0, "error": "no kinDER envs"}
    except ImportError:
        print("  KinDER not installed, skipping")
        return {"benchmark": "kinder", "backbone": bb_name, "trust": tr_name,
                "task_rewards": [], "avg_reward": 0.0, "error": "kinder not installed"}

    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    all_rewards = []
    run = wandb.init(project="continualwam", name=f"kinder-{bb_name}-{tr_name}",
        tags=[bb_name, tr_name, "kinder"],
        config={"backbone": bb_name, "trust": tr_name}, reinit=True)

    for ti, env_id in enumerate(env_ids):
        try:
            env = gym.make(env_id, render_mode=None)
            obs_dim = int(np.asarray(env.reset()[0], dtype=np.float32).flatten().shape[0])
            act_dim = int(env.action_space.shape[0])
        except Exception as e:
            print(f"  Task {ti} ({env_id}): SKIP ({e})")
            all_rewards.append(0.0)
            continue

        backbone = BACKBONES[bb_name](obs_dim, act_dim).to(device)
        trust = make_trust(tr_name, obs_dim, act_dim)
        policy = Policy(obs_dim, act_dim).to(device)
        ewc = EWC(policy)

        # Train world model
        bb_opt = torch.optim.Adam(backbone.parameters(), lr=3e-4)
        for _ in range(20):
            obs_list, act_list = [], []
            for ep in range(n_ep):
                obs, _ = env.reset(seed=ti * 1000 + ep)
                obs = np.asarray(obs, dtype=np.float32).flatten()
                ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                for _ in range(max_steps):
                    with torch.no_grad():
                        at = policy(ot)
                    at_clamped = np.clip(at.detach().cpu().numpy().flatten(), env.action_space.low, env.action_space.high)
                    no, r, term, trunc, _ = env.step(at_clamped)
                    obs_list.append(ot.squeeze(0))
                    act_list.append(at.squeeze(0))
                    ot = torch.from_numpy(np.asarray(no, dtype=np.float32).flatten()).float().unsqueeze(0).to(device)
                    if term or trunc:
                        break
            if len(obs_list) > 2:
                all_obs = torch.stack(obs_list)
                all_act = torch.stack(act_list)
                T = min(32, len(obs_list))
                n_seqs = len(obs_list) // T
                if n_seqs > 0:
                    loss = backbone.train_loss(
                        all_obs[:n_seqs * T].view(n_seqs, T, -1),
                        all_act[:n_seqs * T].view(n_seqs, T, -1),
                    )
                    bb_opt.zero_grad()
                    loss.backward()
                    bb_opt.step()

        ewc.consolidate()

        # Train policy
        pol_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        for ep in range(n_ep):
            obs, _ = env.reset(seed=ti * 1000 + ep + 5000)
            obs = np.asarray(obs, dtype=np.float32).flatten()
            ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            for _ in range(max_steps):
                at = policy(ot)
                at_clamped = np.clip(at.detach().cpu().numpy().flatten(), env.action_space.low, env.action_space.high)
                no, r, term, trunc, _ = env.step(at_clamped)
                not_ = torch.from_numpy(np.asarray(no, dtype=np.float32).flatten()).float().unsqueeze(0).to(device)
                tw = 1.0
                if trust is not None:
                    with torch.no_grad():
                        err = backbone.predict_error(ot, at, not_)
                        if tr_name in ("ema", "multi_step"):
                            tw = float(trust.compute_trust(err, ti).mean())
                        elif tr_name == "ffdc":
                            tw = float(trust.compute_trust(ot, not_, at).mean())
                        elif tr_name == "ensemble":
                            tw = float(trust.compute_trust(ot).mean())
                target = torch.from_numpy(env.action_space.sample()).float().to(device).unsqueeze(0)
                loss = F.mse_loss(at, target) * tw + ewc.penalty()
                pol_opt.zero_grad()
                loss.backward()
                pol_opt.step()
                ot = not_
                if term or trunc:
                    break

        # Evaluate
        eval_rews = []
        for ep in range(5):
            obs, _ = env.reset(seed=ti * 10000 + ep + 90000)
            obs = np.asarray(obs, dtype=np.float32).flatten()
            ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            total_r = 0
            for _ in range(max_steps):
                with torch.no_grad():
                    at = policy(ot)
                at_clamped = np.clip(at.detach().cpu().numpy().flatten(), env.action_space.low, env.action_space.high)
                no, r, term, trunc, _ = env.step(at_clamped)
                total_r += r
                ot = torch.from_numpy(np.asarray(no, dtype=np.float32).flatten()).float().unsqueeze(0).to(device)
                if term or trunc:
                    break
            eval_rews.append(total_r)
        avg_r = float(np.mean(eval_rews))
        all_rewards.append(avg_r)
        env.close()
        print(f"  Task {ti}: {avg_r:.3f}")

    return {
        "benchmark": "kinder",
        "backbone": bb_name,
        "trust": tr_name,
        "task_rewards": all_rewards,
        "avg_reward": float(np.mean(all_rewards)),
    }
    log_reward_chart(run, "chart/learning_curve", all_rewards, f"KinDER {bb_name}+{tr_name}")
    run.log({"avg_reward": final["avg_reward"]})
    run.finish()
    return final


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-benchmark sweep")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["maniskill"],
                        choices=["maniskill", "libero_spatial", "libero_object", "libero_goal", "kinder"],
                        help="Which benchmarks to run")
    parser.add_argument("--backbones", nargs="+", default=None)
    parser.add_argument("--trusts", nargs="+", default=None)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output", default="multi_benchmark_results.json")
    args = parser.parse_args()

    bb_names = args.backbones or list(BACKBONES.keys())
    tr_names = args.trusts or TRUST_NAMES
    results = []

    for bench in args.benchmarks:
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK: {bench}")
        print(f"{'=' * 60}")

        for bb in bb_names:
            for tr in tr_names:
                print(f"\n--- {bb} + {tr} ---")
                t0 = time.time()
                try:
                    if bench == "maniskill":
                        r = run_maniskill(bb, tr, n_ep=args.n_episodes, max_steps=args.max_steps)
                    elif bench.startswith("libero_"):
                        r = run_libero_suite(bb, tr, suite_name=bench, n_ep=args.n_episodes, max_steps=args.max_steps)
                    elif bench == "kinder":
                        r = run_kinder(bb, tr, n_ep=args.n_episodes, max_steps=args.max_steps)
                    else:
                        continue
                    r["time_sec"] = time.time() - t0
                    results.append(r)
                    print(f"  -> avg={r['avg_reward']:.3f} ({r['time_sec']:.0f}s)")
                except Exception:
                    import traceback
                    traceback.print_exc()
                    results.append({"benchmark": bench, "backbone": bb, "trust": tr, "error": traceback.format_exc()})

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"{'Benchmark':<20} {'Backbone':<14} {'Trust':<14} {'Avg':>8}")
    print("-" * 60)
    for r in results:
        if "error" not in r:
            print(f"{r.get('benchmark','?'):<20} {r['backbone']:<14} {r['trust']:<14} {r['avg_reward']:>8.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
