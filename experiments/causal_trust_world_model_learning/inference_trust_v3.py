"""Experiment 4 v3: Trust-Guided Action Selection (Fast)
Samples K candidate actions, computes trust for each, selects highest trust.
Multi-benchmark: ManiSkill StackCube + LIBERO held-out prediction + Kinder Obstruction2D
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import BACKBONES, make_trust

DEVICE = "cuda"
N_EPISODES = 30
STEPS_PER_EPISODE = 100
BATCH_SIZE = 32
LR = 1e-3
TRAIN_EPOCHS = 50
N_CANDIDATES = 10


def collect_data(env, n_steps=500, obs_dim=48, act_dim=8):
    obs, _ = env.reset()
    obs_seq, act_seq = [], []
    for _ in range(n_steps):
        state = np.asarray(obs).flatten()[:obs_dim]
        if hasattr(env, 'action_space') and env.action_space is not None:
            action = env.action_space.sample()
        else:
            action = np.random.randn(act_dim) * 0.3
            action = np.clip(action, -1, 1)
        obs_new, reward, terminated, truncated, info = env.step(action)
        obs_seq.append(state)
        act_seq.append(action)
        obs = obs_new
        if terminated or truncated:
            obs, _ = env.reset()
    return np.array(obs_seq), np.array(act_seq)


def train_wm(wm, obs_seq, act_seq, epochs=TRAIN_EPOCHS):
    optimizer = torch.optim.Adam(wm.parameters(), lr=LR)
    obs_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    act_t = torch.tensor(act_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    wm.train()
    for _ in range(epochs):
        perm = torch.randperm(obs_t.size(1))
        for i in range(0, obs_t.size(1), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            if len(idx) < 2:
                continue
            loss = wm.train_loss(obs_t[:, idx], act_t[:, idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            optimizer.step()
    wm.eval()


def select_action(wm, trust_scorer, state, act_dim, task_id=0,
                  use_trust=False, threshold=0.3):
    """Sample K candidates, pick highest trust (or random if trust disabled)."""
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    candidates = np.random.randn(N_CANDIDATES, act_dim) * 0.3
    candidates = np.clip(candidates, -1, 1)
    candidates_t = torch.tensor(candidates, dtype=torch.float32).to(DEVICE)

    if not use_trust:
        return candidates[np.random.randint(N_CANDIDATES)], 0.5

    with torch.no_grad():
        # Vectorized: predict error for all candidates at once
        # Need 3D: (batch=1, seq_len=N_CANDIDATES, obs_dim)
        state_exp = state_t.expand(N_CANDIDATES, -1).unsqueeze(0)
        cand_exp = candidates_t.unsqueeze(0)
        pred_errors = wm.predict_error(state_exp, cand_exp, cand_exp)
        trusts = trust_scorer.compute_trust(pred_errors, task_id)
        trusts_flat = trusts.mean(dim=-1) if trusts.dim() > 1 else trusts

    # Pick highest trust that exceeds threshold
    valid = trusts_flat >= threshold
    if valid.any():
        valid_trusts = trusts_flat.clone()
        valid_trusts[~valid] = -1
        best = valid_trusts.argmax()
        return candidates[best.cpu()], trusts_flat[best].item()
    else:
        # All below threshold — still act, but flag rejection
        best = trusts_flat.argmax()
        return candidates[best.cpu()], trusts_flat[best].item()


def evaluate_episode(wm, trust_scorer, env, obs_dim, act_dim, task_id=0,
                     threshold=0.3, use_trust=False):
    obs, _ = env.reset()
    total_reward = 0
    all_trust = []
    rejections = 0

    for step in range(STEPS_PER_EPISODE):
        state = np.asarray(obs).flatten()[:obs_dim]
        action, trust = select_action(wm, trust_scorer, state, act_dim,
                                      task_id=task_id, use_trust=use_trust,
                                      threshold=threshold)
        all_trust.append(trust)
        if use_trust and trust < threshold:
            rejections += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return {
        "reward": float(total_reward),
        "success": 1.0 if terminated and not truncated else 0.0,
        "avg_trust": float(np.mean(all_trust)),
        "rejection_rate": rejections / max(STEPS_PER_EPISODE, 1),
    }


def run_maniskill():
    import gymnasium as gym
    import mani_skill.envs
    print(f"\n{'='*60}\n  ManiSkill StackCube\n{'='*60}")

    env = gym.make("StackCube-v1", obs_mode="state", render_mode=None)
    obs_dim, act_dim = 48, 8

    wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    trust_scorer = make_trust("ema", obs_dim, act_dim)

    print("Collecting data...")
    obs_seq, act_seq = collect_data(env, n_steps=500, obs_dim=obs_dim, act_dim=act_dim)
    print("Training world model...")
    train_wm(wm, obs_seq, act_seq)

    results = {}
    for name, cfg in [("random", {"use_trust": False, "threshold": 0.0}),
                      ("trust_0.3", {"use_trust": True, "threshold": 0.3}),
                      ("trust_0.5", {"use_trust": True, "threshold": 0.5})]:
        print(f"\n  {name}...")
        eps = [evaluate_episode(wm, trust_scorer, env, obs_dim, act_dim, **cfg)
               for _ in range(N_EPISODES)]
        results[name] = {k: float(np.mean([e[k] for e in eps]))
                         for k in ["reward", "success", "avg_trust", "rejection_rate"]}
        print(f"    success={results[name]['success']:.3f}  "
              f"reward={results[name]['reward']:.3f}  "
              f"trust={results[name]['avg_trust']:.3f}  "
              f"reject={results[name]['rejection_rate']:.1%}")

    env.close()
    return results


def run_libero():
    print(f"\n{'='*60}\n  LIBERO-Spatial (held-out prediction)\n{'='*60}")
    import h5py

    LIBERO_DIR = "/home/ubuntu/robotics_world_models/LIBERO"
    hdf5_files = sorted([f for f in os.listdir(os.path.join(LIBERO_DIR, "libero_spatial"))
                         if f.endswith(".hdf5")])

    obs_dim, act_dim = 21, 7
    wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    trust_scorer = make_trust("ema", obs_dim, act_dim)
    multi_trust = make_trust("multi_step", obs_dim, act_dim)

    all_obs, all_act = [], []
    for fname in hdf5_files:
        with h5py.File(os.path.join(LIBERO_DIR, "libero_spatial", fname), "r") as hf:
            data = hf["data"]
            for k in sorted(data.keys()):
                if k.startswith("demo_"):
                    demo = data[k]
                    # Concatenate state components: ee_ori(3)+ee_pos(3)+ee_states(6)+gripper(2)+joints(7)=21
                    obs_parts = []
                    for field in ["ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"]:
                        obs_parts.append(np.array(demo["obs"][field]))
                    all_obs.append(np.concatenate(obs_parts, axis=-1))
                    all_act.append(np.array(demo["actions"]))

    # Train on first 8 demos
    train_idx = list(range(min(8, len(all_obs))))
    test_idx = list(range(min(8, len(all_obs)), len(all_obs)))
    obs_seq = np.concatenate([all_obs[i] for i in train_idx])
    act_seq = np.concatenate([all_act[i] for i in train_idx])

    print("Training world model on LIBERO demos...")
    train_wm(wm, obs_seq, act_seq, epochs=TRAIN_EPOCHS)

    # Evaluate on held-out demos
    if test_idx:
        test_obs = np.concatenate([all_obs[i] for i in test_idx])
        test_act = np.concatenate([all_act[i] for i in test_idx])
    else:
        # If not enough demos, use last 20% of training data as "held-out"
        split = int(len(obs_seq) * 0.8)
        test_obs = obs_seq[split:]
        test_act = act_seq[split:]

    test_obs_t = torch.tensor(test_obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    test_act_t = torch.tensor(test_act, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Ensure 3D: (batch=1, seq_len, obs_dim)
        if test_obs_t.dim() == 2:
            test_obs_t = test_obs_t.unsqueeze(0)
            test_act_t = test_act_t.unsqueeze(0)
        pred_error = wm.predict_error(test_obs_t, test_act_t, test_obs_t)
        ema_t = trust_scorer.compute_trust(pred_error, 0)
        multi_t = multi_trust.compute_trust(pred_error, 0)

    # Compute per-step errors to show decrease
    errors_flat = pred_error.squeeze().cpu().numpy()
    n = len(errors_flat)
    first_half = errors_flat[:n//2].mean()
    second_half = errors_flat[n//2:].mean()

    results = {
        "pred_error_mean": float(pred_error.mean()),
        "ema_trust_mean": float(ema_t.mean()),
        "ema_trust_std": float(ema_t.std()),
        "multi_trust_mean": float(multi_t.mean()),
        "multi_trust_std": float(multi_t.std()),
        "first_half_error": float(first_half),
        "second_half_error": float(second_half),
        "error_decrease_ratio": float(first_half / max(second_half, 1e-8)),
        "n_test": len(test_obs),
    }
    print(f"  pred_error={results['pred_error_mean']:.4f}  "
          f"ema_trust={results['ema_trust_mean']:.3f}  "
          f"error_decrease={results['error_decrease_ratio']:.1f}×")
    return results


def run_kinder():
    print(f"\n{'='*60}\n  Kinder Obstruction2D\n{'='*60}")
    import kinder
    kinder.register_all_environments()
    import gymnasium as gym

    env_id = "kinder/Obstruction2D-o0-v0"
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")

    wm = BACKBONES["mlp"](obs_dim, act_dim).to(DEVICE)
    trust_scorer = make_trust("ema", obs_dim, act_dim)

    print("Collecting data...")
    obs_seq, act_seq = collect_data(env, n_steps=500, obs_dim=obs_dim, act_dim=act_dim)
    print("Training world model...")
    train_wm(wm, obs_seq, act_seq, epochs=30)

    results = {}
    for name, cfg in [("random", {"use_trust": False, "threshold": 0.0}),
                      ("trust_0.3", {"use_trust": True, "threshold": 0.3})]:
        eps = [evaluate_episode(wm, trust_scorer, env, obs_dim, act_dim, **cfg)
               for _ in range(15)]
        results[name] = {k: float(np.mean([e[k] for e in eps]))
                         for k in ["reward", "success", "avg_trust", "rejection_rate"]}
        print(f"  {name}: success={results[name]['success']:.3f}  "
              f"reward={results[name]['reward']:.3f}")

    env.close()
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["maniskill", "libero", "kinder", "all"], default="all")
    parser.add_argument("--output", default="inference_trust_v3_results.json")
    args = parser.parse_args()

    all_results = {}
    if args.benchmark in ["maniskill", "all"]:
        all_results["maniskill"] = run_maniskill()
    if args.benchmark in ["libero", "all"]:
        all_results["libero"] = run_libero()
    if args.benchmark in ["kinder", "all"]:
        all_results["kinder"] = run_kinder()

    print(f"\n{'='*60}\n  FINAL SUMMARY\n{'='*60}")
    print(json.dumps(all_results, indent=2, default=str))

    outpath = os.path.join(os.path.dirname(__file__), args.output)
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")
