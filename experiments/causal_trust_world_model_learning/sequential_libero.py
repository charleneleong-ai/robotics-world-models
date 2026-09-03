"""Sequential LIBERO experiment: single world model learns tasks 1-10.
Logs prediction error per task to show world model improvement curve."""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import BACKBONES, make_trust

# Config
DEVICE = "cuda"
SUITE = "libero_spatial"
N_TASKS = 10
PRETRAIN_EPOCHS = 30
FINETUNE_EPISODES = 5
STEPS_PER_EPISODE = 50
BATCH_SIZE = 32
LR = 1e-3


def load_demos_hdf5(suite, task_idx, n_demos=5):
    """Load demos directly from HDF5 files."""
    demo_dir = f"/home/ubuntu/robotics_world_models/LIBERO/{suite}"
    files = sorted([f for f in os.listdir(demo_dir) if f.endswith(".hdf5")])
    
    if task_idx >= len(files):
        return None, None
    
    filepath = os.path.join(demo_dir, files[task_idx])
    f = h5py.File(filepath, "r")
    
    all_obs, all_actions = [], []
    demo_keys = sorted(f["data"].keys())
    
    for i in range(min(n_demos, len(demo_keys))):
        demo = f["data"][demo_keys[i]]
        
        # Extract state features (exclude images)
        obs = demo["obs"]
        state_parts = []
        for k in sorted(obs.keys()):
            if k not in ["agentview_rgb", "eye_in_hand_rgb"]:
                state_parts.append(obs[k][:])
        
        state = np.concatenate(state_parts, axis=-1)  # (T, obs_dim)
        actions = demo["actions"][:]  # (T, act_dim)
        
        all_obs.append(state)
        all_actions.append(actions)
    
    f.close()
    
    if not all_obs:
        return None, None
    
    # Pad to same length
    max_len = max(o.shape[0] for o in all_obs)
    obs_dim = all_obs[0].shape[-1]
    act_dim = all_actions[0].shape[-1]
    
    obs_padded = np.zeros((len(all_obs), max_len, obs_dim))
    act_padded = np.zeros((len(all_actions), max_len, act_dim))
    
    for i, (o, a) in enumerate(zip(all_obs, all_actions)):
        T = o.shape[0]
        obs_padded[i, :T] = o
        act_padded[i, :T] = a
    
    return obs_padded, act_padded


def pretrain_world_model(wm, obs_seq, act_seq, epochs=30):
    optimizer = torch.optim.Adam(wm.parameters(), lr=LR)
    obs_t = torch.tensor(obs_seq, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(act_seq, dtype=torch.float32).to(DEVICE)

    wm.train()
    total_loss = 0
    n_batches = 0
    for epoch in range(epochs):
        perm = torch.randperm(obs_t.size(0))
        for i in range(0, obs_t.size(0), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:
                continue
            batch_obs = obs_t[idx]
            batch_act = act_t[idx]
            loss = wm.train_loss(batch_obs, batch_act)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def make_env(suite, task_idx):
    """Create LIBERO environment for a specific task."""
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["DISPLAY"] = ""
    try:
        from libero.lifelong.benchmark import LIBEROSuiteBenchmark
        benchmark = LIBEROSuiteBenchmark(suite)
        task_obj = benchmark.get_task(task_idx)
        env = task_obj.env
        return env
    except Exception as e:
        print(f"  Env creation error: {e}")
        return None


def flatten_obs(obs_dict):
    exclude = {"agentview_rgb", "eye_in_hand_rgb"}
    vals = []
    for k in sorted(obs_dict.keys()):
        if k not in exclude:
            v = obs_dict[k]
            if hasattr(v, "numpy"):
                v = v.numpy()
            if isinstance(v, np.ndarray):
                vals.append(v.flatten())
    return np.concatenate(vals) if vals else np.array([])


def evaluate_task(wm, trust_scorer, task_idx, n_episodes=5):
    """Evaluate on a task, return reward, trust stats, prediction error."""
    env = make_env(SUITE, task_idx)
    if env is None:
        return -1.0, 0.0, 1.0

    wm.eval()
    total_reward = 0
    all_trust = []
    all_errors = []

    for ep in range(n_episodes):
        try:
            obs = env.reset()
        except Exception:
            obs, _ = env.reset()
        state = flatten_obs(obs)
        ep_reward = 0

        for step in range(STEPS_PER_EPISODE):
            obs_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_error = wm.predict_error(
                    obs_t,
                    torch.zeros(1, 7, device=DEVICE),
                    obs_t,
                )
                err_val = pred_error.mean().item()
                all_errors.append(err_val)

                if trust_scorer is not None:
                    trust = trust_scorer.compute_trust(pred_error, task_idx).mean().item()
                    all_trust.append(trust)

                # Simple policy: random action with small magnitude
                action = np.random.randn(7) * 0.3
                action = np.clip(action, -1, 1)

            try:
                obs_new, reward, terminated, truncated, info = env.step(action)
            except Exception:
                obs_new, reward, done, info = env.step(action)
            new_state = flatten_obs(obs_new)

            state = new_state
            ep_reward += reward

        total_reward += ep_reward

    avg_reward = total_reward / n_episodes
    avg_trust = np.mean(all_trust) if all_trust else 1.0
    avg_error = np.mean(all_errors) if all_errors else 0.0

    return avg_reward, avg_trust, avg_error


def run_sequential(backbone_name, trust_name):
    """Run sequential task learning, log world model state after each task."""
    print(f"\n{'='*60}")
    print(f"  Sequential: {backbone_name} + {trust_name}")
    print(f"{'='*60}")

    obs_dim = 21  # LIBERO state dim (ee_pos + ee_ori + ee_states + gripper + joints)
    act_dim = 7
    wm = BACKBONES[backbone_name](obs_dim, act_dim).to(DEVICE)

    trust_scorer = None
    if trust_name != "none":
        trust_scorer = make_trust(trust_name, obs_dim, act_dim)

    results = []
    wm_errors_over_time = []

    for task_idx in range(N_TASKS):
        t0 = time.time()
        print(f"\n--- Task {task_idx + 1}/{N_TASKS} ---")

        # Load demos
        obs_seq, act_seq = load_demos_hdf5(SUITE, task_idx, n_demos=5)
        if obs_seq is None:
            print(f"  No demos for task {task_idx}, skipping")
            continue

        print(f"  Demos: {obs_seq.shape[0]} episodes, max len {obs_seq.shape[1]}, obs_dim={obs_seq.shape[2]}")

        # Measure prediction error BEFORE pretraining on this task
        pre_error = None
        if task_idx > 0:
            pre_obs = torch.tensor(obs_seq[:2, :10], dtype=torch.float32).to(DEVICE)
            pre_act = torch.tensor(act_seq[:2, :10], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                pre_error = wm.train_loss(pre_obs, pre_act).item()
            print(f"  Pre-train error (generalization): {pre_error:.4f}")

        # Pretrain on this task
        loss = pretrain_world_model(wm, obs_seq, act_seq, epochs=PRETRAIN_EPOCHS)
        print(f"  Post-train loss: {loss:.4f}")

        # Measure prediction error AFTER pretraining
        post_obs = torch.tensor(obs_seq[:2, :10], dtype=torch.float32).to(DEVICE)
        post_act = torch.tensor(act_seq[:2, :10], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            post_error = wm.train_loss(post_obs, post_act).item()
        print(f"  Post-train error (on-task): {post_error:.4f}")

        # Evaluate on this task
        reward, trust, eval_error = evaluate_task(wm, trust_scorer, task_idx, n_episodes=FINETUNE_EPISODES)
        print(f"  Eval reward: {reward:.3f}, trust: {trust:.3f}, error: {eval_error:.4f}")

        # Evaluate on ALL previous tasks (forgetting check)
        prev_rewards = []
        prev_errors = []
        for prev_idx in range(task_idx):
            prev_r, _, prev_e = evaluate_task(wm, trust_scorer, prev_idx, n_episodes=2)
            prev_rewards.append(prev_r)
            prev_errors.append(prev_e)

        if prev_rewards:
            avg_prev_r = np.mean(prev_rewards)
            avg_prev_e = np.mean(prev_errors)
            print(f"  Prev tasks: reward={avg_prev_r:.3f}, error={avg_prev_e:.4f}")
        else:
            avg_prev_r = 0.0
            avg_prev_e = 0.0

        results.append({
            "task": task_idx,
            "reward": reward,
            "trust": trust,
            "eval_error": eval_error,
            "pre_generalization_error": pre_error,
            "on_task_error_after_train": post_error,
            "prev_avg_reward": avg_prev_r,
            "prev_avg_error": avg_prev_e,
            "time": time.time() - t0,
        })

        wm_errors_over_time.append({
            "task": task_idx,
            "pre_error": pre_error,
            "post_error": post_error,
            "on_task_error": eval_error,
            "prev_error": avg_prev_e,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {backbone_name} + {trust_name}")
    print(f"{'='*60}")
    for r in results:
        print(f"  Task {r['task'] + 1}: reward={r['reward']:.3f}, "
              f"error={r['eval_error']:.4f}, prev_avg={r['prev_avg_reward']:.3f}")

    return results, wm_errors_over_time


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="rssm")
    parser.add_argument("--trust", default="ema")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results, errors = run_sequential(args.backbone, args.trust)

    outname = args.output or f"sequential_{args.backbone}_{args.trust}.json"
    outpath = os.path.join(os.path.dirname(__file__), outname)
    with open(outpath, "w") as f:
        json.dump({"results": results, "wm_errors": errors}, f, indent=2)
    print(f"\nSaved to {outpath}")
