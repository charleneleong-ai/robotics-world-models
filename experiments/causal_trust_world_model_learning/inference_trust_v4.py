"""Experiment 4 v4: Trust-Guided Action Selection (Multi-Benchmark)
Samples K candidates within env action space, picks highest trust.
ManiSkill StackCube + LIBERO-Spatial + Kinder Obstruction2D
"""

import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from full_backbone_sweep import BACKBONES, make_trust

DEVICE = "cuda"
N_EP = 30
MAX_STEPS = 100
BATCH = 32
LR = 1e-3
TRAIN_EP = 50
K = 10


def train_wm(wm, obs_seq, act_seq, epochs=TRAIN_EP):
    opt = torch.optim.Adam(wm.parameters(), lr=LR)
    o = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    a = torch.tensor(act_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    wm.train()
    for _ in range(epochs):
        perm = torch.randperm(o.size(1))
        for i in range(0, o.size(1), BATCH):
            idx = perm[i:i+BATCH]
            if len(idx) < 2: continue
            loss = wm.train_loss(o[:, idx], a[:, idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            opt.step()
    wm.eval()


def sample_actions(env, n):
    """Sample n actions respecting the env's action space."""
    low = env.action_space.low
    high = env.action_space.high
    actions = np.random.uniform(low, high, size=(n, len(low))).astype(np.float32)
    return actions


def select_action(wm, scorer, state, env, task_id=0, use_trust=False, threshold=0.3):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    cands = sample_actions(env, K)
    cands_t = torch.tensor(cands, dtype=torch.float32).to(DEVICE)

    if not use_trust:
        return cands[np.random.randint(K)], 0.5, False

    with torch.no_grad():
        # predict_error expects (B, obs_dim), not (B, T, obs_dim)
        # Process each candidate individually since RSSM uses GRU state
        pe_list = []
        for i in range(K):
            s_i = state_t  # (1, obs_dim)
            c_i = cands_t[i:i+1]  # (1, act_dim)
            pe_i = wm.predict_error(s_i, c_i, s_i)  # (1, obs_dim)
            pe_list.append(pe_i.mean().item())
        pe_vals = torch.tensor(pe_list, device=DEVICE)
        # Higher trust = lower error
        tr_vals = torch.exp(-pe_vals / (pe_vals.mean() + 1e-8)).clamp(0, 1)
        best = tr_vals.argmax().item()
        trust_val = tr_vals[best].item()
        action = cands[best].copy()
        # Clip to action space bounds
        action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
        return action, trust_val, trust_val < threshold

    best = tr_flat.argmax()
    trust_val = tr_flat[best].item()
    return cands[best.cpu()], trust_val, trust_val < threshold


def evaluate(wm, scorer, env, obs_dim, task_id=0, use_trust=False, threshold=0.3):
    obs, _ = env.reset()
    total_r, trusts, rejections = 0.0, [], 0
    for _ in range(MAX_STEPS):
        state = np.asarray(obs).flatten()[:obs_dim]
        action, trust, rejected = select_action(wm, scorer, state, env,
                                                 task_id=task_id, use_trust=use_trust,
                                                 threshold=threshold)
        trusts.append(trust)
        if rejected: rejections += 1
        obs, r, term, trunc, _ = env.step(np.asarray(action, dtype=np.float32))
        total_r += r
        if term or trunc: break
    return {
        "reward": float(total_r),
        "success": 1.0 if (term and not trunc) else 0.0,
        "avg_trust": float(np.mean(trusts)),
        "rejection_rate": rejections / max(MAX_STEPS, 1),
    }


def run_maniskill():
    import gymnasium as gym
    import mani_skill.envs
    print(f"\n{'='*60}\n  ManiSkill StackCube\n{'='*60}")
    env = gym.make("StackCube-v1", obs_mode="state", render_mode=None)
    obs_dim, act_dim = 48, 8

    wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    scorer = make_trust("ema", obs_dim, act_dim)

    # Collect
    obs, _ = env.reset()
    obs_s, act_s = [], []
    for _ in range(500):
        s = np.asarray(obs).flatten()[:obs_dim]
        a = sample_actions(env, 1)[0]
        obs, r, t, tr, _ = env.step(a)
        obs_s.append(s); act_s.append(a)
        if t or tr: obs, _ = env.reset()

    train_wm(wm, np.array(obs_s), np.array(act_s))

    results = {}
    for nm, cfg in [("random", dict(use_trust=False)),
                     ("trust_0.3", dict(use_trust=True, threshold=0.3)),
                     ("trust_0.5", dict(use_trust=True, threshold=0.5))]:
        eps = [evaluate(wm, scorer, env, obs_dim, **cfg) for _ in range(N_EP)]
        results[nm] = {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}
        print(f"  {nm}: success={results[nm]['success']:.3f} reward={results[nm]['reward']:.3f} trust={results[nm]['avg_trust']:.3f} reject={results[nm]['rejection_rate']:.1%}")
    env.close()
    return results


def run_libero():
    print(f"\n{'='*60}\n  LIBERO-Spatial\n{'='*60}")
    import h5py
    DIR = "/home/ubuntu/robotics_world_models/LIBERO/libero_spatial"
    files = sorted(f for f in os.listdir(DIR) if f.endswith(".hdf5"))

    obs_dim, act_dim = 21, 7
    wm = BACKBONES["rssm"](obs_dim, act_dim).to(DEVICE)
    scorer = make_trust("ema", obs_dim, act_dim)
    multi = make_trust("multi_step", obs_dim, act_dim)

    all_o, all_a = [], []
    for fn in files:
        with h5py.File(os.path.join(DIR, fn), "r") as hf:
            for k in sorted(hf["data"].keys()):
                if not k.startswith("demo_"): continue
                demo = hf["data"][k]
                parts = [np.array(demo["obs"][f]) for f in
                         ["ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"]]
                all_o.append(np.concatenate(parts, axis=-1))
                all_a.append(np.array(demo["actions"]))

    # Train on first 8, test on rest
    n = min(8, len(all_o))
    obs_seq = np.concatenate(all_o[:n])
    act_seq = np.concatenate(all_a[:n])
    train_wm(wm, obs_seq, act_seq)

    test_idx = list(range(n, len(all_o))) if n < len(all_o) else list(range(int(len(obs_seq)*0.8), len(obs_seq)))
    if n < len(all_o):
        test_obs = np.concatenate([all_o[i] for i in test_idx])
        test_act = np.concatenate([all_a[i] for i in test_idx])
    else:
        split = int(len(obs_seq) * 0.8)
        test_obs = obs_seq[split:]
        test_act = act_seq[split:]

    to = torch.tensor(test_obs, dtype=torch.float32).to(DEVICE)
    ta = torch.tensor(test_act, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        # predict_error needs (B, obs_dim), process in batches
        n = to.size(0)
        pe_list = []
        bs = 32
        for j in range(0, n, bs):
            chunk = min(bs, n - j)
            pe_i = wm.predict_error(to[j:j+chunk], ta[j:j+chunk], to[j:j+chunk])
            pe_list.append(pe_i)
        pe = torch.cat(pe_list, dim=0)
        ema_t = scorer.compute_trust(pe, 0)
        mul_t = multi.compute_trust(pe, 0)
    ef = pe.squeeze().cpu().numpy()
    half = len(ef) // 2
    results = {
        "pred_error": float(pe.mean()),
        "ema_trust": float(ema_t.mean()),
        "ema_std": float(ema_t.std()),
        "multi_trust": float(mul_t.mean()),
        "multi_std": float(mul_t.std()),
        "first_half_err": float(ef[:half].mean()),
        "second_half_err": float(ef[half:].mean()),
        "error_decrease": float(ef[:half].mean() / max(ef[half:].mean(), 1e-8)),
    }
    print(f"  pred_error={results['pred_error']:.4f} ema_trust={results['ema_trust']:.3f} "
          f"decrease={results['error_decrease']:.1f}x")
    return results


def run_kinder():
    print(f"\n{'='*60}\n  Kinder Obstruction2D\n{'='*60}")
    import kinder; kinder.register_all_environments()
    import gymnasium as gym
    env = gym.make("kinder/Obstruction2D-o0-v0")
    obs_dim = env.observation_space.shape[0]

    wm = BACKBONES["mlp"](obs_dim, env.action_space.shape[0]).to(DEVICE)
    scorer = make_trust("ema", obs_dim, env.action_space.shape[0])

    # Collect
    obs, _ = env.reset()
    obs_s, act_s = [], []
    for _ in range(500):
        s = np.asarray(obs).flatten()[:obs_dim]
        a = env.action_space.sample()
        obs, r, t, tr, _ = env.step(a)
        obs_s.append(s); act_s.append(a)
        if t or tr: obs, _ = env.reset()

    train_wm(wm, np.array(obs_s), np.array(act_s), epochs=30)

    results = {}
    for nm, cfg in [("random", dict(use_trust=False)),
                     ("trust_0.3", dict(use_trust=True, threshold=0.3))]:
        eps = [evaluate(wm, scorer, env, obs_dim, **cfg) for _ in range(15)]
        results[nm] = {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}
        print(f"  {nm}: success={results[nm]['success']:.3f} reward={results[nm]['reward']:.3f} trust={results[nm]['avg_trust']:.3f}")
    env.close()
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["maniskill", "libero", "kinder", "all"], default="all")
    p.add_argument("--output", default="inference_trust_v4_results.json")
    args = p.parse_args()

    R = {}
    if args.benchmark in ("maniskill", "all"): R["maniskill"] = run_maniskill()
    if args.benchmark in ("libero", "all"): R["libero"] = run_libero()
    if args.benchmark in ("kinder", "all"): R["kinder"] = run_kinder()

    print(f"\n{'='*60}\n  FINAL\n{'='*60}")
    print(json.dumps(R, indent=2, default=str))

    out = os.path.join(os.path.dirname(__file__), args.output)
    with open(out, "w") as f: json.dump(R, f, indent=2, default=str)
    print(f"Saved to {out}")
