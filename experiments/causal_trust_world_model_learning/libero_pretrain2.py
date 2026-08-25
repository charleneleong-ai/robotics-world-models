"""Pretrain RSSM on LIBERO expert demonstrations."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from huggingface_hub import snapshot_download
import h5py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from rssm_world_model import WorldModel


def load_libero_demos(dataset="libero_spatial", max_episodes=50):
    """Load LIBERO demonstrations from HuggingFace cache."""
    import os
    cache_dir = "/home/ubuntu/.cache/huggingface/hub/datasets--yifengzhu-hf--LIBERO-datasets/snapshots/f13aa24a3da8c43c7225569f28c562979fa0e35a"
    dataset_dir = Path(cache_dir) / dataset

    all_obs, all_actions = [], []
    hdf5_files = sorted(list(dataset_dir.glob("*.hdf5")))
    print("Found %d HDF5 files" % len(hdf5_files))

    for hdf5_path in hdf5_files[:max_episodes]:
        try:
            with h5py.File(hdf5_path, "r") as f:
                data = f["data"]
                for demo_key in list(data.keys()):
                    demo = data[demo_key]
                    actions = np.array(demo["actions"])  # (T, 7)
                    robot_states = np.array(demo["robot_states"])  # (T, 9)
                    states = np.array(demo["states"])  # (T, 92)
                    # Use states as observation (92-dim)
                    all_obs.append(states)
                    all_actions.append(actions)
        except Exception as e:
            print("  Error loading %s: %s" % (hdf5_path.name, str(e)))
            continue

    if all_obs:
        obs = np.concatenate(all_obs)
        actions = np.concatenate(all_actions)
        print("Loaded: obs=%s actions=%s" % (obs.shape, actions.shape))
        return obs, actions
    else:
        print("No data loaded!")
        return None, None


def pretrain_rssm(obs, actions, epochs=100, device="cuda"):
    """Pretrain RSSM on expert demonstrations."""
    obs_dim = obs.shape[1]
    action_dim = actions.shape[1]
    print("Pretraining RSSM: obs_dim=%d, action_dim=%d" % (obs_dim, action_dim))

    model = WorldModel(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Split into sequences
    seq_len = 50
    n_seqs = len(obs) // seq_len
    obs_seq = obs[:n_seqs * seq_len].reshape(n_seqs, seq_len, obs_dim)
    act_seq = actions[:n_seqs * seq_len].reshape(n_seqs, seq_len, action_dim)

    obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=device)
    act_t = torch.tensor(act_seq, dtype=torch.float32, device=device)
    rew_t = torch.zeros(n_seqs, seq_len, device=device)
    done_t = torch.zeros(n_seqs, seq_len, device=device)

    for epoch in range(epochs):
        metrics = model.training_step(obs_t, act_t, rew_t, done_t)
        total_loss = metrics["total_loss"]
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print("  Epoch %d: loss=%.4f obs_loss=%.4f" %
                  (epoch + 1, total_loss.item(), metrics["obs_loss"]))

    return model


def train_policy(rssm, obs, actions, n_epochs=200, device="cuda"):
    """Train policy using pretrained RSSM."""
    state_dim = 512 + 32 * 32
    action_dim = actions.shape[1]

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, action_dim), nn.Tanh(),
            )
        def forward(self, state):
            return self.net(state)

    actor = Actor().to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.tensor(actions, dtype=torch.float32, device=device)
    rssm.eval()

    for epoch in range(n_epochs):
        total_loss = 0
        for _ in range(30):
            idx = np.random.randint(len(obs))
            h, z = rssm.rssm.initial_state(1, device)
            with torch.no_grad():
                result = rssm.rssm.observe_step(h, z, act_t[idx:idx+1], obs_t[idx:idx+1])
                h, z = result["h"], result["z"]

            log_probs, rewards = [], []
            for t in range(50):
                state = torch.cat([h, z], dim=-1)
                action_mean = actor(state)
                dist = torch.distributions.Normal(action_mean, 0.1)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                with torch.no_grad():
                    result = rssm.rssm.imagine_step(h, z, action)
                    h, z = result["h"], result["z"]
                    r = result["reward_pred"].squeeze()
                log_probs.append(log_prob)
                rewards.append(r)

            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + 0.99 * R
                returns.insert(0, R)
            returns_t = torch.stack([r.detach() for r in returns])
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            loss = -(torch.stack(log_probs) * returns_t).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print("  Epoch %d: actor_loss=%.4f" % (epoch + 1, total_loss / 30))

    return actor


def evaluate(actor, rssm, device="cuda"):
    """Evaluate on real LIBERO."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bm = benchmark.get_benchmark_dict()
    spatial_bm = bm["libero_spatial"]()

    print("\nEvaluating on real LIBERO...")
    random_rewards, trained_rewards = [], []

    for t in range(3):
        bddl = spatial_bm.get_task_bddl_file_path(t)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)

        # Random policy
        ep_rews = []
        for ep in range(5):
            result = env.reset()
            obs = result[0] if isinstance(result, tuple) else result
            ep_r = 0
            for step in range(100):
                a = np.random.randn(env.env.action_dim) * 0.5
                result = env.step(np.clip(a, -1, 1))
                obs, r, d = result[0], result[1], result[2]
                ep_r += r
                if d: break
            ep_rews.append(ep_r)
        random_rewards.append(np.mean(ep_rews))

        # Trained policy
        ep_rews = []
        for ep in range(5):
            result = env.reset()
            obs = result[0] if isinstance(result, tuple) else result
            ep_r = 0
            h, z = rssm.rssm.initial_state(1, device)
            for step in range(100):
                # Get full state from LIBERO
                state = np.concatenate([obs["robot0_proprio-state"], obs["object-state"]])
                # Encode through RSSM
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    r = rssm.rssm.observe_step(h, z, torch.zeros(1, 7, device=device), state_t)
                    h, z = r["h"], r["z"]
                    rssm_state = torch.cat([h, z], dim=-1)
                    action = actor(rssm_state).squeeze(0).cpu().numpy()
                result = env.step(np.clip(action, -1, 1))
                obs, rew, d = result[0], result[1], result[2]
                ep_r += rew
                if d: break
            ep_rews.append(ep_r)
        trained_rewards.append(np.mean(ep_rews))
        env.close()

        print("  Task %d: random=%.4f trained=%.4f" % (t, random_rewards[-1], trained_rewards[-1]))

    print("\nSummary:")
    print("  Random avg: %.4f" % np.mean(random_rewards))
    print("  Trained avg: %.4f" % np.mean(trained_rewards))
    return np.mean(random_rewards), np.mean(trained_rewards)


def main():
    import typer
    app = typer.Typer()

    @app.command()
    def run(
        dataset: str = typer.Option("libero_spatial"),
        max_episodes: int = typer.Option(50),
        pretrain_epochs: int = typer.Option(100),
        policy_epochs: int = typer.Option(200),
        device: str = typer.Option("cuda"),
        save_dir: str = typer.Option("trained_models/libero_pretrained"),
    ):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        print("Step 1: Loading LIBERO demonstrations...")
        obs, actions = load_libero_demos(dataset, max_episodes)
        if obs is None:
            return

        print("\nStep 2: Pretraining RSSM on demonstrations...")
        rssm = pretrain_rssm(obs, actions, pretrain_epochs, device)
        torch.save(rssm.state_dict(), "%s/rssm_pretrained.pt" % save_dir)
        print("Saved pretrained RSSM")

        print("\nStep 3: Training policy with pretrained RSSM...")
        actor = train_policy(rssm, obs, actions, policy_epochs, device)
        torch.save(actor.state_dict(), "%s/actor_pretrained.pt" % save_dir)
        print("Saved actor")

        print("\nStep 4: Evaluating on real LIBERO...")
        random_r, trained_r = evaluate(actor, rssm, device)

        results = {"random": float(random_r), "trained": float(trained_r)}
        import json
        with open("%s/libero_pretrained_results.json" % save_dir, "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to %s/libero_pretrained_results.json" % save_dir)

    app()

if __name__ == "__main__":
    main()
