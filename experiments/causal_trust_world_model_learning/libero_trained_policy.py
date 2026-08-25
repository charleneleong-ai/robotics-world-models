"""Train a policy on LIBERO using the RSSM world model (DreamerV3-style)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import sys
sys.path.insert(0, str(Path(__file__).parent))
from rssm_world_model import WorldModel


class Actor(nn.Module):
    """Policy network that outputs actions from RSSM state."""
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )
    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """Value network that predicts expected return from RSSM state."""
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, state):
        return self.net(state)


def collect_real_data(benchmark_obj, n_episodes=20, max_steps=100):
    """Collect real trajectories from LIBERO."""
    all_obs, all_actions, all_rewards, all_dones = [], [], [], []
    for ep in range(n_episodes):
        task_idx = ep % len(benchmark_obj.get_task_names())
        bddl = benchmark_obj.get_task_bddl_file_path(task_idx)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        for step in range(max_steps):
            state = np.concatenate([obs["robot0_proprio-state"], obs["object-state"]])
            action = np.random.randn(env.env.action_dim) * 0.3
            action = np.clip(action, -1, 1)
            all_obs.append(state)
            all_actions.append(action)
            result = env.step(action)
            next_obs, reward, done = result[0], result[1], result[2]
            all_rewards.append(reward)
            all_dones.append(float(done))
            obs = next_obs
            if done:
                break
        env.close()
        if (ep + 1) % 10 == 0:
            print("  Collected %d/%d episodes" % (ep + 1, n_episodes))
    return (np.array(all_obs), np.array(all_actions),
            np.array(all_rewards), np.array(all_dones))


def train_with_world_model(rssm, real_obs, real_actions, real_rewards,
                           n_epochs=100, n_imagined=50, horizon=50, device="cuda"):
    """Train actor-critic using imagined rollouts from the RSSM."""
    obs_dim = real_obs.shape[1]
    action_dim = real_actions.shape[1]
    state_dim = 512 + 32 * 32  # deterministic + stochastic

    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

    # Prepare real data as tensors
    obs_t = torch.tensor(real_obs, dtype=torch.float32, device=device)
    act_t = torch.tensor(real_actions, dtype=torch.float32, device=device)

    rssm.eval()
    gamma = 0.99
    lam = 0.95

    for epoch in range(n_epochs):
        # --- Imagine trajectories ---
        imagined_returns = []
        value_targets = []

        for _ in range(n_imagined):
            # Start from a random real state
            idx = np.random.randint(len(real_obs))
            h, z = rssm.rssm.initial_state(1, device)
            # Observe real state to get initial h, z
            with torch.no_grad():
                result = rssm.rssm.observe_step(
                    h, z, act_t[idx:idx+1], obs_t[idx:idx+1]
                )
                h, z = result["h"], result["z"]

            # Imagine forward
            rewards = []
            values = []
            for t in range(horizon):
                state = torch.cat([h, z], dim=-1)
                action = actor(state)
                with torch.no_grad():
                    result = rssm.rssm.imagine_step(h, z, action)
                    h, z = result["h"], result["z"]
                    r = result["reward_pred"].squeeze()
                    v = critic(torch.cat([h, z], dim=-1)).squeeze()
                rewards.append(r)
                values.append(v)

            # Compute returns (GAE)
            returns = []
            R = values[-1].detach()
            for t in reversed(range(horizon)):
                R = rewards[t] + gamma * R
                returns.insert(0, R)

            imagined_returns.extend(returns)
            value_targets.extend([v.item() for v in values])

        # --- Update critic ---
        imagined_returns_t = torch.tensor(imagined_returns, dtype=torch.float32, device=device)
        value_targets_t = torch.tensor(value_targets, dtype=torch.float32, device=device)

        # Re-imagine to get states for gradient
        critic_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for _ in range(20):
            idx = np.random.randint(len(real_obs))
            h, z = rssm.rssm.initial_state(1, device)
            with torch.no_grad():
                result = rssm.rssm.observe_step(h, z, act_t[idx:idx+1], obs_t[idx:idx+1])
                h, z = result["h"], result["z"]

            states = []
            for t in range(horizon):
                state = torch.cat([h, z], dim=-1)
                action = actor(state).detach()
                with torch.no_grad():
                    result = rssm.rssm.imagine_step(h, z, action)
                    h, z = result["h"], result["z"]
                states.append(torch.cat([h, z], dim=-1))

            states_t = torch.stack(states).squeeze(1)
            values_pred = critic(states_t).squeeze(-1)
            target_returns = imagined_returns_t[:horizon]
            critic_loss = F.mse_loss(values_pred, target_returns)

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

        # --- Update actor (policy gradient) ---
        actor_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for _ in range(20):
            idx = np.random.randint(len(real_obs))
            h, z = rssm.rssm.initial_state(1, device)
            with torch.no_grad():
                result = rssm.rssm.observe_step(h, z, act_t[idx:idx+1], obs_t[idx:idx+1])
                h, z = result["h"], result["z"]

            log_probs = []
            entropies = []
            values = []
            for t in range(horizon):
                state = torch.cat([h, z], dim=-1)
                action_mean = actor(state)
                # Add noise for exploration
                dist = torch.distributions.Normal(action_mean, 0.1)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                with torch.no_grad():
                    result = rssm.rssm.imagine_step(h, z, action)
                    h, z = result["h"], result["z"]
                    v = critic(torch.cat([h, z], dim=-1)).squeeze()
                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(v)

            # Simple policy gradient
            returns_list = []
            R = values[-1].detach()
            for t in reversed(range(horizon)):
                R = rewards[t] + gamma * R if t < len(rewards) else R
                returns_list.insert(0, R)

            returns_t = torch.stack([r.detach() for r in returns_list[:horizon]])
            values_t = torch.stack(values)
            advantages = returns_t - values_t.detach()

            actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean() - 0.01 * torch.stack(entropies).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

        if (epoch + 1) % 20 == 0:
            print("  Epoch %d: actor_loss=%.4f critic_loss=%.4f" %
                  (epoch + 1, actor_loss.item(), critic_loss.item()))

    return actor


def evaluate_policy(actor, rssm, benchmark_obj, n_tasks=5, n_episodes=5, device="cuda"):
    """Evaluate trained policy on real LIBERO environments."""
    rssm.eval()
    task_rewards = []
    for task_idx in range(n_tasks):
        bddl = benchmark_obj.get_task_bddl_file_path(task_idx)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
        ep_rews = []
        for ep in range(n_episodes):
            result = env.reset()
            obs = result[0] if isinstance(result, tuple) else result
            ep_r = 0
            h, z = rssm.rssm.initial_state(1, device)
            for step in range(200):
                state = np.concatenate([obs["robot0_proprio-state"], obs["object-state"]])
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    result_rssm = rssm.rssm.observe_step(h, z, torch.zeros(1, 7, device=device), state_t)
                    h, z = result_rssm["h"], result_rssm["z"]
                    rssm_state = torch.cat([h, z], dim=-1)
                    action = actor(rssm_state).squeeze(0).cpu().numpy()
                action = np.clip(action, -1, 1)
                result = env.step(action)
                obs, r, d = result[0], result[1], result[2]
                ep_r += r
                if d:
                    break
            ep_rews.append(ep_r)
        task_rewards.append(np.mean(ep_rews))
        print("  Task %d: reward=%.4f" % (task_idx, np.mean(ep_rews)))
        env.close()
    return task_rewards


def main():
    import typer
    app = typer.Typer()

    @app.command()
    def run(
        n_collect: int = typer.Option(30, help="Real episodes to collect"),
        n_epochs: int = typer.Option(100, help="WM training epochs"),
        n_imagined: int = typer.Option(50, help="Imagined rollouts per epoch"),
        horizon: int = typer.Option(50, help="Imagination horizon"),
        n_eval_tasks: int = typer.Option(5, help="Tasks to evaluate"),
        n_eval_episodes: int = typer.Option(5, help="Episodes per task"),
        device: str = typer.Option("cuda"),
        save_dir: str = typer.Option("trained_models/libero"),
    ):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        print("Step 1: Collecting real LIBERO data...")
        bm = benchmark.get_benchmark_dict()
        spatial_bm = bm["libero_spatial"]()
        real_obs, real_actions, real_rewards, real_dones = collect_real_data(
            spatial_bm, n_collect
        )
        print("Collected: obs=%s actions=%s mean_reward=%.4f" %
              (real_obs.shape, real_actions.shape, np.mean(real_rewards)))

        print("\nStep 2: Training RSSM on real data...")
        rssm = WorldModel(real_obs.shape[1], real_actions.shape[1]).to(device)
        rssm_path = "%s/rssm_libero.pt" % save_dir
        if Path(rssm_path).exists():
            rssm.load_state_dict(torch.load(rssm_path, map_location=device))
            print("Loaded pre-trained RSSM from %s" % rssm_path)
        else:
            opt = torch.optim.Adam(rssm.parameters(), lr=1e-3)
            obs_t = torch.tensor(real_obs, dtype=torch.float32, device=device).unsqueeze(0)
            act_t = torch.tensor(real_actions, dtype=torch.float32, device=device).unsqueeze(0)
            rew_t = torch.zeros(1, obs_t.shape[1], device=device)
            done_t = torch.zeros(1, obs_t.shape[1], device=device)
            for ep in range(50):
                m = rssm.training_step(obs_t, act_t, rew_t, done_t)
                opt.zero_grad()
                m["total_loss"].backward()
                opt.step()
                if (ep + 1) % 10 == 0:
                    print("  Epoch %d: loss=%.4f" % (ep + 1, m["total_loss"].item()))
            torch.save(rssm.state_dict(), rssm_path)
            print("Saved RSSM to %s" % rssm_path)

        print("\nStep 3: Training policy with world model...")
        actor = train_with_world_model(
            rssm, real_obs, real_actions, real_rewards,
            n_epochs=n_epochs, n_imagined=n_imagined, horizon=horizon, device=device
        )
        torch.save(actor.state_dict(), "%s/actor_libero.pt" % save_dir)
        print("Saved actor to %s/actor_libero.pt" % save_dir)

        print("\nStep 4: Evaluating on real LIBERO...")
        task_rewards = evaluate_policy(
            actor, rssm, spatial_bm, n_eval_tasks, n_eval_episodes, device
        )
        import json
        results = {
            "avg_reward": float(np.mean(task_rewards)),
            "task_rewards": [float(r) for r in task_rewards],
        }
        with open("%s/libero_trained_results.json" % save_dir, "w") as f:
            json.dump(results, f, indent=2)
        print("\n" + "=" * 60)
        print("LIBERO TRAINED POLICY RESULTS")
        print("Average reward: %.4f" % np.mean(task_rewards))
        print("=" * 60)

    app()


if __name__ == "__main__":
    main()
