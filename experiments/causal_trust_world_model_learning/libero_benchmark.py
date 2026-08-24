"""LIBERO Benchmark for ContinualWAM."""
import json
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


def collect_libero_episodes(benchmark_obj, n_episodes=50, max_steps=200):
    all_obs, all_actions, all_rewards = [], [], []
    for ep in range(n_episodes):
        task_idx = ep % len(benchmark_obj.get_task_names())
        bddl_path = benchmark_obj.get_task_bddl_file_path(task_idx)
        env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=64, camera_widths=64)
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        ep_obs, ep_act, ep_rew = [], [], []
        for step in range(max_steps):
            state = np.concatenate([obs["robot0_proprio-state"], obs["object-state"]])
            action = np.random.randn(env.env.action_dim) * 0.5
            action = np.clip(action, -1, 1)
            ep_obs.append(state)
            ep_act.append(action)
            result = env.step(action)
            next_obs, reward, done = result[0], result[1], result[2]
            ep_rew.append(reward)
            obs = next_obs
            if done:
                break
        all_obs.append(np.array(ep_obs))
        all_actions.append(np.array(ep_act))
        all_rewards.append(sum(ep_rew))
        env.close()
        if (ep + 1) % 10 == 0:
            print("  Collected %d/%d episodes" % (ep + 1, n_episodes))
    return np.concatenate(all_obs), np.concatenate(all_actions), np.array(all_rewards)


def train_rssm_libero(obs, actions, epochs=50, device="cuda"):
    obs_dim = obs.shape[1]
    action_dim = actions.shape[1]
    print("Training RSSM: obs_dim=%d, action_dim=%d" % (obs_dim, action_dim))
    model = WorldModel(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    act_t = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0)
    rew_t = torch.zeros(obs_t.shape[0], obs_t.shape[1], device=device)
    done_t = torch.zeros(obs_t.shape[0], obs_t.shape[1], device=device)

    for epoch in range(epochs):
        metrics = model.training_step(obs_t, act_t, rew_t, done_t)
        total_loss = metrics["total_loss"]
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print("  Epoch %d: loss=%.4f obs_loss=%.4f" % (epoch + 1, total_loss.item(), metrics["obs_loss"]))
    return model


def run_libero_cl_benchmark(rssm, benchmark_obj, n_tasks=10, n_episodes=20, device="cuda"):
    obs_dim = 109
    action_dim = 7

    class SimplePolicy(nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
        def forward(self, x):
            return self.net(x)

    policy = SimplePolicy(obs_dim, action_dim).to(device)
    results = {}

    for method_name in ["EMA", "No Trust"]:
        print("\nRunning: %s" % method_name)
        task_rewards = []
        for task_idx in range(n_tasks):
            bddl_path = benchmark_obj.get_task_bddl_file_path(task_idx)
            env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=64, camera_widths=64)
            ep_rewards = []
            for ep in range(n_episodes):
                result = env.reset()
                obs = result[0] if isinstance(result, tuple) else result
                ep_reward = 0
                for step in range(200):
                    state = np.concatenate([obs["robot0_proprio-state"], obs["object-state"]])
                    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = policy(state_t).squeeze(0).cpu().numpy()
                    action = action + np.random.randn(*action.shape) * 0.1
                    action = np.clip(action, -1, 1)
                    result = env.step(action)
                    next_obs, reward, done = result[0], result[1], result[2]
                    ep_reward += reward
                    obs = next_obs
                    if done:
                        break
                ep_rewards.append(ep_reward)
            task_rewards.append(np.mean(ep_rewards))
            print("  Task %d: reward=%.4f" % (task_idx + 1, np.mean(ep_rewards)))
            env.close()
        results[method_name] = {"avg_reward": np.mean(task_rewards), "task_rewards": task_rewards}
    return results


def main():
    import typer
    app = typer.Typer(help="LIBERO benchmark for ContinualWAM.")

    @app.command()
    def run(
        n_collect_episodes: int = typer.Option(50, help="Episodes for RSSM training"),
        n_cl_tasks: int = typer.Option(10, help="Number of CL tasks"),
        n_cl_episodes: int = typer.Option(20, help="Episodes per CL task"),
        epochs: int = typer.Option(50, help="RSSM training epochs"),
        device: str = typer.Option("cuda", help="Device"),
        save_dir: str = typer.Option("trained_models/libero", help="Save directory"),
    ):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print("Step 1: Collecting LIBERO data...")
        bm = benchmark.get_benchmark_dict()
        spatial_bm = bm["libero_spatial"]()
        obs, actions, rewards = collect_libero_episodes(spatial_bm, n_collect_episodes)
        print("Collected: obs=%s, actions=%s, mean_reward=%.4f" % (obs.shape, actions.shape, np.mean(rewards)))
        print("\nStep 2: Training RSSM...")
        rssm = train_rssm_libero(obs, actions, epochs, device)
        rssm_path = "%s/rssm_libero_spatial.pt" % save_dir
        torch.save(rssm.state_dict(), rssm_path)
        print("Saved RSSM to %s" % rssm_path)
        print("\nStep 3: Running CL benchmark...")
        results = run_libero_cl_benchmark(rssm, spatial_bm, n_cl_tasks, n_cl_episodes, device)
        results_path = "%s/libero_results.json" % save_dir
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print("\nSaved results to %s" % results_path)
        print("\n" + "=" * 60)
        print("LIBERO CL BENCHMARK RESULTS")
        print("=" * 60)
        for method, data in results.items():
            print("%s: avg_reward=%.4f" % (method, data["avg_reward"]))
        print("=" * 60)

    app()


if __name__ == "__main__":
    main()
