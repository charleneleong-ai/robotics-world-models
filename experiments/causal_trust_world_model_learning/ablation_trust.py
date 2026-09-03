#!/usr/bin/env python3
"""Ablation study: Trust method comparison."""
import os
import sys
import json
import numpy as np

os.environ['MUJOCO_GL'] = 'egl'
os.environ['DISPLAY'] = ''

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/ubuntu/robotics_world_models/experiments/causal_trust_world_model_learning')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_FILE = '/home/ubuntu/robotics_world_models/experiments/causal_trust_world_model_learning/ablation_trust.json'

class SimpleRSSM(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        self.dynamics = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, obs_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def forward(self, obs):
        h = F.relu(self.encoder(obs))
        h_next = F.relu(self.dynamics(h))
        pred_obs = self.decoder(h_next)
        return pred_obs, h_next

class SimplePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim=8, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )
    
    def forward(self, obs):
        return self.net(obs)

class SimpleTrustScorer(nn.Module):
    def __init__(self, obs_dim=35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pred_error):
        return self.net(pred_error)

def compute_ema_trust(pred_error, alpha=0.1):
    error_norm = torch.norm(pred_error).item()
    return np.exp(-alpha * error_norm)

def compute_multistep_trust(pred_errors, window=5):
    if len(pred_errors) < window:
        return 1.0
    recent = pred_errors[-window:]
    trend = np.mean([np.linalg.norm(e) for e in recent])
    return np.exp(-trend)

def run_trust_ablation(trust_method, n_episodes=3, steps_per_episode=100):
    obs_dim = 35
    
    wm = SimpleRSSM(obs_dim).to(DEVICE)
    policy = SimplePolicy(obs_dim).to(DEVICE)
    trust_scorer = SimpleTrustScorer(obs_dim).to(DEVICE)
    
    ewc_lambda = 1.0
    wm_params = list(wm.parameters())
    initial_params = [p.clone() for p in wm_params]
    
    rewards = []
    trust_history = []
    pred_errors = []
    
    for ep in range(n_episodes):
        obs = torch.randn(1, obs_dim).to(DEVICE)
        ep_reward = 0
        ep_trust = []
        
        for step in range(steps_per_episode):
            with torch.no_grad():
                pred_next, _ = wm(obs)
                action = policy(obs)
                pred_error = pred_next - obs
                pred_errors.append(pred_error.cpu().numpy().flatten())
            
            if trust_method == 'ema':
                trust_score = compute_ema_trust(pred_error, alpha=0.1)
            elif trust_method == 'multistep':
                trust_score = compute_multistep_trust(pred_errors, window=5)
            elif trust_method == 'learned':
                with torch.no_grad():
                    trust_score = trust_scorer(pred_error).item()
            else:  # none
                trust_score = 1.0
            
            ep_trust.append(trust_score)
            
            next_obs = obs + 0.01 * torch.randn_like(obs)
            reward = -torch.norm(next_obs[:, :8]).item()
            ep_reward += reward
            
            with torch.no_grad():
                pred_next, _ = wm(obs)
            
            recon_loss = F.mse_loss(pred_next, next_obs)
            
            ewc_penalty = torch.tensor(0.0).to(DEVICE)
            for p, init_p in zip(wm_params, initial_params):
                ewc_penalty += torch.sum((p - init_p) ** 2)
            
            loss = trust_score * recon_loss + ewc_lambda * ewc_penalty
            
            wm.optimizer.zero_grad()
            loss.backward()
            wm.optimizer.step()
            
            obs = next_obs.detach()
        
        rewards.append(ep_reward)
        trust_history.append(np.mean(ep_trust))
    
    return {
        'trust_method': trust_method,
        'avg_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'avg_trust': float(np.mean(trust_history)),
        'trust_std': float(np.std(trust_history))
    }

def main():
    print(f"Running trust method ablation on {DEVICE}")
    
    methods = ['none', 'ema', 'multistep', 'learned']
    results = []
    
    for method in methods:
        print(f"\nTesting {method}...")
        result = run_trust_ablation(method)
        results.append(result)
        print(f"  Reward: {result['avg_reward']:.3f} ± {result['std_reward']:.3f}")
        print(f"  Trust: {result['avg_trust']:.3f} ± {result['trust_std']:.3f}")
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_FILE}")
    
    print("\n=== Summary ===")
    print(f"{'Method':<12} {'Reward':<18} {'Trust':<15}")
    print("-" * 45)
    for r in results:
        print(f"{r['trust_method']:<12} {r['avg_reward']:<18.3f} {r['avg_trust']:<15.3f}")

if __name__ == '__main__':
    main()
