#!/usr/bin/env python3
"""Ablation study: EWC penalty weight (λ) sensitivity."""
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
RESULTS_FILE = '/home/ubuntu/robotics_world_models/experiments/causal_trust_world_model_learning/ablation_lambda.json'

LAMBDA_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]

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

def run_lambda_ablation(lambda_val, n_episodes=3, steps_per_episode=100):
    obs_dim = 35
    act_dim = 8
    
    wm = SimpleRSSM(obs_dim).to(DEVICE)
    policy = SimplePolicy(obs_dim, act_dim).to(DEVICE)
    trust = SimpleTrustScorer(obs_dim).to(DEVICE)
    
    wm_params = list(wm.parameters())
    initial_params = [p.clone() for p in wm_params]
    
    rewards = []
    trust_scores = []
    
    for ep in range(n_episodes):
        obs = torch.randn(1, obs_dim).to(DEVICE)
        ep_reward = 0
        ep_trust = []
        
        for step in range(steps_per_episode):
            with torch.no_grad():
                pred_next, h_next = wm(obs)
                action = policy(obs)
                pred_error = pred_next - obs
                trust_score = trust(pred_error)
            
            ep_trust.append(trust_score.item())
            
            next_obs = obs + 0.01 * torch.randn_like(obs)
            reward = -torch.norm(next_obs[:, :8]).item()
            ep_reward += reward
            
            with torch.no_grad():
                pred_next, _ = wm(obs)
            
            recon_loss = F.mse_loss(pred_next, next_obs)
            
            ewc_penalty = torch.tensor(0.0).to(DEVICE)
            for p, init_p in zip(wm_params, initial_params):
                ewc_penalty += torch.sum((p - init_p) ** 2)
            
            loss = trust_score * recon_loss + lambda_val * ewc_penalty
            
            wm.optimizer.zero_grad()
            loss.backward()
            wm.optimizer.step()
            
            obs = next_obs.detach()
        
        rewards.append(ep_reward)
        trust_scores.append(np.mean(ep_trust))
    
    return {
        'lambda': lambda_val,
        'avg_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'avg_trust': float(np.mean(trust_scores)),
        'final_trust': float(trust_scores[-1])
    }

def main():
    print(f"Running λ ablation on {DEVICE}")
    
    results = []
    for lam in LAMBDA_VALUES:
        print(f"\nTesting λ={lam}...")
        result = run_lambda_ablation(lam)
        results.append(result)
        print(f"  Reward: {result['avg_reward']:.3f} ± {result['std_reward']:.3f}")
        print(f"  Trust: {result['avg_trust']:.3f}")
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_FILE}")
    
    print("\n=== Summary ===")
    print(f"{'Lambda':<10} {'Reward':<15} {'Trust':<10}")
    print("-" * 35)
    for r in results:
        print(f"{r['lambda']:<10} {r['avg_reward']:<15.3f} {r['avg_trust']:<10.3f}")

if __name__ == '__main__':
    main()
