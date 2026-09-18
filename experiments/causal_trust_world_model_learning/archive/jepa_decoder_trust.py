"""JEPA with decoder-aware trust scoring."""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

sys.path.insert(0, os.path.dirname(__file__))

DEVICE = "cuda"
SUITE_DIR = "/home/ubuntu/robotics_world_models/LIBERO/libero_spatial"
N_TASKS = 10
BATCH = 64
LR = 1e-3
EPOCHS = 50
MAX_DEMOS = 5
N_SEEDS = 5
WM_EPOCHS = 20


class JEPABackbone(nn.Module):
    def __init__(self, obs_dim, act_dim, h=256, lat=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, h), nn.LayerNorm(h), nn.SiLU(), nn.Linear(h, lat))
        self.predictor = nn.Sequential(nn.Linear(lat + act_dim, h), nn.SiLU(), nn.Linear(h, lat))
        self.decoder = nn.Sequential(nn.Linear(lat, h), nn.SiLU(), nn.Linear(h, obs_dim))
        self.trust_head = nn.Sequential(nn.Linear(obs_dim * 2, h), nn.SiLU(), nn.Linear(h, 1), nn.Sigmoid())

    def train_loss(self, obs_seq, act_seq):
        B, T, _ = obs_seq.shape
        dev = obs_seq.device
        loss = torch.tensor(0.0, device=dev)
        for t in range(T - 1):
            z = self.encoder(obs_seq[:, t])
            pred_z = self.predictor(torch.cat([z, act_seq[:, t]], dim=-1))
            pred_obs = self.decoder(pred_z)
            loss = loss + F.mse_loss(pred_obs, obs_seq[:, t + 1])
        return loss / (T - 1)

    @torch.no_grad()
    def predict_error(self, obs, act, next_obs):
        z = self.encoder(obs)
        pred_z = self.predictor(torch.cat([z, act], dim=-1))
        pred_obs = self.decoder(pred_z)
        return F.mse_loss(pred_obs, next_obs, reduction="none").mean(dim=-1)

    def compute_trust(self, obs, act, next_obs):
        """Trust from decoder-reconstructed observation error."""
        z = self.encoder(obs)
        pred_z = self.predictor(torch.cat([z, act], dim=-1))
        pred_obs = self.decoder(pred_z)
        # Compare predicted vs actual observation
        obs_error = F.mse_loss(pred_obs, next_obs, reduction="none").mean(dim=-1, keepdim=True)
        # Use trust head to learn when predictions are reliable
        trust = self.trust_head(torch.cat([obs, pred_obs], dim=-1))
        return trust.squeeze(-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def load_demos(suite_dir, n_tasks=10, max_demos=MAX_DEMOS):
    all_demos = []
    files = sorted(f for f in os.listdir(suite_dir) if f.endswith(".hdf5"))
    for fn in files[:n_tasks]:
        with h5py.File(os.path.join(suite_dir, fn), "r") as hf:
            task_demos = []
            count = 0
            for k in sorted(hf["data"].keys()):
                if not k.startswith("demo_") or count >= max_demos:
                    continue
                demo = hf["data"][k]
                parts = [np.array(demo["obs"][f]) for f in
                         ["ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"]]
                task_demos.append({
                    "obs": np.concatenate(parts, axis=-1),
                    "acts": np.array(demo["actions"]),
                })
                count += 1
            all_demos.append(task_demos)
    return all_demos


def train_wm_with_trust_head(wm, demos, epochs=WM_EPOCHS):
    """Train WM with trust head jointly."""
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    wm.train()
    for _ in range(epochs):
        for demo in demos:
            o = torch.tensor(demo["obs"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a = torch.tensor(demo["acts"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Reconstruction loss
            recon_loss = wm.train_loss(o, a)
            
            # Trust head loss
            B, T, _ = o.shape
            with torch.no_grad():
                trust_targets = []
                for t in range(T - 1):
                    z = wm.encoder(o[:, t])
                    pred_z = wm.predictor(torch.cat([z, a[:, t]], dim=-1))
                    pred_obs = wm.decoder(pred_z)
                    err = F.mse_loss(pred_obs, o[:, t + 1], reduction="none").mean(dim=-1)
                    trust_targets.append(torch.exp(-err))
                trust_target = torch.stack(trust_targets, dim=1)  # (B, T-1)
            
            trust_preds = []
            for t in range(T - 1):
                tp = wm.compute_trust(o[:, t], a[:, t], o[:, t + 1])
                trust_preds.append(tp)
            trust_pred = torch.stack(trust_preds, dim=1)  # (B, T-1)
            
            trust_loss = F.mse_loss(trust_pred, trust_target)
            
            loss = recon_loss + 0.1 * trust_loss
            opt.zero_grad(); loss.backward(); opt.step()


def train_bc_trust(policy, wm, demos, epochs=EPOCHS):
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    all_obs = np.concatenate([d["obs"] for d in demos])
    all_acts = np.concatenate([d["acts"] for d in demos])
    obs_t = torch.tensor(all_obs, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(all_acts, dtype=torch.float32).to(DEVICE)
    wm.eval(); policy.train()
    for _ in range(epochs):
        perm = torch.randperm(obs_t.size(0))
        for i in range(0, obs_t.size(0), BATCH):
            idx = perm[i:i+BATCH]
            if len(idx) < 2: continue
            pred = policy(obs_t[idx])
            with torch.no_grad():
                trust = wm.compute_trust(obs_t[idx], act_t[idx], obs_t[idx])
                w = trust.clamp(0.1, 1.0)
            loss = (F.mse_loss(pred, act_t[idx], reduction="none").mean(dim=-1) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def eval_bc(policy, demos):
    policy.eval()
    all_obs = np.concatenate([d["obs"] for d in demos])
    all_acts = np.concatenate([d["acts"] for d in demos])
    obs_t = torch.tensor(all_obs, dtype=torch.float32).to(DEVICE)
    act_t = torch.tensor(all_acts, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        error = F.mse_loss(policy(obs_t), act_t).item()
    return error


def run():
    print(f"\n{'='*60}")
    print(f"  JEPA with Decoder-Aware Trust")
    print(f"{'='*60}")
    
    demos = load_demos(SUITE_DIR, N_TASKS)
    print(f"Loaded {len(demos)} tasks")
    
    obs_dim, act_dim = 21, 7
    results = {"none": [], "decoder_trust": [], "latent_trust": []}
    
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        t0 = __import__('time').time()
        
        # Train WM with trust head
        wm = JEPABackbone(obs_dim, act_dim).to(DEVICE)
        train_wm_with_trust_head(wm, demos[0])
        
        # No-trust BC
        std = MLP(obs_dim, act_dim).to(DEVICE)
        opt = torch.optim.Adam(std.parameters(), lr=LR)
        all_obs = np.concatenate([d["obs"] for d in demos[0][:3]])
        all_acts = np.concatenate([d["acts"] for d in demos[0][:3]])
        obs_t = torch.tensor(all_obs, dtype=torch.float32).to(DEVICE)
        act_t = torch.tensor(all_acts, dtype=torch.float32).to(DEVICE)
        std.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(obs_t.size(0))
            for i in range(0, obs_t.size(0), BATCH):
                idx = perm[i:i+BATCH]
                loss = F.mse_loss(std(obs_t[idx]), act_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        results["none"].append(eval_bc(std, demos[0][3:] if len(demos[0]) > 3 else demos[0][-1:]))
        
        # Decoder-aware trust BC
        tru = MLP(obs_dim, act_dim).to(DEVICE)
        train_bc_trust(tru, wm, demos[0])
        results["decoder_trust"].append(eval_bc(tru, demos[0][3:] if len(demos[0]) > 3 else demos[0][-1:]))
        
        # Latent trust (original predict_error)
        tru2 = MLP(obs_dim, act_dim).to(DEVICE)
        opt2 = torch.optim.Adam(tru2.parameters(), lr=LR)
        tru2.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(obs_t.size(0))
            for i in range(0, obs_t.size(0), BATCH):
                idx = perm[i:i+BATCH]
                pred = tru2(obs_t[idx])
                with torch.no_grad():
                    pe = wm.predict_error(obs_t[idx], act_t[idx], obs_t[idx])
                    trust = torch.exp(-pe / (pe.mean() + 1e-8)).clamp(0, 1)
                    w = trust.clamp(0.1, 1.0)
                loss = (F.mse_loss(pred, act_t[idx], reduction="none").mean(dim=-1) * w).mean()
                opt2.zero_grad(); loss.backward(); opt.step()
        results["latent_trust"].append(eval_bc(tru2, demos[0][3:] if len(demos[0]) > 3 else demos[0][-1:]))
        
        elapsed = __import__('time').time() - t0
        print(f"  Seed {seed}: none={results['none'][-1]:.4f} "
              f"decoder={results['decoder_trust'][-1]:.4f} "
              f"latent={results['latent_trust'][-1]:.4f} "
              f"({elapsed:.1f}s)")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary (mean ± std)")
    print(f"{'='*60}")
    for method in results:
        arr = np.array(results[method])
        print(f"  {method:<15}: {arr.mean():.4f} ± {arr.std():.4f}")
    
    # Save
    out = os.path.join(os.path.dirname(__file__), "jepa_decoder_trust.json")
    with open(out, "w") as f:
        json.dump({m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "seeds": v} 
                   for m, v in results.items()}, f, indent=2)
    print(f"\nSaved to {out}")
    
    return results


if __name__ == "__main__":
    run()
