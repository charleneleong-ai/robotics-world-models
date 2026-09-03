"""Test and compare WAM architectures.

Compares:
1. Original DiffusionWAM (14M params)
2. Scaled DiffusionWAM (100M+ params)
3. Cascaded WAM (separate world model + action decoder)
"""
import torch
import time
from experiments.diffusion_wm.world_action_model import DiffusionWAM
from experiments.diffusion_wm.scaled_wam import ScaledDiffusionWAM
from experiments.diffusion_wm.cascaded_wam import CascadedWAM


def count_params(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_forward_pass(model, name: str, obs_dim: int = 42, act_dim: int = 8):
    """Test forward pass and measure time."""
    model.eval()
    obs = torch.randn(1, obs_dim)
    action = torch.randn(1, act_dim)
    timestep = torch.tensor([500])

    # Forward pass
    start = time.time()
    with torch.no_grad():
        if hasattr(model, 'compute_loss'):
            next_state = torch.randn(1, obs_dim)
            total_loss, state_loss, action_loss = model.compute_loss(obs, next_state, action)
        if hasattr(model, 'denoise_action'):
            denoised = model.denoise_action(obs, num_steps=10)
    elapsed = time.time() - start

    print(f"{name}:")
    print(f"  Parameters: {count_params(model):,}")
    print(f"  Forward pass: {elapsed:.3f}s")
    print()


def main():
    """Compare architectures."""
    obs_dim = 42
    act_dim = 8

    print("=" * 60)
    print("WAM Architecture Comparison")
    print("=" * 60)
    print()

    # 1. Original DiffusionWAM
    model_original = DiffusionWAM(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=512,
        num_blocks=6,
    )
    test_forward_pass(model_original, "Original DiffusionWAM")

    # 2. Scaled DiffusionWAM (100M)
    model_scaled = ScaledDiffusionWAM(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=1024,
        num_blocks=12,
        num_heads=8,
    )
    test_forward_pass(model_scaled, "Scaled DiffusionWAM (100M)")

    # 3. Scaled DiffusionWAM (200M)
    model_scaled_200m = ScaledDiffusionWAM(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=2048,
        num_blocks=12,
        num_heads=16,
    )
    test_forward_pass(model_scaled_200m, "Scaled DiffusionWAM (200M)")

    # 4. Cascaded WAM
    model_cascaded = CascadedWAM(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=512,
        wm_blocks=4,
        ad_blocks=4,
    )
    test_forward_pass(model_cascaded, "Cascaded WAM")

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print(f"{'Model':<30} {'Params':<15} {'vs Original':<15}")
    print("-" * 60)

    models = [
        ("Original DiffusionWAM", model_original),
        ("Scaled DiffusionWAM (100M)", model_scaled),
        ("Scaled DiffusionWAM (200M)", model_scaled_200m),
        ("Cascaded WAM", model_cascaded),
    ]

    original_params = count_params(model_original)
    for name, model in models:
        params = count_params(model)
        ratio = params / original_params
        print(f"{name:<30} {params:>12,}  {ratio:>10.1f}x")


if __name__ == "__main__":
    main()
