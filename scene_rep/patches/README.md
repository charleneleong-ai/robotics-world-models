# OpenSplat3D patches

Patches applied to the vendored OpenSplat3D repo on `pi-a100-80gb`.

## Applying

```bash
ssh pi-a100-80gb
cd ~/opensplat3d
git apply ~/robotics_world_models/scene_rep/patches/eval_lerf_mask_pred_masks.patch
```

## Patch: eval_lerf_mask_pred_masks

Adds `save_pred_masks_dir` parameter to `eval_grounded()`. When set, persists predicted masks as PNGs (one per view × class) alongside the aggregate metrics.

Usage in `__main__`:
```python
save_pred_masks_dir=output_path / "pred_masks"
```

**Prerequisites:** The nerfstudio env needs groundingdino, segment-anything, and compiled CUDA extensions (fused-ssim, simple-knn, diff-gaussian-rasterization). Install via:
```bash
conda activate nerfstudio
pip install -e ".[groundingdino]"
```

The eval script was patched and wired on 2026-08-10. Ready to run once deps are installed.

## W&B training (built-in)

Nerfstudio supports W&B logging natively:
```bash
ns-train splatfacto --vis wandb --steps-per-eval-image 100 --steps-per-eval-all-images 1000
```

This logs train/val rendered images, PSNR/SSIM/LPIPS, and loss curves to W&B automatically.
