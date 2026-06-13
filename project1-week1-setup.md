# Project #1 — Week 1 Environment Bring-Up (datacenter A100 80GB)

> Target box: **Ubuntu 22.04 · conda · CUDA 12.x · datacenter A100 80GB (headless, no display)**.
> Goal of this doc: get to the Week-1 exit gate — *a learned world model training on a ManiSkill3
> manipulation task vs a model-free floor, logged to W&B* — without losing days to install archaeology.
>
> **The one risk that matters:** ManiSkill3 renders via **Vulkan**, and a datacenter A100 has **no
> display and no RT Cores**. Headless Vulkan offscreen rendering is the make-or-break step. Everything
> below front-loads it. Run `setup.sh` step by step (don't blind-run it); the **go/no-go gate** in §4
> is the gate — do not start world-model work until it passes.
>
> Repos move (ManiSkill3 is still beta `3.0.0bXX`); where a command might have drifted, I flag
> *(verify vs current README)*. Source of truth: <https://maniskill.readthedocs.io>.

---

## 0. Why bundled baselines, not `nicklashansen/tdmpc2`

The SOTA doc's anchor was TD-MPC2 — but the official `nicklashansen/tdmpc2` integrates **ManiSkill2**,
and we're on **ManiSkill3**. ManiSkill3 **ships its own PPO / SAC / TD-MPC2 baselines** under
`mani_skill/examples/baselines/`. Use those — same algorithms, zero version-mismatch, one env.
DreamerV3 (Week 2) comes from `NM512/dreamerv3-torch` pointed at a ManiSkill3 gym env.

## 1. System prerequisites (you may need sudo / a base image with these)

```bash
nvidia-smi                      # confirm A100 80GB + driver (need CUDA 12.x capable: driver >= 525)
# Vulkan runtime + tools (THE critical bit for headless render):
sudo apt-get update
sudo apt-get install -y vulkan-tools libvulkan1 libvulkan-dev libglvnd-dev libgl1-mesa-glx
# Confirm the NVIDIA Vulkan ICD exists (the datacenter driver installs it):
ls /usr/share/vulkan/icd.d/        # expect nvidia_icd.json (and/or /etc/vulkan/icd.d/)
```

If `/usr/share/vulkan/icd.d/nvidia_icd.json` is missing, your driver was installed without Vulkan
support — reinstall the NVIDIA datacenter driver (it bundles the ICD) or create the ICD JSON manually
(ManiSkill docs → "Vulkan/rendering troubleshooting"). This is the single most common headless failure.

## 2. Conda env + PyTorch (CUDA 12.x)

```bash
conda create -y -n wm python=3.11      # 3.11 satisfies ManiSkill (>=3.10) AND DreamerV3 (>=3.11)
conda activate wm
# PyTorch matching your CUDA minor (cu121 shown; use cu124 if driver supports it) — verify at pytorch.org:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# must print: <ver> True NVIDIA A100-SXM4-80GB  (or similar)
```

## 3. ManiSkill3

```bash
pip install --upgrade mani-skill      # pulls SAPIEN (the Vulkan renderer)
python -m mani_skill.utils.download_asset PickCube-v1   # download task assets *(verify cmd vs README)*
```

## 4. ✅ GO/NO-GO GATE — headless render smoke test

**Do not proceed to world models until both of these pass.** This proves sim + Vulkan render + GPU
work end-to-end headless.

```bash
# (a) Vulkan sees the A100 headless:
vulkaninfo 2>/dev/null | grep -i -E "deviceName|NVIDIA" | head
#   -> must list "NVIDIA A100". If empty: fix Vulkan ICD (§1) before anything else.

# (b) ManiSkill GPU sim + RGBD render produces real pixels (not a black/empty buffer):
python - <<'PY'
import gymnasium as gym, mani_skill.envs, numpy as np
env = gym.make("PickCube-v1", num_envs=4, obs_mode="rgbd", render_backend="gpu")
obs, _ = env.reset(seed=0)
img = obs["sensor_data"]["base_camera"]["rgb"]   # *(key path may differ by version — print obs.keys())*
arr = np.asarray(img.cpu() if hasattr(img, "cpu") else img)
print("rgb shape:", arr.shape, "| nonzero frac:", (arr != 0).mean())
assert (arr != 0).mean() > 0.01, "BLACK FRAME -> headless Vulkan render is broken, fix before continuing"
print("GATE PASSED ✅  sim+render+GPU OK")
env.close()
PY
```

If (b) throws or prints a black frame: it's almost always the Vulkan ICD or a missing `vulkan-tools`.
Fallback to keep momentum (Week-1 plan): prototype the world-model loop on **MuJoCo Playground**
(`pip install playground`, no Vulkan, A100-fine) while you fix ManiSkill render in parallel.

## 5. First learning curves (Week 1 Day 3–5)

```bash
git clone https://github.com/mani-skill/ManiSkill.git ~/Documents/gen-ai/robotics_world_models/ManiSkill
cd ~/Documents/gen-ai/robotics_world_models/ManiSkill/mani_skill/examples/baselines
wandb login            # set up the W&B project for all runs

# PPO floor on PickCube (confirms training loop end-to-end) — exact flags vary, check each baseline's README:
python ppo/ppo.py --env_id PickCube-v1 --num_envs 512 --total_timesteps 5_000_000 \
       --track --wandb_project_name wm-manip      # *(verify flags vs ppo/README.md)*

# TD-MPC2 anchor on the same task (the world-model method):
python tdmpc2/train.py env_id=PickCube-v1 exp_name=tdmpc2_pickcube wandb=true   # *(verify vs tdmpc2/README.md)*

# SAC floor (kick off 3 seeds as background runs over the weekend):
python sac/sac.py --env_id PickCube-v1 --seed 1 --track --wandb_project_name wm-manip   # repeat seeds 2,3
```

## 6. Week-1 exit check

- [ ] §4 GATE passed (headless Vulkan render produces real pixels)
- [ ] PPO floor training, success curve logged to W&B
- [ ] TD-MPC2 success curve trending up on PickCube
- [ ] SAC 3-seed runs queued
- [ ] (if ManiSkill render fought you) MuJoCo Playground `PandaPickCube` loop running as fallback

Cleared all but the last? You're on schedule for Week 2 (DreamerV3 + the OMPL/MoveIt classical
baseline — the differentiator). See `project1-4week-plan.md`.

## Common failure → fix

| Symptom | Cause | Fix |
|---|---|---|
| `vulkaninfo` lists no NVIDIA device | Missing/!found Vulkan ICD | install `vulkan-tools libvulkan1`; ensure `/usr/share/vulkan/icd.d/nvidia_icd.json` |
| ManiSkill render = black frame | Headless offscreen not initialised | set `export DISPLAY=`; confirm ICD; try SAPIEN headless env vars per docs |
| `torch.cuda.is_available()` False | torch/CUDA minor mismatch | reinstall torch wheel matching driver CUDA (cu121/cu124) |
| TD-MPC2 import/env errors | used `nicklashansen/tdmpc2` (ManiSkill2) | use ManiSkill3's **bundled** `baselines/tdmpc2` instead |
| OOM despite 80GB | `num_envs` too high for render buffers | lower `--num_envs`; you have ample headroom, this is config not hardware |
