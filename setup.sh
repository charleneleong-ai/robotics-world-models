#!/usr/bin/env bash
# Project #1 Week-1 bring-up — datacenter A100 80GB · Ubuntu 22.04 · conda · CUDA 12.x
# Run STEP BY STEP, not blind. The render gate (step 4) is the go/no-go — stop there if it fails.
set -euo pipefail

CUDA_TAG="${CUDA_TAG:-cu121}"   # override: CUDA_TAG=cu124 ./setup.sh  (match your driver)
ENV_NAME="${ENV_NAME:-wm}"
PROJECT_DIR="$HOME/Documents/gen-ai/robotics_world_models"

echo "==> [1/5] System + Vulkan prereqs (needs sudo)"
nvidia-smi
sudo apt-get update
sudo apt-get install -y vulkan-tools libvulkan1 libvulkan-dev libglvnd-dev libgl1-mesa-glx
echo "Vulkan ICDs present:"; ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null || \
  echo "!! No Vulkan ICD found — reinstall NVIDIA datacenter driver (bundles nvidia_icd.json) before continuing"

echo "==> [2/5] Conda env + PyTorch ($CUDA_TAG)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "==> [3/5] ManiSkill3 + assets"
pip install --upgrade mani-skill wandb
python -m mani_skill.utils.download_asset PickCube-v1 -y || \
  echo "!! asset download cmd may have drifted — check https://maniskill.readthedocs.io"

echo "==> [4/5] GO/NO-GO render gate (headless Vulkan)"
unset DISPLAY || true
vulkaninfo 2>/dev/null | grep -i -E "deviceName|NVIDIA" | head || \
  { echo "!! Vulkan sees no NVIDIA device — fix ICD before proceeding"; exit 1; }
python - <<'PY'
import gymnasium as gym, mani_skill.envs, numpy as np
env = gym.make("PickCube-v1", num_envs=4, obs_mode="rgbd", render_backend="gpu")
obs, _ = env.reset(seed=0)
img = obs["sensor_data"]["base_camera"]["rgb"]   # if KeyError: print(obs.keys()) and adjust path
arr = np.asarray(img.cpu() if hasattr(img, "cpu") else img)
frac = float((arr != 0).mean())
print("rgb shape:", arr.shape, "| nonzero frac:", frac)
assert frac > 0.01, "BLACK FRAME -> headless Vulkan render broken (see troubleshooting table)"
print("GATE PASSED  sim+render+GPU OK")
env.close()
PY

echo "==> [5/5] Clone ManiSkill for bundled baselines (PPO/SAC/TD-MPC2)"
[ -d "$PROJECT_DIR/ManiSkill" ] || git clone https://github.com/mani-skill/ManiSkill.git "$PROJECT_DIR/ManiSkill"
echo
echo "DONE. Next (run manually, verify flags vs each baseline's README):"
echo "  wandb login"
echo "  cd $PROJECT_DIR/ManiSkill/mani_skill/examples/baselines"
echo "  python ppo/ppo.py     --env_id PickCube-v1 --num_envs 512 --total_timesteps 5000000 --track --wandb_project_name wm-manip"
echo "  python tdmpc2/train.py env_id=PickCube-v1 exp_name=tdmpc2_pickcube wandb=true"
echo "  python sac/sac.py     --env_id PickCube-v1 --seed 1 --track --wandb_project_name wm-manip"
