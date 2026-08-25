#!/bin/bash
# A100 setup script for WAM e2e loop
# Run this on the A100 instance after provisioning
set -e

echo "=== WAM E2E Loop — A100 Setup ==="

# 1. Clone repo
if [ ! -d "robotics_world_models" ]; then
    git clone git@github.com:charleneleong-ai/robotics-world-models.git
fi
cd robotics_world_models

# 2. Create venv (Python 3.12 for mani_skill compat)
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install all deps
pip install uv
uv sync
uv pip install mani_skill

# 4. Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import mani_skill; print(f'ManiSkill3 OK')"
python -c "import ray; print(f'Ray {ray.__version__}')"

# 5. Run tests
PYTHONPATH=. python -m pytest experiments/diffusion_wm/test_model.py -q

echo ""
echo "=== Setup complete ==="
echo ""
echo "Quick smoke test (1 round, PickCube):"
echo "  PYTHONPATH=. python -m experiments.diffusion_wm.loop --task PickCube-v1 --num-rounds 1"
echo ""
echo "Full sweep (5 tasks x 5 rounds):"
echo "  PYTHONPATH=. python -m experiments.diffusion_wm.loop --task PlugCharger-v1 --num-rounds 5"
echo "  PYTHONPATH=. python -m experiments.diffusion_wm.loop --task PegInsertionSide-v1 --num-rounds 5"
echo "  PYTHONPATH=. python -m experiments.diffusion_wm.loop --task PickCube-v1 --num-rounds 5"
echo "  PYTHONPATH=. python -m experiments.diffusion_wm.loop --task PushCube-v1 --num-rounds 5"
echo "  PYTHONPATH=. python -m experiments.diffusion_wm.loop --task LinkBridge-v1 --num-rounds 5"
