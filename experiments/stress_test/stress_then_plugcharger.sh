#!/usr/bin/env bash
# Robustness stress-test orchestrator, invoked by the watcher when the real
# PegInsertionSide final.pt lands. Runs the WM noise eval, the classical noise
# sweep, and the divergence plot; then launches PlugCharger TD-MPC2 training.
#
# Every step is best-effort: a failing step is logged and we continue to the
# PlugCharger launch regardless.
set -u

STRESS_DIR=/workspace/stress_test
TDMPC2_DIR=/workspace/ManiSkill/examples/baselines/tdmpc2
MPLIB_PY=/workspace/mplib_venv/bin/python   # numpy 1.26 + mplib (no segfault)
SYS_PY=python3                              # torch + CUDA + matplotlib
LOGDIR=/workspace/logs
WM_CKPT=/workspace/ManiSkill/examples/baselines/tdmpc2/logs/PegInsertionSide-v1/1/tdmpc2-vast/models/final.pt

mkdir -p "${LOGDIR}" "${STRESS_DIR}"

# stdout is captured by the launcher's redirect — no separate per-run logfile.
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

log "=== stress_then_plugcharger START ==="

# (a) World-model noise eval (real PegInsertionSide checkpoint).
log "--- (a) WM noise eval ---"
if [ -f "${WM_CKPT}" ]; then
  ( cd "${TDMPC2_DIR}" && "${SYS_PY}" -u "${STRESS_DIR}/wm_noise_eval.py" \
      --checkpoint "${WM_CKPT}" \
      --sigmas-mm 0,5,10,15,20 \
      --eval-episodes 25 \
      --out "${STRESS_DIR}/wm_results.json" ) 2>&1 \
    && log "WM eval OK" || log "WM eval FAILED (continuing)"
else
  log "WM checkpoint not found at ${WM_CKPT} (skipping WM eval)"
fi

# (b) Classical noise sweep (numpy 1.26 venv to avoid mplib segfault).
log "--- (b) Classical noise sweep ---"
( cd "${TDMPC2_DIR}" && "${MPLIB_PY}" -u "${STRESS_DIR}/classical_noise_sweep.py" \
    --sigmas-mm 0,5,10,15,20 \
    --n 50 \
    --out "${STRESS_DIR}/classical_results.json" ) 2>&1 \
  && log "Classical sweep OK" || log "Classical sweep FAILED (continuing)"

# (c) Divergence plot.
log "--- (c) Divergence plot ---"
( cd "${STRESS_DIR}" && "${SYS_PY}" -u "${STRESS_DIR}/plot_divergence.py" \
    --classical "${STRESS_DIR}/classical_results.json" \
    --wm "${STRESS_DIR}/wm_results.json" \
    --out "${STRESS_DIR}/divergence.png" ) 2>&1 \
  && log "Plot OK -> ${STRESS_DIR}/divergence.png" || log "Plot FAILED (continuing)"

# (d) Launch PlugCharger TD-MPC2 training, fully detached.
log "--- (d) Launch PlugCharger training (detached) ---"
PLUG_LOG="${LOGDIR}/tdmpc2_plugcharger_$(date -u +%Y%m%dT%H%M%SZ).log"
cd "${TDMPC2_DIR}" && setsid nohup python3 -u train.py \
  model_size=5 steps=2000000 num_envs=32 control_mode=pd_joint_delta_pos \
  buffer_size=1000000 seed=1 env_id=PlugCharger-v1 obs=state \
  exp_name=tdmpc2-plugcharger wandb=true wandb_project=wm-manip \
  wandb_name=tdmpc2-plugcharger \
  </dev/null >>"${PLUG_LOG}" 2>&1 & disown
log "PlugCharger training launched -> ${PLUG_LOG}"

log "=== stress_then_plugcharger DONE ==="
