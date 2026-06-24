#!/bin/bash
# Cron-driven, idempotent: when the shared GPU frees (>=40GB, i.e. laguna gone), run the
# classical motion-planning baseline on PlugCharger-v1 — the harder-contact rung of the
# task ladder (mirrors classical_launch.sh for PegInsertionSide). Light + fast (1 env, 100
# episodes, ~minutes). Uses ManiSkill's built-in mplib solution (run.py + plug_charger.py).
# Writes PLUGCHARGER_DONE when finished.
set -u
ROOT=/home/ubuntu/robotics_world_models
PY=/home/ubuntu/miniconda3/envs/wm/bin/python
OUT=$ROOT/experiments/plugcharger_classical
STARTED=$ROOT/PLUGCHARGER_STARTED
[ -f "$STARTED" ] && exit 0
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ -z "$FREE" ] && exit 0
[ "$FREE" -lt 40000 ] && exit 0                              # GPU busy (laguna) — wait
# --- GPU free: run the classical baseline once ---
touch "$STARTED"
mkdir -p "$OUT/demos"
ts=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/logs/plugcharger_$ts.log
cd "$ROOT/benchmarks/ManiSkill" || exit 1
# run detached so cron returns; wrapper writes PLUGCHARGER_DONE + a results.json on completion
setsid bash -c "
  $PY -m mani_skill.examples.motionplanning.panda.run -e PlugCharger-v1 -n 100 \
      -b gpu --save-video --record-dir '$OUT/demos' --traj-name classical > '$LOG' 2>&1
  rate=\$(grep -oiE 'success[^0-9]*[0-9.]+' '$LOG' | tail -1)
  printf '{\"task\":\"PlugCharger-v1\",\"method\":\"classical (mplib RRTConnect+screw)\",\"n\":100,\"raw_success_line\":\"%s\",\"log\":\"%s\"}\n' \"\$rate\" '$LOG' > '$OUT/results.json'
  echo \"\$(date -u +%Y-%m-%dT%H:%M:%SZ) done\" > '$ROOT/PLUGCHARGER_DONE'
" </dev/null >/dev/null 2>&1 &
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) GPU free (${FREE}MB) -> launched PlugCharger classical baseline ($ts)"
