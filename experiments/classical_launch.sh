#!/bin/bash
# Cron-driven, idempotent: when the shared GPU frees (>=40GB, i.e. laguna gone), run the
# classical motion-planning baseline on PegInsertionSide — the 3rd leg of the world-model
# vs model-free vs classical comparison. Light + fast (1 env, 100 episodes, ~minutes), so
# it completes in one burst the moment the card is free. Uses ManiSkill's built-in mplib
# solution (run.py). Writes CLASSICAL_DONE when finished.
set -u
ROOT=/home/ubuntu/robotics_world_models
PY=/home/ubuntu/miniconda3/envs/wm/bin/python
OUT=$ROOT/experiments/peginsertion_classical
STARTED=$ROOT/CLASSICAL_STARTED
[ -f "$STARTED" ] && exit 0
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ -z "$FREE" ] && exit 0
[ "$FREE" -lt 40000 ] && exit 0                              # GPU busy (laguna) — wait
# --- GPU free: run the classical baseline once ---
touch "$STARTED"
mkdir -p "$OUT/demos"
ts=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/logs/classical_$ts.log
cd "$ROOT/benchmarks/ManiSkill" || exit 1
# run detached so cron returns; wrapper writes CLASSICAL_DONE + a results.json on completion
setsid bash -c "
  $PY -m mani_skill.examples.motionplanning.panda.run -e PegInsertionSide-v1 -n 100 \
      -b gpu --save-video --record-dir '$OUT/demos' --traj-name classical > '$LOG' 2>&1
  rate=\$(grep -oiE 'success[^0-9]*[0-9.]+' '$LOG' | tail -1)
  printf '{\"task\":\"PegInsertionSide-v1\",\"method\":\"classical (mplib RRTConnect+screw)\",\"n\":100,\"raw_success_line\":\"%s\",\"log\":\"%s\"}\n' \"\$rate\" '$LOG' > '$OUT/results.json'
  echo \"\$(date -u +%Y-%m-%dT%H:%M:%SZ) done\" > '$ROOT/CLASSICAL_DONE'
" </dev/null >/dev/null 2>&1 &
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) GPU free (${FREE}MB) -> launched classical baseline ($ts)"
