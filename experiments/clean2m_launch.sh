#!/bin/bash
# Cron-driven, idempotent: when the shared GPU frees (laguna gone, >=40GB free),
# launch the CLEAN SEQUENTIAL 3-seed TD-MPC2 PegInsertion run to 2M. Sequential (the
# driver's default) avoids the parallel-contention timeout that cut the last attempt
# at ~1.5M; solo each seed finishes ~24h < the 30h cap, recording cleanly via the
# train/success_once extractor fix. Re-arms the peg_check verdict watcher.
set -u
ROOT=/home/ubuntu/robotics_world_models
PY=/home/ubuntu/miniconda3/envs/wm/bin/python
STARTED=$ROOT/CLEAN2M_STARTED
[ -f "$STARTED" ] && exit 0                                   # already launched once
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ -z "$FREE" ] && exit 0
[ "$FREE" -lt 40000 ] && exit 0                               # GPU still busy (laguna) — wait
# --- GPU is free: launch the clean run ---
touch "$STARTED"
cd "$ROOT" || exit 1
ts=$(date -u +%Y%m%dT%H%M%SZ)
# archive the killed parallel rows + stale verdict so this run is clean
[ -f experiments/peginsertion_floor/tdmpc2/results.jsonl ] && \
  mv experiments/peginsertion_floor/tdmpc2/results.jsonl experiments/peginsertion_floor/tdmpc2/results.jsonl.killed_$ts
rm -f PEGINSERTION_DONE peginsertion_verdict.md
# re-arm the verdict watcher (writes peginsertion_verdict.md when 3 valid seeds land)
( crontab -l 2>/dev/null | grep -v "peg_check.py"; \
  echo "*/10 * * * * /usr/bin/python3 $ROOT/peg_check.py >> $ROOT/logs/peg_cron.log 2>&1" ) | crontab -
# launch the driver: ppo skipped (BASELINE done), tdmpc2 seeds 1/2/3 run SEQUENTIALLY
setsid "$PY" -u -m experiments.autoresearch \
  --schedule configs/schedules/peginsertion_floor.yaml \
  </dev/null >> "$ROOT/logs/sweep_clean2m_$ts.log" 2>&1 &
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) GPU free (${FREE}MB) -> launched clean sequential 3-seed run ($ts)"
