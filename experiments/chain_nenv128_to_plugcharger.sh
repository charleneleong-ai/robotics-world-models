#!/usr/bin/env bash
# Chain watcher: when the nenv128 sweep finishes, launch the PlugCharger TD-MPC2 sweep.
# Runs detached (PPID=1) — survives SSH disconnect.
set -u
ROOT=/home/ubuntu/robotics_world_models
PY=/home/ubuntu/miniconda3/envs/wm/bin/python
LOGDIR=$ROOT/logs
NENV128_LOG=$(ls -t $LOGDIR/sweep_nenv128_*.log 2>/dev/null | head -1)
CHAIN_LOG=$LOGDIR/chain_plugcharger_$(date -u +%Y%m%dT%H%M%SZ).log

mkdir -p "$LOGDIR"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$CHAIN_LOG"; }

log "watcher started, waiting for nenv128 sweep to finish..."
log "monitoring: ${NENV128_LOG:-none}"

# Wait for the nenv128 autoresearch process to exit
while pgrep -f "autoresearch.*peginsertion_nenv128" > /dev/null 2>&1; do
  sleep 60
done

log "nenv128 sweep finished, checking GPU..."
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
log "GPU free: ${FREE}MB"

if [ "$FREE" -lt 30000 ]; then
  log "GPU still busy, waiting 5min..."
  sleep 300
fi

log "launching PlugCharger TD-MPC2 sweep..."
cd "$ROOT" && set -a; . ./.env; set +a
PLUG_LOG=$LOGDIR/sweep_plugcharger_$(date -u +%Y%m%dT%H%M%SZ).log
setsid nohup $PY -m experiments.autoresearch --schedule configs/schedules/plugcharger_tdmpc2.yaml \
  </dev/null >>"$PLUG_LOG" 2>&1 & disown

sleep 3
log "PlugCharger sweep launched -> $PLUG_LOG"
pgrep -af "autoresearch.*plugcharger" | head -1 | tee -a "$CHAIN_LOG"
log "watcher done"
