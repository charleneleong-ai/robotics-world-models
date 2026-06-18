#!/usr/bin/env python3
"""One-shot, cron-driven check for the PegInsertionSide floor (ppo+tdmpc2, 3 seeds each).
Idempotent: emits the verdict markdown once (guarded by the DONE marker), then no-ops.
Fires when every seed has a terminal row; scores ONLY valid (BASELINE/KEEP) seeds and
marks killed/crashed ones invalid (not a fake 0.0). stdlib only; cron-driven for durability."""
import json
import sys
import pathlib
import datetime
import statistics

BASE = pathlib.Path.home() / "robotics_world_models/experiments/peginsertion_floor"
OUT = pathlib.Path.home() / "robotics_world_models/peginsertion_verdict.md"
DONE = pathlib.Path.home() / "robotics_world_models/PEGINSERTION_DONE"
NEED = {"ppo": 3, "tdmpc2": 3}
TERMINAL = {"BASELINE", "KEEP", "DISCARD", "EARLY_KILL", "CRASH"}
VALID = {"BASELINE", "KEEP"}


def rows(cfg):
    f = BASE / cfg / "results.jsonl"
    if not f.exists():
        return []
    out = []
    for ln in f.read_text().splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def latest_by_seed(cfg):
    """Last terminal row per seed (re-runs append, so last wins)."""
    by_seed = {}
    for r in rows(cfg):
        if str(r.get("status", "")).upper() in TERMINAL:
            by_seed[r.get("seed")] = r
    return by_seed


now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
if DONE.exists():
    sys.exit(0)

counts = {c: len(latest_by_seed(c)) for c in NEED}
if not all(counts[c] >= n for c, n in NEED.items()):
    print("[%s] waiting: %s need %s" % (now, counts, NEED))
    sys.exit(0)

md = [
    "# PegInsertionSide-v1 floor - world-model (TD-MPC2) vs model-free (PPO)",
    "",
    "_Auto-generated %s when all seeds reached a terminal state._" % now,
    "",
    "| method | seed | success_once | steps | runtime_min | status |",
    "|---|---|---|---|---|---|",
]
means = {}
for cfg in ("ppo", "tdmpc2"):
    valid_scores = []
    for seed, r in sorted(latest_by_seed(cfg).items(), key=lambda kv: kv[0] if kv[0] is not None else 0):
        status = str(r.get("status", "")).upper()
        s = float(r.get("score", r.get("success_once", 0.0)) or 0.0)
        if status in VALID:
            valid_scores.append(s)
            cell = "%.3f" % s
        else:
            cell = "INVALID (%s)" % status
        md.append("| %s | %s | %s | %s | %s | %s |" % (
            cfg, seed, cell, format(int(r.get("steps", 0) or 0), ","),
            r.get("runtime_min", "?"), status))
    means[cfg] = statistics.mean(valid_scores) if valid_scores else None

def fmt(m):
    return "%.3f" % m if m is not None else "no valid seeds"

ppo_m, td_m = means.get("ppo"), means.get("tdmpc2")
md += [
    "",
    "**Mean success_once (valid seeds only)** - PPO (50M-step floor): %s - TD-MPC2 (2M-step world model): %s" % (fmt(ppo_m), fmt(td_m)),
    "",
]
if td_m is not None and ppo_m is not None:
    md.append("**Headline:** on contact-rich PegInsertionSide, the world model (TD-MPC2) reaches "
              "%.2f success at 2M env steps vs the model-free PPO floor at %.2f at 50M - the "
              "sample-efficiency gap PickCube (trivial) cannot show." % (td_m, ppo_m))
else:
    md.append("**Incomplete:** not all seeds produced a valid (BASELINE/KEEP) result - see INVALID rows above.")

OUT.write_text("\n".join(md) + "\n")
DONE.write_text(now + "\n")
print("[%s] VERDICT WRITTEN -> %s" % (now, OUT))
