"""SweepRunner driver for the world-models manipulation sweeps.

Reads ``configs/schedules/<sweep>.yaml`` and launches each config x seed through
``autoresearch.SweepRunner`` with:

  - SchedulePlanner   — yields one IterPlan per (config, seed); waits for free GPU
                        before each launch (shared box) and skips iters already in
                        results.jsonl (resumable).
  - GPUTriageMonitor  — hang-only (wasted/undersized disabled: the configs are
                        intentionally different sizes — PPO ~76GB/high-util,
                        TD-MPC2 ~30GB/lower-util — so those kills would false-fire).
  - WandbResultExtractor — after each run, pulls eval/success_once + wandb_url from
                        the W&B run by name -> a results.jsonl row, logged natively.

Supersedes the ad-hoc bash queues + the harvest stopgap: results land in
experiments/<tag>/<config>/results.jsonl as runs finish.

    # run as a module from the repo root (a script named autoresearch.py would
    # otherwise shadow the installed autoresearch package on sys.path[0]):
    python -m experiments.autoresearch --schedule configs/schedules/pickcube_floor.yaml
    python -m experiments.autoresearch --schedule configs/schedules/pickcube_floor.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from autoresearch.gpu_monitor import GPUTriageThresholds
from autoresearch.results import load_results
from autoresearch.sweep_runner import GPUTriageMonitor, IterPlan, SweepRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wm-sweep")

REPO = Path(__file__).resolve().parents[1]          # ~/robotics_world_models
WANDB_PROJECT = "wm-manip"
# Per-config GPU memory the run needs free before we launch it (shared box).
NEED_MB = {"ppo": 60_000, "sac": 40_000, "tdmpc2": 35_000, "dreamerv3": 35_000}
# Model-free floors are BASELINE; world models are the methods under test.
BASELINE_METHODS = {"ppo", "sac"}
# argparse/tyro CLIs take --k=v; hydra CLIs take k=v.
HYDRA_CLIS = {"tdmpc2", "dreamerv3"}
TIMEOUT_MIN = {"ppo": 75, "sac": 120, "tdmpc2": 240, "dreamerv3": 360}


def env_short(env_id: str) -> str:
    return env_id.split("-")[0].lower()


def free_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()
    return int(out[0]) if out and out[0].strip().isdigit() else 0


def build_cmd(method: str, launcher: Path, overrides: dict[str, Any], seed: int,
              env_id: str, obs: str, wandb_name: str) -> list[str]:
    py = sys.executable  # absolute path to THIS interpreter (the wm env) — bare "python" isn't on PATH under setsid
    script = launcher.name
    if method in HYDRA_CLIS:  # hydra: key=value, wandb=true wandb_project=... wandb_name=...
        args = [f"{k}={v}" for k, v in overrides.items()]
        args += [f"seed={seed}", f"env_id={env_id}", f"obs={obs}", "exp_name=" + method,
                 "wandb=true", f"wandb_project={WANDB_PROJECT}", f"wandb_name={wandb_name}",
                 "setting_tag=walltime_efficient"]
    else:                     # tyro: --key=value, --track --wandb_project_name=...
        args = [f"--{k}={v}" for k, v in overrides.items()]
        args += [f"--env_id={env_id}", f"--seed={seed}", f"--exp_name={wandb_name}",
                 "--track", f"--wandb_project_name={WANDB_PROJECT}"]
    return [py, "-u", script, *args]


class SchedulePlanner:
    """Yields IterPlans from the schedule, GPU-gated and resumable."""

    def __init__(self, sched: dict[str, Any], dry_run: bool = False,
                 experiments_dir: str = "experiments") -> None:
        self.sched = sched
        self.dry_run = dry_run
        self.tag = sched["task"]
        self.experiments_dir = experiments_dir
        self.env_id = sched["env_id"]
        self.obs = sched.get("obs", "state")

    def _already_done(self, history: list[dict[str, Any]], method: str, seed: int) -> bool:
        return any(
            r.get("method") == method and r.get("seed") == seed
            and str(r.get("status", "")).upper() in {"BASELINE", "KEEP"}
            for r in history
        )

    def plan_iters(self, history: list[dict[str, Any]]) -> Iterator[IterPlan]:
        for method, cfg in self.sched["configs"].items():
            launcher = REPO / cfg["launcher"]
            overrides = cfg.get("overrides", {})
            cfg_hist = load_results(self.experiments_dir, self.tag, method)  # per-config, not tag-level
            for seed in cfg.get("seeds", [1]):
                if self._already_done(cfg_hist, method, seed):
                    log.info("skip %s seed %d (already in results.jsonl)", method, seed)
                    continue
                wandb_name = f"{method}-{env_short(self.env_id)}-state-s{seed}"
                cmd = build_cmd(method, launcher, overrides, seed, self.env_id, self.obs, wandb_name)
                if not self.dry_run:
                    need = NEED_MB.get(method, 35_000)
                    while free_mb() < need:
                        log.info("waiting for GPU: free=%dMB < %dMB (%s s%d)", free_mb(), need, method, seed)
                        time.sleep(120)
                log.info("PLAN %s seed %d -> %s", method, seed, " ".join(cmd))
                yield IterPlan(
                    cmd=cmd,
                    description=f"{method} {self.env_id} {self.obs} seed {seed}",
                    config_name=method,
                    timeout_min=cfg.get("timeout_min", TIMEOUT_MIN.get(method, 240)),  # per-config override
                    cwd=launcher.parent,
                    sidecar_payload={"wandb_name": wandb_name, "seed": seed, "method": method},
                )


EVAL_METRIC = "eval/success_once"
# World-model baselines (TD-MPC2 / DreamerV3) log train/success_once, not eval/* — accept
# either, else a converged world-model run is mis-recorded as metric-less (the bug that
# tagged solved TD-MPC2 seeds EARLY_KILL with score 0).
SUCCESS_METRICS = (EVAL_METRIC, "train/success_once")


def _success_metric(summary: Any) -> tuple[str | None, float]:
    """First success metric present in the summary as (key, value); (None, 0.0) if absent."""
    for k in SUCCESS_METRICS:
        if k in summary:
            return k, float(summary.get(k) or 0.0)
    return None, 0.0


def _default_wandb_api():
    import wandb
    return wandb.Api()


class WandbResultExtractor:
    """Pull a finished run's metrics from W&B by name -> a results row.

    Hardened against W&B sync lag: a run only reaches ``finished`` (and its summary
    fills in) on the backend *after* the training process exits + flushes, which can
    lag seconds-to-minutes under load. So poll with exponential backoff up to
    ``max_wait_s``, tolerate transient API/network errors (retry, don't crash the
    sweep), and only fall back to CRASH if no run ever appears. A clean (exit 0) run
    that exists but hasn't fully synced is logged with its best-available summary and
    a ``degraded`` note — never silently dropped (the bug that lost sac seed 5).

    ``api_factory`` / ``sleep`` are injectable for testing without real W&B or waits.
    """

    def __init__(self, entity_project: str, *, max_wait_s: float = 300.0,
                 api_factory=_default_wandb_api, sleep=time.sleep) -> None:
        self.entity_project = entity_project
        self.max_wait_s = max_wait_s
        self._api_factory = api_factory
        self._sleep = sleep

    def _find_run(self, name: str):
        try:
            runs = list(self._api_factory().runs(self.entity_project, filters={"display_name": name}))
        except Exception as e:  # noqa: BLE001 — transient API/network error; caller retries
            log.warning("wandb lookup failed for %s (will retry): %s", name, e)
            return None
        return max(runs, key=lambda r: getattr(r, "created_at", "") or "") if runs else None

    def extract(self, plan: IterPlan, run_id: str | None, exit_code: int) -> list[dict[str, Any]]:
        name = plan.sidecar_payload.get("wandb_name", "")
        method = plan.sidecar_payload.get("method", plan.config_name or "")
        seed = plan.sidecar_payload.get("seed", 0)
        base = {"method": method, "seed": seed, "config_name": method, "description": plan.description}

        if exit_code != 0:
            return [{**base, "status": "CRASH", "score": 0.0,
                     "notes": f"exit_code={exit_code}", "_relabel_target": True}]

        deadline = time.monotonic() + self.max_wait_s
        run = None
        delay = 10.0
        while True:
            run = self._find_run(name)
            ready = (run is not None and (run.state or "").lower() == "finished"
                     and any(k in run.summary for k in SUCCESS_METRICS))
            if ready or time.monotonic() >= deadline:
                break
            self._sleep(min(delay, 30.0))
            delay *= 1.5

        if run is None:
            return [{**base, "status": "CRASH", "score": 0.0,
                     "notes": f"no W&B run '{name}' found after {self.max_wait_s:.0f}s"}]

        s = run.summary
        metric_key, success_once = _success_metric(s)
        row = {
            **base,
            "status": "BASELINE" if method in BASELINE_METHODS else "KEEP",
            "score": success_once,
            "steps": int(s.get("global_step", s.get("_step", 0)) or 0),
            "runtime_min": round(float(s.get("_runtime", 0.0) or 0.0) / 60.0, 1),
            "wandb_url": run.url,
            "success_once": success_once,
            "success_metric": metric_key,
            "eval_success_once": success_once,  # back-compat alias
            "eval_success_at_end": float(s.get("eval/success_at_end", s.get("train/success_at_end", 0.0)) or 0.0),
            "eval_return": float(s.get("eval/return", s.get("train/return", 0.0)) or 0.0),
        }
        if (run.state or "").lower() != "finished" or metric_key is None:
            row["notes"] = f"degraded: state={run.state}, success_metric={metric_key}"
        return [row]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true", help="plan + print cmds, no GPU wait, no launch")
    args = ap.parse_args()

    sched = yaml.safe_load(args.schedule.read_text())
    tag = sched["task"]
    planner = SchedulePlanner(sched, dry_run=args.dry_run)

    if args.dry_run:
        hist = load_results("experiments", tag)
        for _ in planner.plan_iters(hist):
            pass
        log.info("dry-run complete (no runs launched)")
        return

    thresholds = GPUTriageThresholds(hang_util_pct=8, hang_window_s=300,
                                     wasted_util_pct=0, undersized_mem_pct=0)
    runner = SweepRunner(
        tag=tag,
        planner=planner,
        extractor=WandbResultExtractor(f"chaleong/{WANDB_PROJECT}"),
        triage=GPUTriageMonitor(thresholds=thresholds),
        experiments_dir="experiments",
        pause_between_iters_s=30,
    )
    result = runner.run()
    log.info("SWEEP DONE tag=%s iters=%d kills=%d blocked=%s",
             result.tag, result.iterations, result.kills, result.blocked)


if __name__ == "__main__":
    main()
