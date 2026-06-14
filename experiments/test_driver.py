"""Local (Mac, no-GPU) tests for the SweepRunner driver + autoresearch plumbing.

Covers the logic that bit us on the A100 (command building per CLI style, resume/skip,
and the launch->extract->results.jsonl path) so it's caught in <1s locally instead of a
30-min remote round-trip. Real ManiSkill training is NVIDIA/Vulkan-only and not tested here.

    .venv/bin/python -m pytest experiments/test_driver.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from autoresearch.results import load_results, log_experiment
from autoresearch.sweep_runner import IterPlan, NullTriageMonitor, SweepRunner
from experiments.autoresearch import SchedulePlanner, build_cmd, env_short

FAKE = Path("benchmarks/ManiSkill/examples/baselines/ppo/ppo.py")


def _sched(**configs: dict[str, Any]) -> dict[str, Any]:
    return {"task": "t", "env_id": "PickCube-v1", "obs": "state", "configs": configs}


@pytest.mark.parametrize("env_id,expected", [("PickCube-v1", "pickcube"), ("PegInsertionSide-v1", "peginsertionside")])
def test_env_short(env_id, expected):
    assert env_short(env_id) == expected


class TestBuildCmd:
    """Command building must match each baseline's CLI dialect."""

    def test_ppo_uses_tyro_dashes_and_track(self):
        cmd = build_cmd("ppo", FAKE, {"num_envs": 1024}, 3, "PickCube-v1", "state", "ppo-pickcube-state-s3")
        s = " ".join(cmd)
        assert cmd[0] == sys.executable and cmd[1:3] == ["-u", "ppo.py"]  # absolute interp, not bare "python"
        assert "--num_envs=1024" in s and "--seed=3" in s
        assert "--track" in s and "--wandb_project_name=wm-manip" in s
        assert "wandb=true" not in s  # hydra style must not leak into tyro cmd

    def test_tdmpc2_uses_hydra_keyvals(self):
        cmd = build_cmd("tdmpc2", Path("x/train.py"), {"model_size": 5}, 2, "PickCube-v1", "state", "tdmpc2-pickcube-state-s2")
        s = " ".join(cmd)
        assert "model_size=5" in s and "seed=2" in s
        assert "wandb=true" in s and "wandb_name=tdmpc2-pickcube-state-s2" in s
        assert "--seed" not in s  # tyro dashes must not leak into hydra cmd


class TestResumeSkip:
    """Planner skips seeds already logged BASELINE/KEEP; re-plans the rest."""

    def test_skips_logged_seed_only(self, tmp_path):
        log_experiment(experiments_dir=str(tmp_path), tag="t", config_name="ppo",
                       status="BASELINE", score=1.0, extra={"method": "ppo", "seed": 1})
        sched = _sched(ppo={"launcher": str(FAKE), "overrides": {"num_envs": 8}, "seeds": [1, 2]})
        planner = SchedulePlanner(sched, dry_run=True, experiments_dir=str(tmp_path))
        seeds = [p.sidecar_payload["seed"] for p in planner.plan_iters([])]
        assert seeds == [2]  # seed 1 skipped (already done), seed 2 planned

    def test_plans_all_when_history_empty(self, tmp_path):
        sched = _sched(ppo={"launcher": str(FAKE), "overrides": {}, "seeds": [1, 2, 3]})
        planner = SchedulePlanner(sched, dry_run=True, experiments_dir=str(tmp_path))
        assert [p.sidecar_payload["seed"] for p in planner.plan_iters([])] == [1, 2, 3]


class _StubExtractor:
    """Stand-in for WandbResultExtractor — no W&B, returns a canned row."""

    def extract(self, plan: IterPlan, run_id: str | None, exit_code: int) -> list[dict[str, Any]]:
        return [{"config_name": plan.config_name, "status": "KEEP", "score": 0.5, "steps": 123}]


def test_sweeprunner_launch_to_results_jsonl(tmp_path):
    """End-to-end plumbing: real subprocess launch -> extract -> results.jsonl row."""
    plan = IterPlan(cmd=[sys.executable, "-c", "print('stub train ok')"], description="stub", config_name="stub")
    outcome = SweepRunner.run_one(plan, tag="smoke", extractor=_StubExtractor(),
                                  triage=NullTriageMonitor(), experiments_dir=str(tmp_path))
    assert outcome.exit_code == 0 and outcome.kill_reason is None
    rows = load_results(str(tmp_path), "smoke", "stub")
    assert len(rows) == 1
    assert rows[0]["score"] == 0.5 and rows[0]["status"] == "KEEP" and rows[0]["steps"] == 123
