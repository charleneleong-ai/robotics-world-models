"""Classical motion-planning robustness sweep under perceived-pose noise.

Vendors the ManiSkill PegInsertionSide motion-planning solution (RRTConnect +
screw), but injects per-episode Gaussian translation noise on the *perceived*
poses the planner reads (`env.peg.pose`, `env.goal_pose`). The physics object
stays at its true pose, so the open-loop plan targets a wrong location and the
peg misses the hole -> the brittleness signal we want to measure.

Must run under a numpy<2 interpreter because mplib 0.1.1 segfaults against
numpy 2.x. On the A100 box: /workspace/mplib_venv/bin/python (numpy 1.26.4).
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import mani_skill  # noqa: F401  (registers PegInsertionSide-v1)
import numpy as np
import sapien
import typer
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

ENV_ID = "PegInsertionSide-v1"
FINGER_LENGTH = 0.025


def _to_sapien(pose) -> sapien.Pose:
    """ManiSkill batched Pose -> single sapien.Pose (xyz, wxyz)."""
    raw = pose.raw_pose[0].cpu().numpy()
    return sapien.Pose(raw[:3], raw[3:])


def _noisy(pose, rng: np.random.Generator, sigma_m: float) -> sapien.Pose:
    """Perceived pose = true pose with Gaussian translation error."""
    sp = _to_sapien(pose)
    if sigma_m > 0:
        sp = sapien.Pose(sp.p + rng.normal(0.0, sigma_m, size=3), sp.q)
    return sp


def _solve_noisy(env, seed: int, sigma_m: float) -> bool:
    """Run one episode; planner sees pose estimates corrupted by N(0, sigma_m)."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    u = env.unwrapped

    # One fixed (wrong) perception estimate per pose source for this episode.
    peg_perceived = _noisy(u.peg.pose, rng, sigma_m)
    goal_perceived = _noisy(u.goal_pose, rng, sigma_m)

    obb = get_actor_obb(u.peg)
    approaching = np.array([0, 0, -1])
    target_closing = u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = u.agent.build_grasp_pose(approaching, closing, center)
    grasp_pose = grasp_pose * sapien.Pose(
        [-max(0.05, u.peg_half_sizes[0, 0].item() / 2 + 0.01), 0, 0]
    )

    # Reach
    if planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([0, 0, -0.05])) == -1:
        return False
    # Grasp
    if planner.move_to_pose_with_screw(grasp_pose) == -1:
        return False
    planner.close_gripper()

    # Align peg with the (perceived) hole -- the open-loop target.
    insert_pose = goal_perceived * peg_perceived.inv() * grasp_pose
    offset = sapien.Pose([-0.01 - u.peg_half_sizes[0, 0].item(), 0, 0])
    pre_insert_pose = insert_pose * offset
    if planner.move_to_pose_with_screw(pre_insert_pose) == -1:
        return False
    for _ in range(3):
        # Refinement also reads perceived peg pose (corrupted same way).
        delta_pose = goal_perceived * offset * _noisy(u.peg.pose, rng, sigma_m).inv()
        pre_insert_pose = delta_pose * pre_insert_pose
        if planner.move_to_pose_with_screw(pre_insert_pose) == -1:
            return False

    # Insert
    res = planner.move_to_pose_with_screw(insert_pose * sapien.Pose([0.05, 0, 0]))
    planner.close()
    if res == -1:
        return False
    return bool(res[-1]["success"].item())


def _sweep(env, sigma_mm: float, n: int) -> float:
    successes = 0
    sigma_m = sigma_mm / 1000.0
    for seed in range(n):
        try:
            ok = _solve_noisy(env, seed=seed, sigma_m=sigma_m)
        except Exception as exc:  # noqa: BLE001 -- a failed plan is a failed episode
            print(f"    seed {seed} sigma={sigma_mm}mm exception: {exc!r}")
            ok = False
        successes += int(ok)
        print(f"    sigma={sigma_mm}mm seed={seed} success={ok}", flush=True)
    return successes / n


def main(
    sigmas_mm: str = typer.Option("0,5,10,15,20", help="Comma-separated noise sigmas (mm)."),
    n: int = typer.Option(50, help="Episodes per sigma."),
    out: Path = typer.Option(Path("classical_results.json"), help="Output JSON path."),
) -> None:
    sigmas = [float(s) for s in sigmas_mm.split(",")]
    env = gym.make(
        ENV_ID,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    results = []
    for sigma_mm in sigmas:
        print(f"=== sigma = {sigma_mm} mm (n={n}) ===", flush=True)
        rate = _sweep(env, sigma_mm, n)
        print(f"=== sigma={sigma_mm}mm success_rate={rate:.3f} ===", flush=True)
        results.append({"sigma_mm": sigma_mm, "n": n, "success_rate": rate})
    env.close()

    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    typer.run(main)
