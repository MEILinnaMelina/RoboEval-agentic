from __future__ import annotations

import numpy as np
import pytest

from roboeval.agentic_v2.motion.collision_checker import data_signature, signatures_equal
from roboeval.agentic_v2.motion.ik import MultiStartIK
from roboeval.agentic_v2.state import collect_scene_state
from roboeval.agentic_v2.task_specs import make_task_env
from roboeval.agentic_v2.types import FailureCode, Pose


@pytest.fixture(scope="module")
def env_and_ik():
    env = make_task_env("cube_handover")
    env.reset(seed=0)
    try:
        yield env, MultiStartIK(env)
    finally:
        env.close()


def test_current_world_pose_is_reachable_and_does_not_mutate_live(env_and_ik) -> None:
    env, ik = env_and_ik
    state = collect_scene_state(env)
    before = data_signature(env.mojo.physics.data)
    result = ik.solve_candidates({"left": state.robot.arms["left"].ee_pose}, count=3)
    after = data_signature(env.mojo.physics.data)
    assert result.report.feasible
    assert result.accepted[0].position_error < 1e-4
    assert result.accepted[0].orientation_error < 1e-4
    assert signatures_equal(before, after)


def test_unreachable_target_has_structured_failure(env_and_ik) -> None:
    _, ik = env_and_ik
    target = Pose((10.0, 10.0, 10.0), (1.0, 0.0, 0.0, 0.0))
    result = ik.solve_candidates({"right": target}, count=3)
    assert not result.report.feasible
    assert result.report.failure_code is FailureCode.IK_UNREACHABLE
    assert all(not candidate.converged for candidate in result.rejected)


def test_paired_targets_are_solved_as_combined_configuration(env_and_ik) -> None:
    _, ik = env_and_ik
    state = collect_scene_state(ik.env)
    targets = {side: arm.ee_pose for side, arm in state.robot.arms.items()}
    result = ik.solve_candidates(targets, count=3)
    assert result.report.feasible
    assert len(result.accepted[0].joint_positions) == 14
    assert set(result.accepted[0].diagnostics["per_side"]) == {"left", "right"}


def test_multi_start_is_deterministic(env_and_ik) -> None:
    _, ik = env_and_ik
    target = collect_scene_state(ik.env).robot.arms["right"].ee_pose
    first = ik.solve_candidates({"right": target}, count=4)
    second = ik.solve_candidates({"right": target}, count=4)
    np.testing.assert_allclose(
        first.accepted[0].joint_positions,
        second.accepted[0].joint_positions,
        atol=1e-8,
    )


def test_single_arm_ik_freezes_the_uncommanded_arm(env_and_ik) -> None:
    env, ik = env_and_ik
    state = collect_scene_state(env)
    current = np.asarray(state.robot.joint_positions)
    target = state.robot.arms["left"].ee_pose
    result = ik.solve_candidates({"left": target}, count=6)
    assert result.report.feasible
    for candidate in result.accepted:
        np.testing.assert_allclose(candidate.joint_positions[7:], current[7:], atol=1e-8)
