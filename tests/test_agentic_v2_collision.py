from __future__ import annotations

import numpy as np
import pytest

from roboeval.agentic_v2.motion.collision_checker import (
    CollisionChecker,
    data_signature,
    signatures_equal,
)
from roboeval.agentic_v2.state import capture_attachment, collect_scene_state
from roboeval.agentic_v2.task_specs import make_task_env
from roboeval.agentic_v2.types import ConstraintSet, FailureCode


@pytest.fixture(scope="module")
def env_and_checker():
    env = make_task_env("cube_handover")
    env.reset(seed=0)
    try:
        yield env, CollisionChecker(env)
    finally:
        env.close()


def test_clone_and_checks_do_not_mutate_live_data(env_and_checker) -> None:
    env, checker = env_and_checker
    clone = checker.clone_live_data()
    assert clone.ptr is not env.mojo.physics.data.ptr
    assert not np.shares_memory(clone.qpos, env.mojo.physics.data.qpos)
    before = data_signature(env.mojo.physics.data)
    result = checker.check(collect_scene_state(env).robot.joint_positions)
    after = data_signature(env.mojo.physics.data)
    assert result.feasible
    assert signatures_equal(before, after)


def test_ten_safe_and_ten_colliding_in_limit_poses(env_and_checker) -> None:
    _, checker = env_and_checker
    rng = np.random.default_rng(20260904)
    expected = (
        True, False, True, True, True, True, True,
        False, False, True, True, True, False, True,
        False, False, True, False, False, False, False,
    )
    actual = tuple(
        checker.check(rng.uniform(checker.lower, checker.upper)).feasible
        for _ in expected
    )
    assert actual == expected
    assert sum(actual) >= 10
    assert len(actual) - sum(actual) >= 10


def test_joint_limit_rejection_is_structured(env_and_checker) -> None:
    _, checker = env_and_checker
    candidate = (checker.lower + checker.upper) / 2.0
    candidate[0] = checker.upper[0] + 0.01
    result = checker.check(candidate)
    assert not result.feasible
    assert result.failure_code is FailureCode.JOINT_LIMIT


def test_held_object_prediction_leaves_live_object_unchanged(env_and_checker) -> None:
    env, checker = env_and_checker
    state = collect_scene_state(env)
    attachment = capture_attachment(state, "cube", "right")
    before = data_signature(env.mojo.physics.data)
    checker.check(
        state.robot.joint_positions,
        ConstraintSet(held_objects=(attachment,)),
    )
    after = data_signature(env.mojo.physics.data)
    assert signatures_equal(before, after)
