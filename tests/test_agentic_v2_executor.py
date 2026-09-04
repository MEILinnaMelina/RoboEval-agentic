from __future__ import annotations

import pytest

from roboeval.agentic_v2.executor import CLOSE_COMMAND, OPEN_COMMAND, JointActionAdapter
from roboeval.agentic_v2.skills.base import SkillContext
from roboeval.agentic_v2.state import collect_scene_state
from roboeval.agentic_v2.task_specs import make_task_env


def test_joint_action_adapter_has_14_plus_2_layout() -> None:
    env = make_task_env("cube_handover")
    env.reset(seed=0)
    try:
        adapter = JointActionAdapter(env)
        joints = collect_scene_state(env).robot.joint_positions
        adapter.set_gripper("left", OPEN_COMMAND)
        adapter.set_gripper("right", CLOSE_COMMAND)
        action = adapter.build(joints)
        assert action.shape == (16,)
        assert tuple(action[-2:]) == (OPEN_COMMAND, CLOSE_COMMAND)
    finally:
        env.close()


def test_joint_action_adapter_rejects_non_binary_gripper_command() -> None:
    env = make_task_env("cube_handover")
    env.reset(seed=0)
    try:
        adapter = JointActionAdapter(env)
        with pytest.raises(ValueError, match="0 .* or 1"):
            adapter.set_gripper("left", 0.5)
    finally:
        env.close()


def test_executor_stops_cleanly_when_observed_condition_is_reached() -> None:
    env = make_task_env("cube_handover")
    env.reset(seed=0)
    try:
        executor = SkillContext.create(env).executor
        plan = executor.hold_plan("observed_stop", steps=10)
        report = executor.execute(
            plan,
            require_holds=False,
            stop_condition=lambda state: True,
        )
        assert report.success
        assert report.executed_points == 1
    finally:
        env.close()
