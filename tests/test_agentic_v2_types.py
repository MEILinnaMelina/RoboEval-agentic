from __future__ import annotations

import pytest

from roboeval.agentic_v2.evaluator import assess_behavior_quality, benchmark_success
from roboeval.agentic_v2.types import (
    AllowedContactPolicy,
    AllowedContactRule,
    ArmState,
    ObjectState,
    Pose,
    RobotState,
    SceneState,
    FailureCode,
    SkillName,
    SkillRequest,
)


def make_state() -> SceneState:
    pose = Pose.identity()
    arms = {
        side: ArmState(side, (0.0,) * 7, (0.0,) * 7, pose, 0.0, 0.08)
        for side in ("left", "right")
    }
    return SceneState(
        task_key="stack_two_blocks",
        task_name="StackTwoBlocks",
        seed=3,
        control_frequency=20,
        action_shape=(16,),
        robot=RobotState((0.0,) * 14, (0.0,) * 14, arms),
        objects={
            "block_0": ObjectState("block_0", pose, (0.0, 0.0, 0.0), (0.04,) * 3),
            "block_1": ObjectState("block_1", pose, (0.0, 0.0, 0.0), (0.04,) * 3),
        },
        metrics={"task_success": 1.0, "subtask_progress": 1.0, "env_collision_count": 2},
    )


def test_scene_state_json_round_trip() -> None:
    state = make_state()
    restored = SceneState.from_json(state.to_json())
    assert restored == state


def test_skill_request_accepts_only_semantic_fields() -> None:
    request = SkillRequest.from_dict(
        {
            "skill": "handover",
            "object": "cube",
            "roles": {"donor": "right", "receiver": "left"},
            "goal": "transfer to the opposite gripper",
        }
    )
    assert request.skill is SkillName.HANDOVER
    assert request.object_name == "cube"


@pytest.mark.parametrize("field", ["qpos", "target_pose", "ee_offset", "yaw", "steps"])
def test_skill_request_rejects_low_level_fields(field: str) -> None:
    with pytest.raises(ValueError, match="low-level"):
        SkillRequest.from_dict({"skill": "grasp", "object": "cube", field: [0.0]})


def test_skill_request_rejects_bad_roles_and_missing_target() -> None:
    with pytest.raises(ValueError, match="invalid role"):
        SkillRequest(SkillName.GRASP, "cube", {"middle": "cube"})
    with pytest.raises(ValueError, match="requires"):
        SkillRequest(SkillName.GRASP)


def test_quality_does_not_override_raw_success() -> None:
    state = make_state()
    assert benchmark_success(state) == 1.0
    quality = assess_behavior_quality(state)
    assert quality["passed"] is False
    assert quality["checks"]["no_environment_collision"] is False


def test_allowed_contact_policy_is_symmetric_and_tolerance_bounded() -> None:
    policy = AllowedContactPolicy(
        (AllowedContactRule("robot:*:finger", "object:cube"),),
        penetration_tolerance=0.002,
    )
    assert policy.allows("robot:left:finger", "object:cube", -0.001)
    assert policy.allows("object:cube", "robot:right:finger", 0.0)
    assert not policy.allows("robot:left:link", "object:cube", 0.0)
    assert not policy.allows("robot:left:finger", "object:cube", -0.003)


def test_phase8_failure_codes_are_distinct() -> None:
    assert FailureCode.HANDOVER_REGION_EMPTY is not FailureCode.NO_VALID_GRASP
    assert FailureCode.PLACEMENT_UNREACHABLE is not FailureCode.PATH_BLOCKED
    assert FailureCode.RELEASE_FAILED.value == "RELEASE_FAILED"
    assert FailureCode.OTHER_OBJECT_COLLISION.value == "OTHER_OBJECT_COLLISION"
