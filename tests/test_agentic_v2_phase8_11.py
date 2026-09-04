from __future__ import annotations

import json

import pytest

from roboeval.agentic_v2.evaluation import (
    aggregate_reports,
    paired_comparison,
    wilson_interval,
)
from roboeval.agentic_v2.llm_planner import parse_semantic_response
from roboeval.agentic_v2.prompts import SCHEMA_VERSION, build_planner_prompts
from roboeval.agentic_v2.replanner import OnlineReplanner
from roboeval.agentic_v2.skills.place import PlaceSkill
from roboeval.agentic_v2.task_plans import fixed_plan
from roboeval.agentic_v2.task_specs import TASK_SPECS
from roboeval.agentic_v2.types import (
    ArmState,
    ObjectState,
    Pose,
    RobotState,
    SceneState,
    SkillName,
)


def state(task_key: str = "stack_two_blocks") -> SceneState:
    pose = Pose.identity()
    arms = {
        side: ArmState(
            side,
            (0.0,) * 7,
            (0.0,) * 7,
            pose,
            0.0,
            0.08,
        )
        for side in ("left", "right")
    }
    objects = {
        "block_0": ObjectState(
            "block_0",
            Pose((0.2, -0.2, 1.0), (1.0, 0.0, 0.0, 0.0)),
            (0.2, -0.2, 1.0),
            (0.04, 0.04, 0.04),
        ),
        "block_1": ObjectState(
            "block_1",
            Pose((0.7, 0.2, 1.0), (1.0, 0.0, 0.0, 0.0)),
            (0.7, 0.2, 1.0),
            (0.04, 0.04, 0.04),
        ),
    }
    return SceneState(
        task_key=task_key,
        task_name=task_key,
        seed=0,
        control_frequency=20,
        action_shape=(16,),
        robot=RobotState((0.0,) * 14, (0.0,) * 14, arms),
        objects=objects,
        metrics={"task_success": 0.0, "subtask_progress": 0.0},
    )


def response(request: dict) -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "thought": "Use the verified semantic skill.",
            "request": request,
        }
    )


def test_fixed_plans_are_semantic_and_stack_uses_handover() -> None:
    forbidden = {
        "pose",
        "offset",
        "yaw",
        "qpos",
        "joint_positions",
        "steps",
        "gain",
        "tolerance",
    }
    for task_key in TASK_SPECS:
        plan = fixed_plan(task_key)
        assert plan[-1].skill is SkillName.FINISH
        assert not any(forbidden & set(item.to_dict()) for item in plan)
    assert [item.skill for item in fixed_plan("stack_two_blocks")] == [
        SkillName.GRASP,
        SkillName.HANDOVER,
        SkillName.PLACE,
        SkillName.FINISH,
    ]


def test_planner_response_rejects_low_level_and_extra_fields() -> None:
    valid = {
        "skill": "grasp",
        "object": "block_0",
        "roles": {"left": "grasping_arm"},
        "goal": "stable_hold",
        "strategy": None,
    }
    decision = parse_semantic_response(
        response(valid),
        provider="test",
        model="test",
    )
    assert decision.request.skill is SkillName.GRASP
    invalid = {**valid, "target_pose": [0.0] * 7}
    with pytest.raises(ValueError, match="low-level"):
        parse_semantic_response(
            response(invalid),
            provider="test",
            model="test",
        )


def test_prompt_contains_typed_feedback_but_no_control_authority() -> None:
    system, user = build_planner_prompts(
        TASK_SPECS["stack_two_blocks"],
        state(),
    )
    payload = json.loads(user)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert "joint values" in system
    assert "target_pose" not in payload["output_schema"]["properties"]["request"]["properties"]
    assert set(payload["current_state"]["objects"]) == {"block_0", "block_1"}


def test_online_replanner_uses_fresh_state_and_records_history() -> None:
    class Client:
        provider = "test"
        model = "semantic-test"

        def complete(self, system_prompt, user_prompt):
            return response(
                {
                    "skill": "grasp",
                    "object": "block_0",
                    "roles": {},
                    "goal": "stable_hold",
                    "strategy": "nearest_arm",
                }
            ), {"input_tokens": 10, "output_tokens": 5}

    planner = OnlineReplanner(Client())
    decision = planner.next_request(
        TASK_SPECS["stack_two_blocks"],
        state(),
        None,
    )
    assert decision.request.object_name == "block_0"
    assert decision.usage["input_tokens"] == 10


def test_placement_candidates_derive_height_and_support_contact() -> None:
    candidates = PlaceSkill.placement_candidates(
        "block_0",
        "block_1",
        state(),
    )
    assert len(candidates) > 1
    expected_z = 1.0 + 0.5 * (0.04 + 0.04) + 0.001
    assert candidates[0].placed_object_pose.position[2] == pytest.approx(expected_z)
    assert candidates[0].contact_policy.allows(
        "object:block_0",
        "object:block_1",
    )


def test_phase11_uses_raw_success_confidence_and_paired_seeds() -> None:
    reports = [
        {
            "method": method,
            "task_key": "lift_pot",
            "seed": seed,
            "benchmark_success": success,
            "subtask_progress": success,
            "behavior_quality": {"passed": False},
            "metrics": {"env_collision_count": 1},
            "failure_code": None if success else "POSTCONDITION_FAILED",
        }
        for method, seed, success in (
            ("v1-p22-independent", 0, 0.0),
            ("v2-full", 0, 1.0),
            ("v1-p22-independent", 1, 1.0),
            ("v2-full", 1, 1.0),
        )
    ]
    low, high = wilson_interval(8, 10)
    assert 0.0 < low < 0.8 < high < 1.0
    rows = aggregate_reports(reports)
    assert next(row for row in rows if row["method"] == "v2-full")["successes"] == 2
    paired = paired_comparison(reports)[0]
    assert paired["paired_seeds"] == 2
    assert paired["mean_success_delta"] == pytest.approx(0.5)

