from __future__ import annotations

from types import SimpleNamespace

from roboeval.agentic_v2.skills.base import SkillContext
from roboeval.agentic_v2.skills.grasp import GraspSkill
from roboeval.agentic_v2.task_specs import make_task_env
from roboeval.agentic_v2.types import (
    AllowedContactPolicy,
    GraspCandidate,
    Pose,
    SkillName,
    SkillRequest,
)


def request() -> SkillRequest:
    return SkillRequest(
        SkillName.GRASP,
        "cube",
        {"right": "cube"},
        "establish a verified donor hold",
    )


def test_real_rod_grasp_survives_monitored_verification_lift() -> None:
    env = make_task_env("cube_handover")
    env.reset(seed=0)
    try:
        result = GraspSkill(SkillContext.create(env)).execute(request())
        assert result.success
        assert result.failure_code is None
        assert result.state.objects["cube"].held_by == ("right",)
        assert any("verification_lift" in report.plan_name for report in result.execution_reports)
    finally:
        env.close()


def test_unreachable_pregrasp_never_closes_gripper() -> None:
    env = make_task_env("cube_handover")
    env.reset(seed=0)
    try:
        context = SkillContext.create(env)
        unreachable = GraspCandidate(
            "unreachable",
            "cube",
            "right",
            Pose((10, 10, 10), (1, 0, 0, 0)),
            Pose((10, 10, 9.9), (1, 0, 0, 0)),
            (0, 0, -1),
            0.05,
            AllowedContactPolicy(),
            0.0,
        )
        context.candidates = SimpleNamespace(
            grasp_candidates=lambda object_name, side, state: (unreachable,)
        )
        result = GraspSkill(context).execute(request())
        assert not result.success
        plan_names = [report.plan_name for report in result.execution_reports]
        assert plan_names == ["open_right_gripper"]
        assert not any(name.startswith("close_") for name in plan_names)
    finally:
        env.close()
