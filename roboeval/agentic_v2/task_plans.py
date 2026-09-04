"""Task-level fixed semantic plans used to gate the robotics stack."""

from __future__ import annotations

from roboeval.agentic_v2.types import SkillName, SkillRequest


BASE_TASKS = ("cube_handover", "lift_pot", "stack_two_blocks")


_FIXED_PLANS: dict[str, tuple[SkillRequest, ...]] = {
    "cube_handover": (
        SkillRequest(
            SkillName.GRASP,
            object_name="cube",
            roles={"right": "grasping_arm"},
            goal="stable_hold",
        ),
        SkillRequest(
            SkillName.HANDOVER,
            object_name="cube",
            goal="transfer_to_other_arm",
        ),
        SkillRequest(SkillName.FINISH),
    ),
    "lift_pot": (
        SkillRequest(
            SkillName.BIMANUAL_GRASP,
            object_name="kitchenpot",
            roles={"left": "left_handle", "right": "right_handle"},
            goal="stable_bimanual_hold",
        ),
        SkillRequest(
            SkillName.LIFT,
            object_name="kitchenpot",
            roles={"left": "left_handle", "right": "right_handle"},
            goal="task_success_height",
        ),
        SkillRequest(SkillName.FINISH),
    ),
    "stack_two_blocks": (
        SkillRequest(
            SkillName.GRASP,
            object_name="block_0",
            goal="stable_hold",
            strategy="nearest_arm",
        ),
        SkillRequest(
            SkillName.HANDOVER,
            object_name="block_0",
            goal="transfer_for_opposite_workspace",
        ),
        SkillRequest(
            SkillName.PLACE,
            object_name="block_0",
            goal="on:block_1",
            strategy="stack",
        ),
        SkillRequest(SkillName.FINISH),
    ),
}


def fixed_plan(task_key: str) -> tuple[SkillRequest, ...]:
    """Return a pose-free, timing-free fixed plan for one base task."""

    try:
        return _FIXED_PLANS[task_key]
    except KeyError as error:
        raise ValueError(
            f"Agentic v2 supports only {', '.join(BASE_TASKS)}; got {task_key!r}"
        ) from error

