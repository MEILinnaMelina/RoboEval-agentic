from __future__ import annotations

import numpy as np

from roboeval.agentic_v2.constraints.bimanual import (
    ee_targets_for_object_pose,
)
from roboeval.agentic_v2.evaluator import benchmark_success
from roboeval.agentic_v2.skills.base import SkillContext
from roboeval.agentic_v2.skills.bimanual_grasp import BimanualGraspSkill
from roboeval.agentic_v2.skills.lift import LiftSkill
from roboeval.agentic_v2.task_specs import make_task_env
from roboeval.agentic_v2.types import HeldObjectAttachment, Pose, SkillName, SkillRequest


def test_bimanual_attachment_targets_preserve_relative_transforms() -> None:
    object_pose = Pose((0.5, 0.0, 1.1), (1.0, 0.0, 0.0, 0.0))
    attachments = (
        HeldObjectAttachment("kitchenpot", "left", Pose((0.0, -0.2, -0.1), (1, 0, 0, 0))),
        HeldObjectAttachment("kitchenpot", "right", Pose((0.0, 0.2, -0.1), (1, 0, 0, 0))),
    )

    targets = ee_targets_for_object_pose(object_pose, attachments)
    assert set(targets) == {"left", "right"}
    for attachment in attachments:
        observed = targets[attachment.side].inverse().compose(object_pose)
        assert np.allclose(observed.as_matrix(), attachment.ee_to_object.as_matrix())


def test_real_pot_bimanual_grasp_and_lift() -> None:
    env = make_task_env("lift_pot")
    env.reset(seed=0)
    try:
        context = SkillContext.create(env)
        grasp = BimanualGraspSkill(context).execute(
            SkillRequest(
                SkillName.BIMANUAL_GRASP,
                "kitchenpot",
                {"left": "left_handle", "right": "right_handle"},
                "secure both pot handles",
            )
        )
        assert grasp.success, grasp.to_dict()
        assert set(grasp.state.objects["kitchenpot"].held_by) == {"left", "right"}

        lift = LiftSkill(context).execute(
            SkillRequest(
                SkillName.LIFT,
                "kitchenpot",
                {"left": "left_handle", "right": "right_handle"},
                "lift the pot while preserving both grasps",
                "task_height",
            )
        )
        assert lift.success, lift.to_dict()
        assert benchmark_success(lift.state) == 1.0
    finally:
        env.close()
