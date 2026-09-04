from __future__ import annotations

import numpy as np

from roboeval.agentic_v2.motion.candidate_generator import CandidateGenerator
from roboeval.agentic_v2.model import geom_labels, safe_name
from roboeval.agentic_v2.task_specs import make_task_env


def test_rod_candidates_filter_the_too_wide_axis() -> None:
    env = make_task_env("cube_handover")
    env.reset(seed=0)
    try:
        candidates = CandidateGenerator(env).grasp_candidates("cube", "right")
        assert len(candidates) == 2
        assert all(candidate.required_aperture < 0.08 for candidate in candidates)
        assert all(abs(candidate.required_aperture - 0.052) < 1e-6 for candidate in candidates)
        assert all(candidate.pregrasp_pose.position[2] > candidate.grasp_pose.position[2] for candidate in candidates)
        labels = geom_labels(env)
        model = env.mojo.physics.model
        right_pad_ids = [
            geom_id for geom_id in range(model.ngeom)
            if "nohand_right" in (model.id2name(geom_id, "geom") or "")
            and "fingertip_pad" in (model.id2name(geom_id, "geom") or "")
        ]
        assert right_pad_ids
        assert all(labels[geom_id] == "robot:right:finger" for geom_id in right_pad_ids)
    finally:
        env.close()


def test_block_candidates_are_derived_from_current_geometry() -> None:
    env = make_task_env("stack_two_blocks")
    env.reset(seed=0)
    try:
        candidates = CandidateGenerator(env).grasp_candidates("block_0", "right")
        assert len(candidates) == 4
        assert all(abs(candidate.grasp_pose.position[0] - 0.5) < 1e-6 for candidate in candidates)
        assert all(abs(candidate.required_aperture - 0.052) < 1e-6 for candidate in candidates)
    finally:
        env.close()


def test_pot_handle_affordance_is_transformed_from_object_pose() -> None:
    env = make_task_env("lift_pot")
    env.reset(seed=0)
    try:
        generator = CandidateGenerator(env)
        left = generator.grasp_candidates("kitchenpot", "left")
        right = generator.grasp_candidates("kitchenpot", "right")
        assert len(left) == len(right) == 2
        assert all(item.pregrasp_pose.position[1] > item.grasp_pose.position[1] for item in left)
        assert all(item.pregrasp_pose.position[1] < item.grasp_pose.position[1] for item in right)
        assert all(abs(item.required_aperture - 0.037) < 1e-6 for item in left + right)
        assert left[0].contact_policy.allows(
            "robot:left:finger", "object:kitchenpot", -0.001
        )
    finally:
        env.close()


def test_anonymous_finger_meshes_are_not_labeled_as_arm_links() -> None:
    env = make_task_env("cube_handover")
    env.reset(seed=0)
    try:
        labels = geom_labels(env)
        model = env.mojo.physics.model
        anonymous_finger_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if "finger" in safe_name(
                model, int(model.geom_bodyid[geom_id]), "body"
            ).lower()
            and "unnamed_geom" in safe_name(model, geom_id, "geom").lower()
        ]
        assert anonymous_finger_ids
        assert all(
            labels[geom_id].endswith(":finger")
            for geom_id in anonymous_finger_ids
        )
    finally:
        env.close()
