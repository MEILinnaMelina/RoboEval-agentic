"""Geometry-derived grasp and placement candidate generation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from roboeval.agentic_v2.state import collect_scene_state
from roboeval.agentic_v2.types import (
    AllowedContactPolicy,
    AllowedContactRule,
    GraspCandidate,
    Pose,
    SceneState,
)


FINGER_PAD_OFFSET = np.array([0.0, 0.0, 0.1034])
MAXIMUM_APERTURE = 0.08
POT_HANDLE_BAR_LOCAL = {
    # Centers of the outer horizontal handle-bar collision meshes.
    "left": np.array([0.0014, 0.2338, 0.041]),
    "right": np.array([0.0019, -0.2457, 0.041]),
}


class CandidateGenerator:
    def __init__(self, env: Any) -> None:
        self.env = env

    def grasp_candidates(
        self,
        object_name: str,
        side: str,
        state: SceneState | None = None,
    ) -> tuple[GraspCandidate, ...]:
        state = state or collect_scene_state(self.env)
        if object_name not in state.objects:
            raise ValueError(f"unknown task object {object_name!r}")
        if side not in state.robot.arms:
            raise ValueError(f"unknown arm side {side!r}")
        if object_name == "kitchenpot":
            return self._pot_handle_candidates(side, state)
        return self._top_down_candidates(object_name, side, state)

    def _top_down_candidates(
        self,
        object_name: str,
        side: str,
        state: SceneState,
    ) -> tuple[GraspCandidate, ...]:
        obj = state.objects[object_name]
        center = np.asarray(obj.pose.position)
        # Aperture must be checked against the object's fixed physical size,
        # not the live world AABB, which inflates as soon as the object
        # tilts even slightly and would spuriously reject a fitting axis.
        size = np.asarray(obj.canonical_size)
        # canonical_size lives in the object's own body frame; gap_axis below
        # is a world-frame direction, so it must be rotated into the body
        # frame before comparing against size - otherwise this is only
        # correct by accident for objects that happen to be world-aligned.
        object_rotation = np.asarray(obj.pose.as_matrix())[:3, :3]
        candidates: list[GraspCandidate] = []
        for index, yaw in enumerate((-np.pi / 2.0, np.pi / 2.0, 0.0, np.pi)):
            rotation = Rotation.from_euler("xyz", (np.pi, 0.0, yaw))
            gap_axis = rotation.apply((0.0, 1.0, 0.0))
            gap_axis_local = object_rotation.T @ gap_axis
            required_aperture = float(np.dot(np.abs(gap_axis_local), size) + 0.012)
            if required_aperture > MAXIMUM_APERTURE:
                continue
            approach_axis = rotation.apply((0.0, 0.0, 1.0))
            wrist = center - rotation.apply(FINGER_PAD_OFFSET)
            pregrasp = wrist - approach_axis * 0.08
            grasp_pose = self._pose(wrist, rotation)
            pregrasp_pose = self._pose(pregrasp, rotation)
            candidates.append(
                GraspCandidate(
                    name=f"{object_name}_{side}_top_{index}",
                    object_name=object_name,
                    side=side,
                    pregrasp_pose=pregrasp_pose,
                    grasp_pose=grasp_pose,
                    approach_axis=tuple(approach_axis),
                    required_aperture=required_aperture,
                    contact_policy=self._grasp_policy(object_name, side),
                    score=self._score(pregrasp_pose, state.robot.arms[side].ee_pose),
                )
            )
        return tuple(sorted(candidates, key=lambda candidate: candidate.score))

    def _pot_handle_candidates(
        self,
        side: str,
        state: SceneState,
    ) -> tuple[GraspCandidate, ...]:
        pot = state.objects["kitchenpot"]
        pot_matrix = pot.pose.as_matrix()
        result: list[GraspCandidate] = []
        handle_point = pot_matrix @ np.append(POT_HANDLE_BAR_LOCAL[side], 1.0)
        inward = np.array([0.0, -1.0 if side == "left" else 1.0, 0.0])
        gap = np.array([0.0, 0.0, 1.0])
        lateral = np.cross(gap, inward)
        base = np.column_stack((lateral, gap, inward))
        for orientation, matrix in enumerate(
            (base, np.column_stack((-lateral, -gap, inward)))
        ):
            rotation = Rotation.from_matrix(matrix)
            approach_axis = rotation.apply((0.0, 0.0, 1.0))
            wrist = handle_point[:3] - rotation.apply(FINGER_PAD_OFFSET)
            pregrasp = wrist - approach_axis * 0.07
            grasp_pose = self._pose(wrist, rotation)
            pregrasp_pose = self._pose(pregrasp, rotation)
            result.append(
                GraspCandidate(
                    name=f"kitchenpot_{side}_side_handle_{orientation}",
                    object_name="kitchenpot",
                    side=side,
                    pregrasp_pose=pregrasp_pose,
                    grasp_pose=grasp_pose,
                    approach_axis=tuple(approach_axis),
                    required_aperture=0.037,
                    contact_policy=self._grasp_policy("kitchenpot", side),
                    score=self._score(pregrasp_pose, state.robot.arms[side].ee_pose),
                )
            )
        return tuple(sorted(result, key=lambda candidate: candidate.score))

    @staticmethod
    def _grasp_policy(object_name: str, side: str) -> AllowedContactPolicy:
        return AllowedContactPolicy(
            rules=(
                AllowedContactRule(f"robot:{side}:finger", f"object:{object_name}"),
                # A fingertip grazing the support surface while closing on a
                # short object resting on it is normal tabletop picking, not
                # a crash (reproduced on a 4cm block at -0.1mm, blocking an
                # otherwise-correct regrasp). Tolerate it at a few mm only;
                # wrist/link-table contact stays forbidden.
                AllowedContactRule(
                    f"robot:{side}:finger", "scene:*table*", penetration_tolerance=0.004
                ),
            ),
            penetration_tolerance=0.008,
        )

    @staticmethod
    def _pose(position: np.ndarray, rotation: Rotation) -> Pose:
        xyzw = rotation.as_quat()
        return Pose(tuple(position), (xyzw[3], xyzw[0], xyzw[1], xyzw[2]))

    @staticmethod
    def _score(target: Pose, current: Pose) -> float:
        distance = float(np.linalg.norm(np.asarray(target.position) - np.asarray(current.position)))
        target_rotation = Rotation.from_matrix(target.as_matrix()[:3, :3])
        current_rotation = Rotation.from_matrix(current.as_matrix()[:3, :3])
        return distance + 0.05 * float((current_rotation.inv() * target_rotation).magnitude())
