"""Read-only conversion from a live RoboEval episode to typed scene state."""

from __future__ import annotations

from typing import Any

import numpy as np

from roboeval.agentic.state import get_task_objects
from roboeval.agentic_v2.model import body_velocity, contact_pairs, geom_labels, object_geom_ids
from roboeval.agentic_v2.task_specs import task_key_from_env
from roboeval.agentic_v2.types import ArmState, HeldObjectAttachment, ObjectState, Pose, RobotState, SceneState


def _metrics(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metrics(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _body_pose(body: Any) -> Pose:
    return Pose(tuple(body.get_position()), tuple(body.get_quaternion()))


def _object_extent(obj: Any) -> tuple[np.ndarray, np.ndarray]:
    if getattr(obj, "aabb", None) is not None:
        return (
            np.asarray(obj.aabb.get_position(), dtype=float),
            np.asarray(obj.aabb.mjcf.size, dtype=float),
        )
    bbox = getattr(obj, "bbox", None)
    if bbox is not None:
        bbox.update()
        minimum = np.asarray(bbox.min, dtype=float)
        maximum = np.asarray(bbox.max, dtype=float)
        if np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum)):
            return (minimum + maximum) / 2.0, maximum - minimum
    return np.asarray(obj.body.get_position(), dtype=float), np.zeros(3)


def _canonical_size(physics: Any, obj: Any) -> np.ndarray:
    """Rotation-invariant physical extent, from the object's own geoms in its
    body frame - unlike `_object_extent`, this does not inflate when the
    object tilts, so it is safe for gripper-aperture/fit checks."""

    bbox = getattr(obj, "bbox", None)
    body = getattr(bbox, "body", None) or getattr(obj, "body", None)
    if body is not None and getattr(body, "geoms", None):
        minimum = np.array([np.inf, np.inf, np.inf])
        maximum = np.array([-np.inf, -np.inf, -np.inf])
        for geom in body.geoms:
            aabb = np.asarray(physics.bind(geom.mjcf).aabb, dtype=float)
            center, half_size = aabb[:3], aabb[3:]
            minimum = np.minimum(minimum, center - half_size)
            maximum = np.maximum(maximum, center + half_size)
        if np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum)):
            return maximum - minimum
    site = getattr(bbox, "site", None)
    if site is not None:
        return np.asarray(physics.bind(site.mjcf).size, dtype=float) * 2.0
    return np.zeros(3)


def collect_scene_state(env: Any, info: dict[str, Any] | None = None) -> SceneState:
    """Collect current state without stepping or modifying the environment."""

    info = dict(env.get_info() if info is None else info)
    objects = get_task_objects(env)
    labels = geom_labels(env)
    object_geoms = object_geom_ids(env)
    contacts: dict[str, set[str]] = {name: set() for name in objects}
    for geom1, geom2, distance in contact_pairs(env.mojo.physics.data):
        if distance > 1e-8:
            continue
        for name, ids in object_geoms.items():
            if geom1 in ids:
                contacts[name].add(labels.get(geom2, f"geom:{geom2}"))
            if geom2 in ids:
                contacts[name].add(labels.get(geom1, f"geom:{geom1}"))

    object_states: dict[str, ObjectState] = {}
    for name, obj in objects.items():
        center, size = _object_extent(obj)
        canonical_size = _canonical_size(env.mojo.physics, obj)
        linear_velocity, angular_velocity = body_velocity(env, obj.body)
        held_by = tuple(
            side.name.lower()
            for side in env.robot.grippers
            if env.robot.is_gripper_holding_object(obj, side)
        )
        object_states[name] = ObjectState(
            name=name,
            pose=_body_pose(obj.body),
            aabb_center=tuple(center),
            size=tuple(size),
            linear_velocity=tuple(linear_velocity),
            angular_velocity=tuple(angular_velocity),
            contacts=tuple(sorted(contacts[name])),
            held_by=held_by,
            canonical_size=tuple(canonical_size),
        )

    arm_qpos = np.asarray(env.robot.qpos_actuated[:-len(env.robot.grippers)], dtype=float)
    arm_qvel = np.asarray(env.robot.qvel_actuated[:-len(env.robot.grippers)], dtype=float)
    last_action = np.asarray(env.action, dtype=float)
    arm_count = len(env.robot.grippers)
    joints_per_arm = len(arm_qpos) // arm_count
    arms: dict[str, ArmState] = {}
    for index, (side, gripper) in enumerate(env.robot.grippers.items()):
        key = side.name.lower()
        start = index * joints_per_arm
        stop = start + joints_per_arm
        command_index = len(last_action) - arm_count + index
        arms[key] = ArmState(
            side=key,
            joint_positions=tuple(arm_qpos[start:stop]),
            joint_velocities=tuple(arm_qvel[start:stop]),
            ee_pose=Pose(tuple(gripper.wrist_position), tuple(gripper.wrist_orientation.elements)),
            gripper_command=float(last_action[command_index]),
            gripper_aperture_m=float(gripper.aperture_m),
            holding=tuple(sorted(name for name, state in object_states.items() if key in state.held_by)),
        )

    return SceneState(
        task_key=task_key_from_env(env),
        task_name=str(env.task_name),
        seed=int(env.seed),
        control_frequency=int(env.control_frequency),
        action_shape=tuple(env.action_space.shape),
        robot=RobotState(
            joint_positions=tuple(arm_qpos),
            joint_velocities=tuple(arm_qvel),
            arms=arms,
        ),
        objects=object_states,
        metrics=_metrics(info),
    )


def infer_handover_roles(state: SceneState, object_name: str) -> dict[str, str]:
    """Infer donor and receiver from the live hold state."""

    obj = state.objects[object_name]
    if len(obj.held_by) != 1:
        raise ValueError(
            f"handover requires exactly one current holder; {object_name} is held by {obj.held_by}"
        )
    donor = obj.held_by[0]
    receiver = "right" if donor == "left" else "left"
    return {"donor": donor, "receiver": receiver}


def capture_attachment(state: SceneState, object_name: str, side: str) -> HeldObjectAttachment:
    """Capture T_ee_object from an observed verified grasp."""

    arm = state.robot.arms[side]
    obj = state.objects[object_name]
    ee_to_object = arm.ee_pose.inverse().compose(obj.pose)
    return HeldObjectAttachment(object_name=object_name, side=side, ee_to_object=ee_to_object)
