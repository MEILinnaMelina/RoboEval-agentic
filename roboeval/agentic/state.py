"""Convert RoboEval runtime state into JSON-friendly observations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from roboeval.const import HandSide
from roboeval.envs.props.prop import Prop
from roboeval.utils.physics_utils import distance


def _as_list(value: Any, digits: int = 5) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return round(float(arr), digits)
    return np.round(arr.astype(float), digits).tolist()


def _pose_of_body(body: Any) -> dict[str, Any]:
    quat = body.get_quaternion()
    return {
        "position": _as_list(body.get_position()),
        "quaternion_wxyz": _as_list(quat),
        "euler_xyz": _as_list(Rotation.from_quat(quat, scalar_first=True).as_euler("xyz")),
    }


def _pose_of_prop(prop: Prop) -> dict[str, Any]:
    data = _pose_of_body(prop.body)
    if prop.aabb is not None:
        data["aabb_position"] = _as_list(prop.aabb.get_position())
        data["aabb_size"] = _as_list(prop.aabb.mjcf.size)
    elif prop.bbox is not None:
        # No "boundary" site on this prop's XML (e.g. cube/blocks) - fall back
        # to the real per-geom bounding box computed at runtime.
        prop.bbox.update()
        size = prop.bbox.max - prop.bbox.min
        if np.all(np.isfinite(size)):
            data["aabb_position"] = _as_list((prop.bbox.max + prop.bbox.min) / 2.0)
            data["aabb_size"] = _as_list(size)
    if prop.sites:
        data["sites"] = [
            {
                "name": site.mjcf.name,
                "position": _as_list(site.get_position()),
            }
            for site in prop.sites
        ]
    return data


def get_task_objects(env: Any) -> dict[str, Any]:
    """Return named task objects for the initial task subset."""

    objects: dict[str, Any] = {}
    if hasattr(env, "kitchenpot"):
        objects["kitchenpot"] = env.kitchenpot
    if hasattr(env, "cube"):
        objects["cube"] = env.cube
    if hasattr(env, "blocks"):
        for idx, block in enumerate(env.blocks):
            objects[f"block_{idx}"] = block
    if hasattr(env, "packing_box"):
        objects["packing_box"] = env.packing_box
    if hasattr(env, "valves"):
        for idx, valve in enumerate(env.valves):
            objects[f"valve_{idx}"] = valve
    return objects


def get_object_position(obj: Any) -> np.ndarray:
    """Get a representative world position for a prop-like object."""

    if isinstance(obj, Prop):
        return obj.body.get_position()
    if hasattr(obj, "get_position"):
        return obj.get_position()
    if hasattr(obj, "body"):
        return obj.body.get_position()
    raise TypeError(f"Cannot read position from object of type {type(obj).__name__}")


def collect_env_state(env: Any, info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect a compact state summary for debugging and LLM prompting."""

    info = info or env.get_info()
    arm_qpos = env.robot.qpos_actuated[:-len(env.robot.grippers)]
    ee_pose = env.robot.forward_kinematics(arm_qpos)
    objects = get_task_objects(env)

    grippers = {}
    for idx, (side, gripper) in enumerate(env.robot.grippers.items()):
        pose_start = idx * 6
        grippers[side.name.lower()] = {
            "qpos": round(float(gripper.qpos), 5),
            "aperture_m": round(float(gripper.aperture_m), 5),
            "wrist_position": _as_list(gripper.wrist_position),
            "pinch_position": _as_list(gripper.pinch_position),
            "controlled_ee_pose": _as_list(ee_pose[pose_start : pose_start + 6]),
            "holding": {
                name: bool(env.robot.is_gripper_holding_object(obj, side))
                for name, obj in objects.items()
            },
        }

    object_states = {}
    for name, obj in objects.items():
        if isinstance(obj, Prop):
            object_states[name] = _pose_of_prop(obj)
        elif hasattr(obj, "body"):
            object_states[name] = _pose_of_body(obj.body)

    object_distances = {}
    for object_name, obj in objects.items():
        obj_body = obj.body if hasattr(obj, "body") else obj
        for side, gripper in env.robot.grippers.items():
            key = f"{object_name}_to_{side.name.lower()}_gripper"
            object_distances[key] = round(float(distance(obj_body, gripper.body)), 5)

    state = {
        "task_name": env.task_name,
        "seed": env.seed,
        "control_frequency": env.control_frequency,
        "action_space": {
            "shape": list(env.action_space.shape),
            "low_min": round(float(np.nanmin(env.action_space.low)), 5),
            "high_max": round(float(np.nanmax(env.action_space.high)), 5),
        },
        "robot": {
            "qpos_len": int(len(env.robot.qpos)),
            "qvel_len": int(len(env.robot.qvel)),
            "qpos_actuated_len": int(len(env.robot.qpos_actuated)),
            "qpos_actuated": _as_list(env.robot.qpos_actuated),
            "grippers": grippers,
        },
        "objects": object_states,
        "object_distances": object_distances,
        "metrics": _jsonify_metrics(info),
    }
    table = getattr(env, "table", None)
    if table is not None:
        # Not a graspable task object (kept out of get_task_objects/objects)
        # - just table-height/extent context for collision-aware reasoning.
        state["table"] = _pose_of_prop(table) if isinstance(table, Prop) else _pose_of_body(table.body)
    return state


def _jsonify_metrics(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify_metrics(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_metrics(v) for v in value]
    if isinstance(value, np.ndarray):
        return _as_list(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
