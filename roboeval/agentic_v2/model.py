"""MuJoCo model indexing helpers used without changing RoboEval internals."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import mujoco
import numpy as np

from roboeval.agentic.state import get_task_objects
from roboeval.utils.physics_utils import get_colliders


def element_id(physics: Any, element: Any) -> int:
    mjcf_element = getattr(element, "mjcf", element)
    return int(physics.bind(mjcf_element).element_id)


def geom_ids(physics: Any, obj: Any) -> set[int]:
    return {element_id(physics, geom) for geom in get_colliders(obj)}


def safe_name(model: Any, element_id_value: int, kind: str) -> str:
    name = model.id2name(int(element_id_value), kind)
    return str(name) if name else f"{kind}_{int(element_id_value)}"


def arm_joint_addresses(env: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return full-model qpos/qvel addresses in v2's 14-joint action order."""

    model = env.mojo.physics.model
    qpos_addresses: list[int] = []
    qvel_addresses: list[int] = []
    for actuator in env.robot.limb_actuators:
        if actuator.joint is None:
            raise ValueError(f"limb actuator {actuator.full_identifier} has no joint")
        joint_id = element_id(env.mojo.physics, actuator.joint)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        qvel_addresses.append(int(model.jnt_dofadr[joint_id]))
    return tuple(qpos_addresses), tuple(qvel_addresses)


def wrist_site_ids(env: Any) -> dict[str, int]:
    return {
        side.name.lower(): element_id(env.mojo.physics, site)
        for side, site in env.robot._wrist_sites.items()
    }


def object_geom_ids(env: Any) -> dict[str, set[int]]:
    physics = env.mojo.physics
    return {
        name: geom_ids(physics, obj)
        for name, obj in get_task_objects(env).items()
    }


def robot_geom_ids(env: Any) -> set[int]:
    if hasattr(env, "_robot_geoms"):
        return {int(value) for value in env._robot_geoms}
    model = env.mojo.physics.model
    ids = set()
    for geom_id in range(model.ngeom):
        name = safe_name(model, geom_id, "geom").lower()
        if "panda" in name or "hand_" in name or "robot" in name:
            ids.add(geom_id)
    return ids


def _named_scene_geoms(env: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    candidates = {
        "table": getattr(env, "table", None),
        "floor": getattr(env, "floor", None),
        "cabinet_1": getattr(env, "cabinet_1", None),
        "cabinet_2": getattr(env, "cabinet_2", None),
    }
    for name, obj in candidates.items():
        if obj is None:
            continue
        for geom_id in geom_ids(env.mojo.physics, obj):
            result[geom_id] = f"scene:{name}"
    return result


def geom_labels(env: Any) -> dict[int, str]:
    """Build stable selectors consumed by AllowedContactPolicy."""

    model = env.mojo.physics.model
    labels = {
        geom_id: f"scene:{safe_name(model, geom_id, 'geom')}"
        for geom_id in range(model.ngeom)
    }
    labels.update(_named_scene_geoms(env))
    for object_name, ids in object_geom_ids(env).items():
        for geom_id in ids:
            labels[geom_id] = f"object:{object_name}"
    for geom_id in robot_geom_ids(env):
        name = safe_name(model, geom_id, "geom").lower()
        body_name = safe_name(
            model, int(model.geom_bodyid[geom_id]), "body"
        ).lower()
        if "nohand_left/" in name or "hand_left/" in name:
            side = "left"
        elif "nohand_right/" in name or "hand_right/" in name:
            side = "right"
        else:
            side = "body"
        part = (
            "finger"
            if "finger" in name or "pad" in name or "finger" in body_name
            else "link"
        )
        labels[geom_id] = f"robot:{side}:{part}"
    return labels


def object_freejoint_address(env: Any, object_name: str) -> int:
    """Find the seven-value free-joint qpos address for a task object."""

    obj = get_task_objects(env)[object_name]
    body_id = element_id(env.mojo.physics, obj.body)
    model = env.mojo.physics.model
    body_id_cursor = body_id
    while body_id_cursor > 0:
        start = int(model.body_jntadr[body_id_cursor])
        count = int(model.body_jntnum[body_id_cursor])
        for joint_id in range(start, start + count):
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return int(model.jnt_qposadr[joint_id])
        body_id_cursor = int(model.body_parentid[body_id_cursor])
    raise ValueError(f"object {object_name!r} has no free joint")


def contact_pairs(data: Any) -> Iterable[tuple[int, int, float]]:
    for contact in data.contact:
        yield int(contact.geom1), int(contact.geom2), float(contact.dist)


def body_velocity(env: Any, body: Any) -> tuple[np.ndarray, np.ndarray]:
    bound = env.mojo.physics.bind(getattr(body, "mjcf", body))
    cvel = np.asarray(bound.cvel, dtype=float).reshape(-1)
    if cvel.size != 6:
        return np.zeros(3), np.zeros(3)
    return cvel[3:].copy(), cvel[:3].copy()
