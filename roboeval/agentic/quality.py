"""Postcondition checks for physically credible RoboEval task completion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import hypot
from typing import Any


QUALITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "lift_pot": {
        "required_settle_steps": 80,
        "min_lift_m": 0.10,
        "max_pose_error_rad": 0.349066,
        "max_object_drift_m": 0.01,
        "max_rms_cartesian_jerk": 5.0,
        "max_env_collisions": 0,
    },
    "cube_handover": {
        "required_settle_steps": 80,
        "min_lift_m": 0.05,
        "max_object_drift_m": 0.01,
        "max_rms_cartesian_jerk": 5.0,
        "max_env_collisions": 0,
    },
    "stack_two_blocks": {
        "required_settle_steps": 120,
        "max_xy_error_m": 0.03,
        "min_z_separation_m": 0.045,
        "max_z_separation_m": 0.08,
        "max_object_drift_m": 0.015,
        "max_rms_cartesian_jerk": 30.0,
        "max_env_collisions": 2,
    },
}


@dataclass
class QualityAssessment:
    """Machine-readable quality result separate from RoboEval's raw success."""

    passed: bool
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_task_quality(
    task_key: str,
    state: dict[str, Any],
    *,
    settle_diagnostics: dict[str, Any] | None = None,
) -> QualityAssessment:
    """Require raw success plus stable, task-specific physical postconditions."""

    thresholds = dict(QUALITY_THRESHOLDS[task_key])
    metrics = state.get("metrics", {})
    grippers = state.get("robot", {}).get("grippers", {})
    objects = state.get("objects", {})
    settle = settle_diagnostics or {}
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, actual: Any, expected: str) -> None:
        checks[name] = {"passed": bool(passed), "actual": actual, "expected": expected}

    raw_success = float(metrics.get("task_success", metrics.get("success", 0.0)) or 0.0)
    add("raw_task_success", raw_success >= 1.0, raw_success, ">= 1.0")

    required_steps = int(thresholds["required_settle_steps"])
    settle_steps = int(settle.get("steps", 0) or 0)
    add("physical_settle_completed", settle_steps >= required_steps, settle_steps, f">= {required_steps}")
    settle_success_min = float(settle.get("task_success_min", 0.0) or 0.0)
    add("success_stable_during_settle", settle_success_min >= 1.0, settle_success_min, ">= 1.0")
    drift_values = [float(value) for value in settle.get("max_object_drift", {}).values()]
    max_drift = max(drift_values, default=float("inf"))
    max_allowed_drift = thresholds["max_object_drift_m"]
    add("object_drift", max_drift <= max_allowed_drift, max_drift, f"<= {max_allowed_drift}")

    collisions = int(metrics.get("env_collision_count", 0) or 0)
    max_collisions = int(thresholds["max_env_collisions"])
    add("environment_collisions", collisions <= max_collisions, collisions, f"<= {max_collisions}")
    jerk = float(metrics.get("overall_rms_cartesian_jerk", float("inf")) or float("inf"))
    max_jerk = thresholds["max_rms_cartesian_jerk"]
    add("cartesian_rms_jerk", jerk <= max_jerk, jerk, f"<= {max_jerk}")

    if task_key == "lift_pot":
        holding = _holding(grippers, "kitchenpot")
        add("both_grippers_holding", holding == {"left": True, "right": True}, holding, "left=true, right=true")
        lift = float(metrics.get("target_distance", {}).get("lift distance", 0.0) or 0.0)
        add("lift_height", lift >= thresholds["min_lift_m"], lift, f">= {thresholds['min_lift_m']}")
        pose_error = float(metrics.get("object_pose_error", float("inf")) or float("inf"))
        add("pot_pose_error", pose_error <= thresholds["max_pose_error_rad"], pose_error, f"<= {thresholds['max_pose_error_rad']}")
    elif task_key == "cube_handover":
        holding = _holding(grippers, "cube")
        add("receiver_only_holding", holding == {"left": True, "right": False}, holding, "left=true, right=false")
        lift = float(metrics.get("target_distance", {}).get("lift distance", 0.0) or 0.0)
        add("lift_height", lift >= thresholds["min_lift_m"], lift, f">= {thresholds['min_lift_m']}")
    elif task_key == "stack_two_blocks":
        holding_0 = _holding(grippers, "block_0")
        holding_1 = _holding(grippers, "block_1")
        no_holding = not any((*holding_0.values(), *holding_1.values()))
        add("both_grippers_released", no_holding, {"block_0": holding_0, "block_1": holding_1}, "all false")
        block_0 = objects.get("block_0", {}).get("position")
        block_1 = objects.get("block_1", {}).get("position")
        if block_0 is None or block_1 is None:
            xy_error = float("inf")
            z_separation = float("-inf")
        else:
            upper, lower = sorted((block_0, block_1), key=lambda position: position[2], reverse=True)
            xy_error = hypot(float(upper[0]) - float(lower[0]), float(upper[1]) - float(lower[1]))
            z_separation = float(upper[2]) - float(lower[2])
        add("stack_xy_centering", xy_error <= thresholds["max_xy_error_m"], xy_error, f"<= {thresholds['max_xy_error_m']}")
        z_ok = thresholds["min_z_separation_m"] <= z_separation <= thresholds["max_z_separation_m"]
        add("stack_z_separation", z_ok, z_separation, f"{thresholds['min_z_separation_m']}..{thresholds['max_z_separation_m']}")

    return QualityAssessment(
        passed=all(check["passed"] for check in checks.values()),
        checks=checks,
        thresholds=thresholds,
    )


def _holding(grippers: dict[str, Any], object_name: str) -> dict[str, bool]:
    return {
        side: bool(grippers.get(side, {}).get("holding", {}).get(object_name, False))
        for side in ("left", "right")
    }
