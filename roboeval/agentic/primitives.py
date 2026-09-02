"""Low-level primitives for non-policy RoboEval control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from roboeval.const import HandSide
from roboeval.envs.props.prop import Prop
from roboeval.agentic.state import get_object_position, get_task_objects


OPEN_COMMAND = 0.0
CLOSE_COMMAND = 1.0


@dataclass
class PrimitiveResult:
    """Result returned by every primitive."""

    name: str
    success: bool
    steps: int
    message: str
    task_success: float = 0.0
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    distances: dict[str, float] = field(default_factory=dict)
    collisions: dict[str, int] = field(default_factory=dict)
    next_suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrimitiveController:
    """Execute simple end-effector and gripper primitives in a RoboEval env."""

    def __init__(
        self,
        env: Any,
        *,
        render: bool = False,
        sleep_s: float = 0.0,
    ) -> None:
        if not getattr(env.action_mode, "ee", False):
            raise ValueError("PrimitiveController requires JointPositionActionMode(..., ee=True).")
        self.env = env
        self.render = render
        self.sleep_s = sleep_s
        self._last_info: dict[str, Any] = env.get_info()
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._gripper_commands = self._commands_from_gripper_qpos()

    def current_ee_pose(self) -> np.ndarray:
        """Return controlled EE poses as [left xyz+rpy, right xyz+rpy]."""

        return self.env.robot.forward_kinematics(self._arm_qpos())

    def move_left_ee(
        self,
        target_pose: np.ndarray,
        *,
        steps: int = 80,
        pos_tolerance: float = 0.03,
    ) -> PrimitiveResult:
        return self._move_single(HandSide.LEFT, target_pose, steps, pos_tolerance, "move_left_ee")

    def move_right_ee(
        self,
        target_pose: np.ndarray,
        *,
        steps: int = 80,
        pos_tolerance: float = 0.03,
    ) -> PrimitiveResult:
        return self._move_single(HandSide.RIGHT, target_pose, steps, pos_tolerance, "move_right_ee")

    def move_both_ee(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        *,
        steps: int = 80,
        pos_tolerance: float = 0.03,
    ) -> PrimitiveResult:
        start = self.current_ee_pose()
        target = start.copy()
        target[0:6] = self._normalize_pose(left_pose, start[0:6])
        target[6:12] = self._normalize_pose(right_pose, start[6:12])
        return self._execute_pose_ramp(
            start,
            target,
            steps=steps,
            pos_tolerance=pos_tolerance,
            name="move_both_ee",
            sides=[HandSide.LEFT, HandSide.RIGHT],
        )

    def open_gripper(self, side: HandSide | str, *, steps: int = 25) -> PrimitiveResult:
        side = self._parse_side(side)
        self._gripper_commands[self._side_index(side)] = OPEN_COMMAND
        return self._hold_pose(steps, f"open_gripper_{side.name.lower()}", "gripper opened")

    def close_gripper(self, side: HandSide | str, *, steps: int = 25) -> PrimitiveResult:
        side = self._parse_side(side)
        self._gripper_commands[self._side_index(side)] = CLOSE_COMMAND
        return self._hold_pose(steps, f"close_gripper_{side.name.lower()}", "gripper closed")

    def align_to_object(
        self,
        side: HandSide | str,
        object_name: str,
        *,
        ee_offset: np.ndarray | None = None,
        steps: int = 90,
        pos_tolerance: float = 0.04,
    ) -> PrimitiveResult:
        side = self._parse_side(side)
        objects = get_task_objects(self.env)
        if object_name not in objects:
            return self._failure(
                "align_to_object",
                f"object {object_name!r} is not available",
                "Use task_probe.py to inspect available object names.",
            )
        obj_pos = get_object_position(objects[object_name]).copy()
        ee_offset = np.asarray(ee_offset if ee_offset is not None else [0.0, 0.0, 0.08])
        target_pose = self.current_ee_pose()[self._pose_slice(side)].copy()
        target_pose[:3] = obj_pos + ee_offset
        return self._move_single(
            side,
            target_pose,
            steps,
            pos_tolerance,
            f"align_{side.name.lower()}_to_{object_name}",
        )

    def grasp_object(
        self,
        side: HandSide | str,
        object_name: str,
        *,
        ee_offset: np.ndarray | None = None,
        steps: int = 90,
    ) -> PrimitiveResult:
        side = self._parse_side(side)
        open_result = self.open_gripper(side, steps=15)
        align_result = self.align_to_object(
            side,
            object_name,
            ee_offset=ee_offset,
            steps=steps,
            pos_tolerance=0.05,
        )
        close_result = self.close_gripper(side, steps=35)

        objects = get_task_objects(self.env)
        holding = object_name in objects and self.env.robot.is_gripper_holding_object(objects[object_name], side)
        result = self._result(
            f"grasp_{side.name.lower()}_{object_name}",
            bool(holding),
            open_result.steps + align_result.steps + close_result.steps,
            "object is held" if holding else "object was approached but not detected as held",
            "Try a different pinch_offset or approach side if holding is false.",
        )
        return result

    def lift_object(
        self,
        side: HandSide | str | None = None,
        *,
        height: float = 0.1,
        steps: int = 70,
        pos_tolerance: float = 0.04,
    ) -> PrimitiveResult:
        pose = self.current_ee_pose()
        target = pose.copy()
        sides: list[HandSide]
        if side is None or str(side).lower() == "both":
            sides = [HandSide.LEFT, HandSide.RIGHT]
        else:
            sides = [self._parse_side(side)]

        for hand in sides:
            pose_slice = self._pose_slice(hand)
            target[pose_slice.start + 2] += height

        return self._execute_pose_ramp(
            pose,
            target,
            steps=steps,
            pos_tolerance=pos_tolerance,
            name="lift_object",
            sides=sides,
        )

    def release_object(self, side: HandSide | str, *, steps: int = 25) -> PrimitiveResult:
        side = self._parse_side(side)
        return self.open_gripper(side, steps=steps)

    def rotate_tool_or_object(
        self,
        side: HandSide | str,
        *,
        yaw_delta: float = 0.4,
        steps: int = 60,
    ) -> PrimitiveResult:
        side = self._parse_side(side)
        pose = self.current_ee_pose()
        target = pose.copy()
        pose_slice = self._pose_slice(side)
        target[pose_slice.start + 5] += yaw_delta
        target[pose_slice.start + 3 : pose_slice.start + 6] = self._wrap_angles(
            target[pose_slice.start + 3 : pose_slice.start + 6]
        )
        return self._execute_pose_ramp(
            pose,
            target,
            steps=steps,
            pos_tolerance=0.05,
            name=f"rotate_{side.name.lower()}_tool",
            sides=[side],
        )

    def _move_single(
        self,
        side: HandSide,
        target_pose: np.ndarray,
        steps: int,
        pos_tolerance: float,
        name: str,
    ) -> PrimitiveResult:
        start = self.current_ee_pose()
        target = start.copy()
        pose_slice = self._pose_slice(side)
        target[pose_slice] = self._normalize_pose(target_pose, start[pose_slice])
        return self._execute_pose_ramp(
            start,
            target,
            steps=steps,
            pos_tolerance=pos_tolerance,
            name=name,
            sides=[side],
        )

    def _execute_pose_ramp(
        self,
        start: np.ndarray,
        target: np.ndarray,
        *,
        steps: int,
        pos_tolerance: float,
        name: str,
        sides: list[HandSide],
    ) -> PrimitiveResult:
        steps_done = 0
        for alpha in np.linspace(0.0, 1.0, max(1, steps)):
            pose = start + (target - start) * alpha
            self._step_with_pose(pose)
            steps_done += 1
            if self._last_terminated or self._last_truncated:
                break

        current = self.current_ee_pose()
        distances = {}
        ok = True
        for side in sides:
            pose_slice = self._pose_slice(side)
            dist = float(np.linalg.norm(current[pose_slice.start : pose_slice.start + 3] - target[pose_slice.start : pose_slice.start + 3]))
            distances[f"{side.name.lower()}_ee_to_target"] = round(dist, 5)
            ok = ok and dist <= pos_tolerance

        return self._result(
            name,
            ok and not self._last_truncated,
            steps_done,
            "target reached" if ok else "target not reached within tolerance",
            "Increase steps or choose a closer waypoint." if not ok else "",
            distances=distances,
        )

    def _hold_pose(self, steps: int, name: str, message: str) -> PrimitiveResult:
        pose = self.current_ee_pose()
        steps_done = 0
        for _ in range(max(1, steps)):
            self._step_with_pose(pose)
            steps_done += 1
            if self._last_terminated or self._last_truncated:
                break
        return self._result(name, not self._last_truncated, steps_done, message, "")

    def _step_with_pose(self, pose: np.ndarray) -> None:
        action = np.concatenate([pose, self._gripper_commands]).astype(np.float32)
        _, reward, terminated, truncated, info = self.env.step(action, fast=False)
        self._last_reward = float(reward)
        self._last_terminated = bool(terminated)
        self._last_truncated = bool(truncated)
        self._last_info = info
        if self.render and self.env.render_mode:
            self.env.render()
        if self.sleep_s:
            import time

            time.sleep(self.sleep_s)

    def _commands_from_gripper_qpos(self) -> np.ndarray:
        commands = []
        for _, gripper in self.env.robot.grippers.items():
            commands.append(OPEN_COMMAND if gripper.qpos > 0.5 else CLOSE_COMMAND)
        return np.asarray(commands, dtype=np.float32)

    def _arm_qpos(self) -> np.ndarray:
        return self.env.robot.qpos_actuated[:-len(self.env.robot.grippers)]

    def _pose_slice(self, side: HandSide) -> slice:
        idx = self._side_index(side)
        return slice(idx * 6, idx * 6 + 6)

    def _side_index(self, side: HandSide) -> int:
        sides = list(self.env.robot.grippers.keys())
        return sides.index(side)

    def _parse_side(self, side: HandSide | str) -> HandSide:
        if isinstance(side, HandSide):
            return side
        normalized = side.lower()
        if normalized in {"left", "l"}:
            return HandSide.LEFT
        if normalized in {"right", "r"}:
            return HandSide.RIGHT
        raise ValueError(f"Unknown hand side: {side}")

    def _normalize_pose(self, target_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
        target_pose = np.asarray(target_pose, dtype=np.float32)
        if target_pose.shape == (3,):
            pose = current_pose.copy()
            pose[:3] = target_pose
            return pose
        if target_pose.shape == (6,):
            pose = target_pose.copy()
            pose[3:6] = self._wrap_angles(pose[3:6])
            return pose
        raise ValueError("target_pose must be shape (3,) or (6,)")

    def _wrap_angles(self, angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (2 * np.pi) - np.pi

    def _result(
        self,
        name: str,
        success: bool,
        steps: int,
        message: str,
        next_suggestion: str,
        *,
        distances: dict[str, float] | None = None,
    ) -> PrimitiveResult:
        return PrimitiveResult(
            name=name,
            success=success,
            steps=steps,
            message=message,
            task_success=float(self._last_info.get("task_success", 0.0)),
            reward=self._last_reward,
            terminated=self._last_terminated,
            truncated=self._last_truncated,
            distances=distances or {},
            collisions={
                "env_collision_count": int(self._last_info.get("env_collision_count", 0)),
                "self_collision_count": int(self._last_info.get("self_collision_count", 0)),
            },
            next_suggestion=next_suggestion,
        )

    def _failure(self, name: str, message: str, next_suggestion: str) -> PrimitiveResult:
        return self._result(name, False, 0, message, next_suggestion)
