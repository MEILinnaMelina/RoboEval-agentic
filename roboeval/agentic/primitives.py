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
    diagnostics: dict[str, Any] = field(default_factory=dict)
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
        frame_callback: Any | None = None,
        frame_every: int = 15,
    ) -> None:
        if not getattr(env.action_mode, "ee", False):
            raise ValueError("PrimitiveController requires JointPositionActionMode(..., ee=True).")
        self.env = env
        self.render = render
        self.sleep_s = sleep_s
        self.frame_callback = frame_callback
        self.frame_every = max(1, int(frame_every))
        self._control_step_index = 0
        self._last_info: dict[str, Any] = env.get_info()
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._gripper_commands = self._commands_from_gripper_qpos()

    def current_ee_pose(self) -> np.ndarray:
        """Return controlled EE poses as [left xyz+rpy, right xyz+rpy]."""

        return self.env.robot.forward_kinematics(self._arm_qpos())

    # Below this horizontal (xy) travel distance, keep the arm's current yaw
    # rather than recomputing one - avoids perturbing the many small/vertical
    # approaches (e.g. top-down object grasps) that already converge fine.
    _MIN_REORIENT_XY_DISPLACEMENT = 0.03

    def face_target_pose(self, side: HandSide | str, target_position: np.ndarray) -> np.ndarray:
        """Build a 6-DoF pose at target_position for a large horizontal reach.

        Keeps the current roll/pitch, but rotates yaw to face the horizontal
        direction of travel once xy displacement is non-trivial. Restored in
        P21 - validated (P14-followup) and re-confirmed necessary (P20's
        real-API run: without any reorientation, lift_pot's grasp never
        once succeeded) specifically for named-affordance targets (P18
        gates it to only that case) - see
        docs/phase8_success_rate_debug_log.md P14/P18/P20/P21.
        """
        side = self._parse_side(side)
        current_pose = self.current_ee_pose()[self._pose_slice(side)].copy()
        target_position = np.asarray(target_position, dtype=np.float32)
        pose = current_pose.copy()
        pose[:3] = target_position
        dx = float(target_position[0] - current_pose[0])
        dy = float(target_position[1] - current_pose[1])
        if (dx * dx + dy * dy) ** 0.5 >= self._MIN_REORIENT_XY_DISPLACEMENT:
            # The +-pi/2 sign was empirically validated per side (clean,
            # fresh-env tests, not reused-env warm-started ones - see
            # docs/phase8_success_rate_debug_log.md P14) and is consistent
            # with a mirrored bimanual rig: left and right gripper frames
            # are mirror images, so the same heading needs opposite bias.
            bias = np.pi / 2.0 if side == HandSide.LEFT else -np.pi / 2.0
            yaw = float(np.arctan2(dy, dx) + bias)
            pose[5] = (yaw + np.pi) % (2 * np.pi) - np.pi
        return pose

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
        reorient: bool = False,
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
        target_position = obj_pos + ee_offset
        if reorient:
            # Only for a named affordance target (P18) - the one case this
            # was ever actually validated on. Generic object+ee_offset
            # grasps keep the arm's current orientation.
            target_pose = self.face_target_pose(side, target_position)
        else:
            target_pose = self.current_ee_pose()[self._pose_slice(side)].copy()
            target_pose[:3] = target_position
        return self._move_single(
            side,
            target_pose,
            steps,
            pos_tolerance,
            f"align_{side.name.lower()}_to_{object_name}",
            # P22: protect the target object from being shoved/displaced by
            # an approach that isn't actually gripping it yet - see
            # docs/phase8_success_rate_debug_log.md P22.
            protect_object=(object_name, objects[object_name]),
        )

    def grasp_object(
        self,
        side: HandSide | str,
        object_name: str,
        *,
        ee_offset: np.ndarray | None = None,
        steps: int = 90,
        preopen: bool = True,
        close_after: bool = True,
        reorient: bool = False,
    ) -> PrimitiveResult:
        side = self._parse_side(side)
        open_result = self.open_gripper(side, steps=15) if preopen else None
        align_result = self.align_to_object(
            side,
            object_name,
            ee_offset=ee_offset,
            steps=steps,
            pos_tolerance=0.05,
            reorient=reorient,
        )
        close_result = self.close_gripper(side, steps=35) if close_after else None

        objects = get_task_objects(self.env)
        holding = object_name in objects and self.env.robot.is_gripper_holding_object(objects[object_name], side)
        total_steps = align_result.steps
        if open_result is not None:
            total_steps += open_result.steps
        if close_result is not None:
            total_steps += close_result.steps
        result = self._result(
            f"grasp_{side.name.lower()}_{object_name}",
            bool(holding),
            total_steps,
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
        self._gripper_commands[self._side_index(side)] = OPEN_COMMAND
        return self._hold_pose(
            steps,
            f"release_object_{side.name.lower()}",
            "gripper opened",
            stop_on_done=False,
        )

    def settle(self, *, steps: int = 100) -> PrimitiveResult:
        """Hold the robot still and validate object motion under real physics."""

        objects = get_task_objects(self.env)
        start_positions = {name: get_object_position(obj).copy() for name, obj in objects.items()}
        max_drift = {name: 0.0 for name in objects}
        task_success_samples: list[float] = []
        pose = self.current_ee_pose()
        steps_done = 0
        for _ in range(max(1, steps)):
            self._step_with_pose(pose)
            steps_done += 1
            task_success_samples.append(float(self._last_info.get("task_success", 0.0)))
            for name, obj in objects.items():
                drift = float(np.linalg.norm(get_object_position(obj) - start_positions[name]))
                max_drift[name] = max(max_drift[name], drift)

        diagnostics = {
            "steps": steps_done,
            "task_success_min": min(task_success_samples, default=0.0),
            "task_success_final": task_success_samples[-1] if task_success_samples else 0.0,
            "max_object_drift": {name: round(value, 6) for name, value in max_drift.items()},
            "start_object_positions": {name: value.tolist() for name, value in start_positions.items()},
            "final_object_positions": {
                name: get_object_position(obj).tolist() for name, obj in objects.items()
            },
        }
        stable_success = diagnostics["task_success_min"] >= 1.0
        return self._result(
            "settle",
            stable_success,
            steps_done,
            "task remained successful during physical settle"
            if stable_success
            else "task success was not maintained during physical settle",
            "Replan the grasp or placement before settling again." if not stable_success else "",
            diagnostics=diagnostics,
        )

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
        *,
        protect_object: tuple[str, Any] | None = None,
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
            protect_object=protect_object,
        )

    # Abort a primitive early if it causes this many *new* self-collision
    # events during its own execution - a general safety net independent of
    # why the collisions are happening (bad orientation, bad LLM-chosen
    # offset, etc.), so a single primitive call can't grind through its full
    # step budget while repeatedly hitting itself. See
    # docs/phase8_success_rate_debug_log.md P18/P19.
    _MAX_NEW_SELF_COLLISIONS_PER_PRIMITIVE = 3

    # P22: if the object we're approaching (but not yet gripping) moves more
    # than this during the approach, we're shoving it, not missing it -
    # abort rather than let a longer step budget push it further/off the
    # table. Calibrated against a real successful grasp attempt, which
    # legitimately jostles the object ~0.03m while closing in before making
    # stable contact (measured directly, not guessed) - set well above that
    # (safety margin) but far below the >=0.2m displacement seen in the
    # real trials where the object was actually knocked away. See
    # docs/phase8_success_rate_debug_log.md P22.
    _MAX_OBJECT_BUMP_DISPLACEMENT = 0.08

    def _collision_breaker_tripped(self, start_self_collision: int) -> bool:
        current = int(self._last_info.get("self_collision_count", 0))
        return (current - start_self_collision) >= self._MAX_NEW_SELF_COLLISIONS_PER_PRIMITIVE

    def _execute_pose_ramp(
        self,
        start: np.ndarray,
        target: np.ndarray,
        *,
        steps: int,
        pos_tolerance: float,
        name: str,
        sides: list[HandSide],
        protect_object: tuple[str, Any] | None = None,
    ) -> PrimitiveResult:
        steps_done = 0
        start_self_collision = int(self._last_info.get("self_collision_count", 0))
        aborted_collision = False
        aborted_object_bump = False
        protect_start_pos = None
        if protect_object is not None:
            _, protect_obj = protect_object
            protect_start_pos = get_object_position(protect_obj).copy()
        for alpha in np.linspace(0.0, 1.0, max(1, steps)):
            pose = start + (target - start) * alpha
            self._step_with_pose(pose)
            steps_done += 1
            if self._last_terminated or self._last_truncated:
                break
            if self._collision_breaker_tripped(start_self_collision):
                aborted_collision = True
                break
            if protect_start_pos is not None:
                _, protect_obj = protect_object
                bump = float(np.linalg.norm(get_object_position(protect_obj) - protect_start_pos))
                if bump >= self._MAX_OBJECT_BUMP_DISPLACEMENT:
                    aborted_object_bump = True
                    break

        current = self.current_ee_pose()
        distances = {}
        ok = True
        for side in sides:
            pose_slice = self._pose_slice(side)
            dist = float(np.linalg.norm(current[pose_slice.start : pose_slice.start + 3] - target[pose_slice.start : pose_slice.start + 3]))
            distances[f"{side.name.lower()}_ee_to_target"] = round(dist, 5)
            ok = ok and dist <= pos_tolerance

        if aborted_collision:
            return self._result(
                name,
                False,
                steps_done,
                "aborted early: self-collision increased rapidly during this move",
                "Choose a different offset/target, or move one arm out of the way first before retrying.",
                distances=distances,
            )
        if aborted_object_bump:
            object_name = protect_object[0] if protect_object is not None else "object"
            return self._result(
                name,
                False,
                steps_done,
                f"aborted early: {object_name} was displaced during approach (likely bumped, not gripped)",
                "Back off and retry with a different ee_offset or approach angle, rather than continuing toward the same point.",
                distances=distances,
            )
        return self._result(
            name,
            ok and not self._last_truncated,
            steps_done,
            "target reached" if ok else "target not reached within tolerance",
            "Increase steps or choose a closer waypoint." if not ok else "",
            distances=distances,
        )

    def _hold_pose(
        self,
        steps: int,
        name: str,
        message: str,
        *,
        stop_on_done: bool = True,
    ) -> PrimitiveResult:
        pose = self.current_ee_pose()
        steps_done = 0
        for _ in range(max(1, steps)):
            self._step_with_pose(pose)
            steps_done += 1
            if stop_on_done and (self._last_terminated or self._last_truncated):
                break
        return self._result(name, not self._last_truncated, steps_done, message, "")

    def _step_with_pose(self, pose: np.ndarray) -> None:
        action = np.concatenate([pose, self._gripper_commands]).astype(np.float32)
        _, reward, terminated, truncated, info = self.env.step(action, fast=False)
        self._last_reward = float(reward)
        self._last_terminated = bool(terminated)
        self._last_truncated = bool(truncated)
        self._last_info = info
        self._control_step_index += 1
        if self.frame_callback is not None and self._control_step_index % self.frame_every == 0:
            self.frame_callback(self.env, self._control_step_index)
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
        diagnostics: dict[str, Any] | None = None,
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
            diagnostics=diagnostics or {},
            next_suggestion=next_suggestion,
        )

    def _failure(self, name: str, message: str, next_suggestion: str) -> PrimitiveResult:
        return self._result(name, False, 0, message, next_suggestion)
