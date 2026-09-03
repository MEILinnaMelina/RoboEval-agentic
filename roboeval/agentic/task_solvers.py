"""Task-level primitive solvers for the initial RoboEval tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from roboeval.agentic.llm_agent import ActionPlan, PrimitiveExecutor
from roboeval.agentic.primitives import PrimitiveController, PrimitiveResult
from roboeval.agentic.state import collect_env_state, get_task_objects
from roboeval.const import HandSide


@dataclass
class SolverStep:
    """One task-level solver step."""

    index: int
    label: str
    action: dict[str, Any]
    result: dict[str, Any]
    holding: dict[str, dict[str, bool]]


@dataclass
class SolverReport:
    """Final report for one solver rollout."""

    task_key: str
    success: bool
    final_task_success: float
    failure_reasons: list[str]
    steps: list[SolverStep] = field(default_factory=list)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    final_objects: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskSolver:
    """Run hand-written primitive plans through the same executor used by LLM agents."""

    def __init__(self, task_key: str, env: Any, controller: PrimitiveController) -> None:
        self.task_key = task_key
        self.env = env
        self.controller = controller
        self.executor = PrimitiveExecutor(env, controller, task_key=task_key)
        self.steps: list[SolverStep] = []

    def solve(self) -> SolverReport:
        solvers: dict[str, Callable[[], None]] = {
            "lift_pot": self._solve_lift_pot,
            "cube_handover": self._solve_cube_handover,
            "stack_two_blocks": self._solve_stack_two_blocks,
        }
        if self.task_key not in solvers:
            raise ValueError(f"No solver registered for task {self.task_key!r}.")

        solvers[self.task_key]()
        state = collect_env_state(self.env)
        final_success = float(state["metrics"].get("task_success", state["metrics"].get("success", 0.0)))
        return SolverReport(
            task_key=self.task_key,
            success=final_success >= 1.0,
            final_task_success=final_success,
            failure_reasons=self._failure_reasons(state),
            steps=self.steps,
            final_metrics=state["metrics"],
            final_objects=state["objects"],
        )

    def _solve_lift_pot(self) -> None:
        self._execute(
            "open left gripper before approaching the pot handle",
            ActionPlan(
                thought="Open the left gripper so it can wrap around the pot handle.",
                primitive="open_gripper",
                args={"side": "left", "steps": 40},
            ),
        )
        if self._is_done():
            return

        self._execute(
            "open right gripper before approaching the pot handle",
            ActionPlan(
                thought="Open the right gripper so it can wrap around the pot handle.",
                primitive="open_gripper",
                args={"side": "right", "steps": 40},
            ),
        )
        if self._is_done():
            return

        objects = get_task_objects(self.env)
        pot_pos = objects["kitchenpot"].body.get_position().copy()
        current = self.controller.current_ee_pose()

        left = current[0:6].copy()
        right = current[6:12].copy()
        left[:3] = pot_pos + np.array([-0.18, 0.16, 0.13])
        right[:3] = pot_pos + np.array([-0.18, -0.16, 0.13])

        self._execute(
            "locate left/right pot handles and approach with both grippers",
            ActionPlan(
                thought="Move both open grippers near the left and right pot handles.",
                primitive="move_both_ee",
                args={"left_target": left.tolist(), "right_target": right.tolist(), "steps": 220, "pos_tolerance": 0.12},
            ),
        )
        if self._is_done():
            return

        self._execute(
            "close left gripper on the left pot handle",
            ActionPlan(
                thought="Close the left gripper around the left pot handle.",
                primitive="close_gripper",
                args={"side": "left", "steps": 80},
            ),
        )
        if self._is_done():
            return

        self._execute(
            "close right gripper on the right pot handle",
            ActionPlan(
                thought="Close the right gripper around the right pot handle.",
                primitive="close_gripper",
                args={"side": "right", "steps": 80},
            ),
        )
        if self._is_done():
            return

        self._execute(
            "synchronously lift the pot while preserving gripper spacing",
            ActionPlan(
                thought="Lift both grippers together to raise the pot above the success threshold.",
                primitive="lift_object",
                args={"side": "both", "height": 0.18, "steps": 320, "pos_tolerance": 0.18},
            ),
        )
    def _solve_cube_handover(self) -> None:
        self._execute(
            "initial right gripper grasps the rod",
            ActionPlan(
                thought="Use the closed right gripper to clamp the rod without pre-opening it.",
                primitive="grasp_object",
                args={
                    "side": "right",
                    "object": "cube",
                    "ee_offset": [-0.12, 0.0, 0.0],
                    "steps": 220,
                    "preopen": False,
                    "close_after": False,
                },
            ),
        )
        if self._is_done():
            return

        self._execute(
            "move the held rod into the shared handover area",
            ActionPlan(
                thought="Raise the right gripper so the rod can be received safely.",
                primitive="lift_object",
                args={"side": "right", "height": 0.08, "steps": 120, "pos_tolerance": 0.08},
            ),
        )
        if self._is_done():
            return

        self._execute(
            "receiver left gripper approaches and grasps the rod",
            ActionPlan(
                thought="Use the left gripper to contact and receive the rod from the right gripper.",
                primitive="grasp_object",
                args={
                    "side": "left",
                    "object": "cube",
                    "ee_offset": [-0.20, -0.12, 0.02],
                    "steps": 240,
                    "preopen": False,
                    "close_after": False,
                },
            ),
        )
        if self._is_done():
            return

        self._execute(
            "release the original right gripper",
            ActionPlan(
                thought="Open the original right gripper after the left gripper has received the rod.",
                primitive="release_object",
                args={"side": "right", "steps": 80},
            ),
        )

    def _solve_stack_two_blocks(self) -> None:
        current = self.controller.current_ee_pose()
        left = current[0:6].copy()
        left[:3] = np.array([0.45, 0.58, 1.22])
        self._execute(
            "move unused left arm to a safe parking pose",
            ActionPlan(
                thought="Move the left arm away so the right arm can carry block_0 without self-collision.",
                primitive="move_left_ee",
                args={"target": left.tolist(), "steps": 160, "pos_tolerance": 0.12},
            ),
        )
        if self._is_done():
            return

        self._execute(
            "grasp block_0 with the right gripper",
            ActionPlan(
                thought="Open, align, and close the right gripper on block_0.",
                primitive="grasp_object",
                args={"side": "right", "object": "block_0", "target": "block_0_top", "steps": 160},
            ),
        )
        if self._is_done():
            return

        self._execute(
            "lift block_0 clear of the table",
            ActionPlan(
                thought="Lift block_0 before moving laterally above block_1.",
                primitive="lift_object",
                args={"side": "right", "height": 0.10, "steps": 120, "pos_tolerance": 0.08},
            ),
        )
        if self._is_done():
            return

        self._move_held_block0_above_block1(clearance=0.045, label="move block_0 above block_1")
        if self._is_done():
            return

        self._move_held_block0_above_block1(clearance=0.041, label="lower block_0 onto block_1")
        if self._is_done():
            return

        self._execute(
            "release after stacking",
            ActionPlan(
                thought="Open the right gripper once the upper block is resting on the lower block.",
                primitive="release_object",
                args={"side": "right", "steps": 80},
            ),
        )

    def _move_held_block0_above_block1(self, *, clearance: float, label: str) -> None:
        objects = get_task_objects(self.env)
        block0 = objects["block_0"].body.get_position().copy()
        block1 = objects["block_1"].body.get_position().copy()
        right_pose = self.controller.current_ee_pose()[6:12].copy()
        ee_to_block = right_pose[:3] - block0
        desired_block = block1.copy()
        desired_block[2] = block1[2] + clearance
        target = right_pose.copy()
        target[:3] = desired_block + ee_to_block
        self._execute(
            label,
            ActionPlan(
                thought=f"Keep the held block pose and move block_0 to block_1 plus {clearance:.3f} m vertical clearance.",
                primitive="move_right_ee",
                args={"target": target.tolist(), "steps": 240 if clearance > 0.042 else 120, "pos_tolerance": 0.10},
            ),
        )

    def _execute(self, label: str, action: ActionPlan) -> PrimitiveResult:
        result = self.executor.execute(action)
        result_dict = result.to_dict()
        result_dict["task_completed"] = result.task_success >= 1.0
        self.steps.append(
            SolverStep(
                index=len(self.steps),
                label=label,
                action=action.to_dict(),
                result=result_dict,
                holding=self._holding_summary(),
            )
        )
        return result

    def _is_done(self) -> bool:
        if not self.steps:
            return False
        result = self.steps[-1].result
        return bool(result.get("terminated")) or float(result.get("task_success", 0.0)) >= 1.0

    def _holding_summary(self) -> dict[str, dict[str, bool]]:
        objects = get_task_objects(self.env)
        summary: dict[str, dict[str, bool]] = {}
        for object_name, obj in objects.items():
            summary[object_name] = {
                "left": bool(self.env.robot.is_gripper_holding_object(obj, HandSide.LEFT)),
                "right": bool(self.env.robot.is_gripper_holding_object(obj, HandSide.RIGHT)),
            }
        return summary

    def _failure_reasons(self, state: dict[str, Any]) -> list[str]:
        metrics = state.get("metrics", {})
        if float(metrics.get("task_success", metrics.get("success", 0.0))) >= 1.0:
            return []

        reasons = []
        stages = metrics.get("task_stage_reached", {})
        reasons.append(f"task_success remained {metrics.get('task_success', metrics.get('success', 0.0))}")
        if stages:
            reasons.append(f"stages reached: {stages}")
        if self.task_key == "lift_pot":
            target_distance = metrics.get("target_distance", {})
            reasons.append(f"lift distance: {target_distance.get('lift distance')}")
            reasons.append("pot handle contact slips before the pot reaches the 0.10 m success height")
        elif self.task_key == "cube_handover":
            reasons.append("handover requires final holding by the receiver and release by the initial gripper")
        elif self.task_key == "stack_two_blocks":
            reasons.append("stacking requires block-on-block contact and both grippers released")
        return reasons
