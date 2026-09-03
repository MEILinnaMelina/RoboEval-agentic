"""LLM-driven primitive planner for RoboEval agentic experiments."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from roboeval.agentic.primitives import PrimitiveController, PrimitiveResult
from roboeval.agentic.state import collect_env_state, get_object_position, get_task_objects
from roboeval.agentic.task_specs import TASK_SPECS, TaskSpec
from roboeval.const import HandSide


PrimitiveArgs = dict[str, Any]


PRIMITIVE_SCHEMAS: dict[str, dict[str, Any]] = {
    "move_left_ee": {
        "args": {"target": "symbolic target or xyz/rpy pose", "steps": "optional int"},
        "description": "Move the left end effector to a symbolic Cartesian target.",
    },
    "move_right_ee": {
        "args": {"target": "symbolic target or xyz/rpy pose", "steps": "optional int"},
        "description": "Move the right end effector to a symbolic Cartesian target.",
    },
    "move_both_ee": {
        "args": {
            "left_target": "symbolic target or xyz/rpy pose",
            "right_target": "symbolic target or xyz/rpy pose",
            "steps": "optional int",
        },
        "description": "Move both end effectors in one synchronized primitive.",
    },
    "open_gripper": {
        "args": {"side": "left or right", "steps": "optional int"},
        "description": "Open one gripper.",
    },
    "close_gripper": {
        "args": {"side": "left or right", "steps": "optional int"},
        "description": "Close one gripper.",
    },
    "align_to_object": {
        "args": {
            "side": "left or right",
            "object": "object name, for example cube or block_0",
            "target": "optional symbolic target alias",
            "ee_offset": "optional xyz offset from the object",
            "steps": "optional int",
        },
        "description": "Move a gripper to an object-relative pre-grasp pose.",
    },
    "grasp_object": {
        "args": {
            "side": "left or right",
            "object": "object name",
            "target": "optional symbolic target alias",
            "ee_offset": "optional xyz offset from the object",
            "steps": "optional int",
            "preopen": "optional bool; false keeps the gripper in its current state before approach",
            "close_after": "optional bool; false skips the close command after approach",
        },
        "description": "Approach with optional open/close steps and check whether the object is held.",
    },
    "lift_object": {
        "args": {"side": "left, right, or both", "height": "meters", "steps": "optional int"},
        "description": "Lift one or both currently controlled grippers vertically.",
    },
    "place_held_object_on_object": {
        "args": {
            "side": "left or right",
            "held_object": "object currently carried by the gripper",
            "support_object": "object to place onto",
            "high_clearance": "pre-place vertical clearance in meters",
            "place_clearance": "final placement vertical clearance in meters",
            "high_steps": "optional int",
            "place_steps": "optional int",
        },
        "description": "Continuously move a held object above another object and lower it for placement.",
    },    "move_held_object_above_object": {
        "args": {
            "side": "left or right",
            "held_object": "object currently carried by the gripper",
            "support_object": "object or placement target to move above",
            "clearance": "vertical clearance in meters",
            "steps": "optional int",
        },
        "description": "Move a gripper so the held object is positioned above another object while preserving grasp offset.",
    },    "release_object": {
        "args": {"side": "left or right", "steps": "optional int"},
        "description": "Open one gripper to release an object.",
    },
    "rotate_tool_or_object": {
        "args": {"side": "left or right", "yaw_delta": "radians", "steps": "optional int"},
        "description": "Rotate one gripper around yaw while keeping its position.",
    },
    "finish": {
        "args": {},
        "description": "Stop when the task success metric is already satisfied.",
    },
}


SYMBOLIC_TARGETS: dict[str, dict[str, Any]] = {
    "left_pot_handle": {"object": "kitchenpot", "offset": [-0.18, 0.16, 0.13]},
    "right_pot_handle": {"object": "kitchenpot", "offset": [-0.18, -0.16, 0.13]},
    "pot_center_above": {"object": "kitchenpot", "offset": [0.0, 0.0, 0.16]},
    "cube_initial_right_grasp": {"object": "cube", "offset": [-0.12, 0.0, 0.0]},
    "cube_receiver_left_grasp": {"object": "cube", "offset": [-0.20, -0.12, 0.02]},
    "cube_top": {"object": "cube", "offset": [0.0, 0.0, 0.1]},
    "cube_grasp": {"object": "cube", "offset": [0.0, 0.0, 0.1]},
    "block_0_top": {"object": "block_0", "offset": [0.0, 0.0, 0.1]},
    "block_1_top": {"object": "block_1", "offset": [0.0, 0.0, 0.1]},
    "block_1_stack_pose": {"object": "block_1", "offset": [0.0, 0.0, 0.14]},
}


TASK_PLAYBOOKS: dict[str, list[dict[str, Any]]] = {
    "lift_pot": [
        {"purpose": "prepare the left gripper", "primitive": "open_gripper", "args": {"side": "left", "steps": 40}},
        {"purpose": "prepare the right gripper", "primitive": "open_gripper", "args": {"side": "right", "steps": 40}},
        {
            "purpose": "locate the two pot handles and approach them together",
            "primitive": "move_both_ee",
            "args": {"left_target": "left_pot_handle", "right_target": "right_pot_handle", "steps": 220, "pos_tolerance": 0.12},
        },
        {"purpose": "close on the left handle", "primitive": "close_gripper", "args": {"side": "left", "steps": 80}},
        {"purpose": "close on the right handle", "primitive": "close_gripper", "args": {"side": "right", "steps": 80}},
        {
            "purpose": "synchronously raise the pot",
            "primitive": "lift_object",
            "args": {"side": "both", "height": 0.18, "steps": 320, "pos_tolerance": 0.18},
        },
        {"purpose": "stop after task_success is 1.0", "primitive": "finish", "args": {}},
    ],
    "cube_handover": [
        {
            "purpose": "initial right gripper clamps the rod/cube object",
            "primitive": "grasp_object",
            "args": {"side": "right", "object": "cube", "target": "cube_initial_right_grasp", "steps": 220, "preopen": False, "close_after": False},
        },
        {
            "purpose": "move the held object into the shared handover zone",
            "primitive": "lift_object",
            "args": {"side": "right", "height": 0.08, "steps": 120, "pos_tolerance": 0.08},
        },
        {
            "purpose": "receiver left gripper approaches and receives the object",
            "primitive": "grasp_object",
            "args": {"side": "left", "object": "cube", "target": "cube_receiver_left_grasp", "steps": 240, "preopen": False, "close_after": False},
        },
        {"purpose": "release original right gripper if task_success is not already reached", "primitive": "release_object", "args": {"side": "right", "steps": 80}},
        {"purpose": "stop after task_success is 1.0", "primitive": "finish", "args": {}},
    ],
    "stack_two_blocks": [
        {
            "purpose": "move unused left arm to a safe parking pose",
            "primitive": "move_left_ee",
            "args": {"target": "left_safe_parking", "steps": 160, "pos_tolerance": 0.12},
        },
        {
            "purpose": "grasp block_0 with the right gripper",
            "primitive": "grasp_object",
            "args": {"side": "right", "object": "block_0", "target": "block_0_top", "steps": 160},
        },
        {
            "purpose": "lift block_0 clear of the table",
            "primitive": "lift_object",
            "args": {"side": "right", "height": 0.10, "steps": 120, "pos_tolerance": 0.08},
        },
        {
            "purpose": "place held block_0 onto block_1 using a continuous high-then-lower motion",
            "primitive": "place_held_object_on_object",
            "args": {"side": "right", "held_object": "block_0", "support_object": "block_1", "high_clearance": 0.045, "place_clearance": 0.041, "high_steps": 400, "place_steps": 240, "pos_tolerance": 0.10},
        },
        {"purpose": "release once the blocks are stacked", "primitive": "release_object", "args": {"side": "right", "steps": 80}},
        {"purpose": "stop after task_success is 1.0", "primitive": "finish", "args": {}},
    ],
}

FORBIDDEN_ARG_FRAGMENTS = {
    "action",
    "actuator",
    "joint",
    "qpos",
    "qvel",
    "torque",
}


@dataclass
class ActionPlan:
    """One structured action selected by an LLM planner."""

    thought: str
    primitive: str
    args: PrimitiveArgs = field(default_factory=dict)

    @classmethod
    def from_llm_response(cls, text: str) -> "ActionPlan":
        data = _parse_first_json_object(text)
        if isinstance(data, list):
            if not data:
                raise ValueError("LLM returned an empty list.")
            data = data[0]
        if not isinstance(data, dict):
            raise ValueError("LLM response must be a JSON object.")

        primitive = str(data.get("primitive", "")).strip()
        if not primitive:
            raise ValueError("LLM response is missing primitive.")

        args = data.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ValueError("LLM args must be a JSON object.")

        return cls(
            thought=str(data.get("thought", "")).strip(),
            primitive=primitive,
            args=args,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentStepRecord:
    """Trace for one plan-observe-execute iteration."""

    index: int
    state_summary: dict[str, Any]
    system_prompt: str
    user_prompt: str
    raw_response: str
    action: dict[str, Any]
    result: dict[str, Any]
    feedback: list[str] = field(default_factory=list)


@dataclass
class AgentRunResult:
    """Summary for one LLM-agent rollout."""

    task_key: str
    provider: str
    model: str | None
    completed: bool
    final_task_success: float
    steps: list[AgentStepRecord]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["steps"] = [asdict(step) for step in self.steps]
        return data


class PlannerClient(Protocol):
    """Minimal interface implemented by concrete LLM providers."""

    provider: str
    model: str | None

    def plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        history: list[AgentStepRecord],
    ) -> str:
        """Return one JSON action plan as text."""


class MockPlanner:
    """Deterministic planner used to test the full loop without an API key."""

    provider = "mock"
    model = None

    def __init__(self, task_key: str) -> None:
        self.task_key = task_key
        self._plans = self._build_plans(task_key)

    def plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        history: list[AgentStepRecord],
    ) -> str:
        idx = min(len(history), len(self._plans) - 1)
        return json.dumps(self._plans[idx])

    def _build_plans(self, task_key: str) -> list[dict[str, Any]]:
        if task_key in TASK_PLAYBOOKS:
            return [
                {
                    "thought": step["purpose"],
                    "primitive": step["primitive"],
                    "args": dict(step.get("args", {})),
                }
                for step in TASK_PLAYBOOKS[task_key]
            ]
        if task_key == "lift_pot":
            return [
                {
                    "thought": "Open both grippers before approaching the pot handles.",
                    "primitive": "open_gripper",
                    "args": {"side": "left"},
                },
                {
                    "thought": "Open the right gripper before the bimanual grasp.",
                    "primitive": "open_gripper",
                    "args": {"side": "right"},
                },
                {
                    "thought": "Move each gripper to a different pot handle.",
                    "primitive": "move_both_ee",
                    "args": {
                        "left_target": "left_pot_handle",
                        "right_target": "right_pot_handle",
                        "steps": 140,
                        "pos_tolerance": 0.07,
                    },
                },
                {
                    "thought": "Close the left gripper on the left handle.",
                    "primitive": "close_gripper",
                    "args": {"side": "left"},
                },
                {
                    "thought": "Close the right gripper on the right handle.",
                    "primitive": "close_gripper",
                    "args": {"side": "right"},
                },
                {
                    "thought": "Lift both grippers together while keeping the pot stable.",
                    "primitive": "lift_object",
                    "args": {"side": "both", "height": 0.06, "steps": 120},
                },
                {"thought": "Stop after checking the task metric.", "primitive": "finish", "args": {}},
            ]
        if task_key == "stack_two_blocks":
            return [
                {
                    "thought": "Open the right gripper before approaching block_0.",
                    "primitive": "open_gripper",
                    "args": {"side": "right"},
                },
                {
                    "thought": "Grasp block_0 with the right gripper.",
                    "primitive": "grasp_object",
                    "args": {"side": "right", "object": "block_0", "target": "block_0_top", "steps": 140},
                },
                {
                    "thought": "Lift block_0 high enough to clear block_1.",
                    "primitive": "lift_object",
                    "args": {"side": "right", "height": 0.09, "steps": 120},
                },
                {
                    "thought": "Move block_0 above block_1 for stacking.",
                    "primitive": "move_right_ee",
                    "args": {"target": "block_1_stack_pose", "steps": 140, "pos_tolerance": 0.08},
                },
                {
                    "thought": "Release block_0 over block_1.",
                    "primitive": "release_object",
                    "args": {"side": "right"},
                },
                {"thought": "Stop after checking the task metric.", "primitive": "finish", "args": {}},
            ]
        return [
            {
                "thought": "Open the left gripper before approaching the cube.",
                "primitive": "open_gripper",
                "args": {"side": "left"},
            },
            {
                "thought": "Grasp the cube with the left gripper.",
                "primitive": "grasp_object",
                "args": {"side": "left", "object": "cube", "target": "cube_grasp", "steps": 140},
            },
            {
                "thought": "Lift the cube for handover.",
                "primitive": "lift_object",
                "args": {"side": "left", "height": 0.08, "steps": 120},
            },
            {
                "thought": "Bring the left gripper toward the shared handover midpoint.",
                "primitive": "move_left_ee",
                "args": {"target": "handover_midpoint", "steps": 100, "pos_tolerance": 0.08},
            },
            {
                "thought": "Close the right gripper to receive the cube.",
                "primitive": "close_gripper",
                "args": {"side": "right"},
            },
            {
                "thought": "Release the cube from the left gripper.",
                "primitive": "release_object",
                "args": {"side": "left"},
            },
            {"thought": "Stop after checking the task metric.", "primitive": "finish", "args": {}},
        ]


class OpenAIPlanner:
    """OpenAI-backed planner using the optional openai Python package."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str | None = "low",
        max_output_tokens: int = 600,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install optional LLM dependencies first: pip install -e .[llm]") from exc
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI()

    def plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        history: list[AgentStepRecord],
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}

        response = self.client.responses.create(**request)
        if hasattr(response, "output_text"):
            return response.output_text
        return str(response)

class AnthropicPlanner:
    """Claude-backed planner using the optional anthropic Python package."""

    provider = "anthropic"

    def __init__(self, model: str) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("Install optional LLM dependencies first: pip install -e .[llm]") from exc
        self.model = model
        self.client = Anthropic()

    def plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        history: list[AgentStepRecord],
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        chunks = []
        for part in response.content:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
        return "\n".join(chunks)


class TargetResolver:
    """Resolve symbolic LLM targets into primitive-friendly Cartesian poses."""

    def __init__(self, env: Any, controller: PrimitiveController) -> None:
        self.env = env
        self.controller = controller

    def resolve_pose(self, target: Any, side: str | HandSide) -> np.ndarray:
        side_enum = self._parse_side(side)
        if isinstance(target, (list, tuple)):
            arr = np.asarray(target, dtype=np.float32)
            if arr.shape == (6,):
                return arr
        current_pose = self.controller.current_ee_pose()[self._pose_slice(side_enum)].copy()
        current_pose[:3] = self.resolve_position(target, side_enum)
        return current_pose

    def resolve_position(self, target: Any, side: str | HandSide | None = None) -> np.ndarray:
        if isinstance(target, (list, tuple)):
            arr = np.asarray(target, dtype=np.float32)
            if arr.shape == (3,):
                return arr
            if arr.shape == (6,):
                return arr[:3]
            raise ValueError("Cartesian target lists must have length 3 or 6.")
        if isinstance(target, dict):
            if "position" in target:
                return self.resolve_position(target["position"], side)
            if "pose" in target:
                return self.resolve_position(target["pose"], side)
            if "target" in target:
                return self.resolve_position(target["target"], side)
        if not isinstance(target, str):
            raise ValueError("Target must be a symbolic string or Cartesian xyz/rpy list.")

        target_key = target.strip().lower()
        if target_key == "handover_midpoint":
            return self._handover_midpoint()
        if target_key == "left_safe_parking":
            return np.asarray([0.45, 0.58, 1.22], dtype=np.float32)
        if target_key == "held_block0_above_block1_high":
            return self._held_block0_stack_target(clearance=0.045)
        if target_key == "held_block0_on_block1":
            return self._held_block0_stack_target(clearance=0.041)

        object_name, offset = self.resolve_object_offset(target_key, side)
        objects = get_task_objects(self.env)
        if object_name not in objects:
            raise ValueError(f"Object {object_name!r} is not available for target {target!r}.")
        return get_object_position(objects[object_name]).astype(np.float32) + offset

    def resolve_object_offset(
        self,
        target_or_object: str,
        side: str | HandSide | None = None,
        explicit_offset: Any | None = None,
    ) -> tuple[str, np.ndarray]:
        key = target_or_object.strip().lower()
        if key in SYMBOLIC_TARGETS:
            spec = SYMBOLIC_TARGETS[key]
            return str(spec["object"]), self._vector(explicit_offset, spec["offset"])

        objects = get_task_objects(self.env)
        if key in objects:
            return key, self._vector(explicit_offset, [0.0, 0.0, 0.1])

        raise ValueError(f"Unknown symbolic target or object: {target_or_object!r}.")

    def _handover_midpoint(self) -> np.ndarray:
        ee_pose = self.controller.current_ee_pose()
        left = ee_pose[0:3]
        right = ee_pose[6:9]
        midpoint = ((left + right) / 2.0).astype(np.float32)
        midpoint[1] = 0.0
        midpoint[2] = max(float(midpoint[2]), 1.08)
        return midpoint
    def _held_block0_stack_target(self, *, clearance: float) -> np.ndarray:
        objects = get_task_objects(self.env)
        if "block_0" not in objects or "block_1" not in objects:
            raise ValueError("held block stacking targets require block_0 and block_1")
        block0 = get_object_position(objects["block_0"]).copy()
        block1 = get_object_position(objects["block_1"]).copy()
        right_pose = self.controller.current_ee_pose()[6:12].copy()
        ee_to_block = right_pose[:3] - block0
        desired_block = block1.copy()
        desired_block[2] = block1[2] + clearance
        return (desired_block + ee_to_block).astype(np.float32)

    def _parse_side(self, side: str | HandSide) -> HandSide:
        if isinstance(side, HandSide):
            return side
        normalized = side.lower()
        if normalized in {"left", "l"}:
            return HandSide.LEFT
        if normalized in {"right", "r"}:
            return HandSide.RIGHT
        raise ValueError(f"Unknown side: {side!r}.")

    def _pose_slice(self, side: HandSide) -> slice:
        return slice(0, 6) if side == HandSide.LEFT else slice(6, 12)

    def _vector(self, value: Any | None, default: Any) -> np.ndarray:
        arr = np.asarray(default if value is None else value, dtype=np.float32)
        if arr.shape != (3,):
            raise ValueError("Offsets must be xyz lists with length 3.")
        return arr


class PrimitiveExecutor:
    """Validate and execute structured LLM action plans through primitives."""

    def __init__(
        self,
        env: Any,
        controller: PrimitiveController,
        *,
        task_key: str | None = None,
        success_threshold: float = 1.0,
    ) -> None:
        self.env = env
        self.controller = controller
        self.resolver = TargetResolver(env, controller)
        self.primary_object = (
            TASK_SPECS[task_key].primary_object
            if task_key
            else next(iter(get_task_objects(env)), "")
        )
        self.success_threshold = success_threshold

    def preview(self, action: ActionPlan) -> PrimitiveResult:
        issue = self._validate_action(action)
        if issue:
            return self._failure(action.primitive, issue, "Ask the LLM to select an allowed primitive.")
        return self._result(
            f"dry_run_{action.primitive}",
            True,
            0,
            "dry run; primitive was parsed but not executed",
            "",
        )

    def execute(self, action: ActionPlan) -> PrimitiveResult:
        issue = self._validate_action(action)
        if issue:
            return self._failure(action.primitive, issue, "Ask the LLM to select an allowed primitive.")

        try:
            primitive = action.primitive
            args = action.args
            if primitive == "move_left_ee":
                return self.controller.move_left_ee(
                    self.resolver.resolve_pose(self._required(args, "target"), "left"),
                    steps=self._int_arg(args, "steps", 80),
                    pos_tolerance=self._float_arg(args, "pos_tolerance", 0.03),
                )
            if primitive == "move_right_ee":
                return self.controller.move_right_ee(
                    self.resolver.resolve_pose(self._required(args, "target"), "right"),
                    steps=self._int_arg(args, "steps", 80),
                    pos_tolerance=self._float_arg(args, "pos_tolerance", 0.03),
                )
            if primitive == "move_both_ee":
                return self.controller.move_both_ee(
                    self.resolver.resolve_pose(self._required(args, "left_target"), "left"),
                    self.resolver.resolve_pose(self._required(args, "right_target"), "right"),
                    steps=self._int_arg(args, "steps", 80),
                    pos_tolerance=self._float_arg(args, "pos_tolerance", 0.03),
                )
            if primitive == "open_gripper":
                return self.controller.open_gripper(
                    self._required(args, "side"),
                    steps=self._int_arg(args, "steps", 25),
                )
            if primitive == "close_gripper":
                return self.controller.close_gripper(
                    self._required(args, "side"),
                    steps=self._int_arg(args, "steps", 25),
                )
            if primitive == "align_to_object":
                side = self._required(args, "side")
                object_name, offset = self._object_and_offset(args, side)
                return self.controller.align_to_object(
                    side,
                    object_name,
                    ee_offset=offset,
                    steps=self._int_arg(args, "steps", 90),
                    pos_tolerance=self._float_arg(args, "pos_tolerance", 0.04),
                )
            if primitive == "grasp_object":
                side = self._required(args, "side")
                object_name, offset = self._object_and_offset(args, side)
                return self.controller.grasp_object(
                    side,
                    object_name,
                    ee_offset=offset,
                    steps=self._int_arg(args, "steps", 90),
                    preopen=self._bool_arg(args, "preopen", True),
                    close_after=self._bool_arg(args, "close_after", True),
                )
            if primitive == "lift_object":
                return self.controller.lift_object(
                    args.get("side", None),
                    height=self._float_arg(args, "height", 0.1),
                    steps=self._int_arg(args, "steps", 70),
                    pos_tolerance=self._float_arg(args, "pos_tolerance", 0.04),
                )
            if primitive == "place_held_object_on_object":
                side = self._required(args, "side")
                held_object = str(self._required(args, "held_object"))
                support_object = str(self._required(args, "support_object"))
                side_enum = self.resolver._parse_side(side)
                move = self.controller.move_left_ee if side_enum == HandSide.LEFT else self.controller.move_right_ee
                high_result = move(
                    self._held_object_above_object_target(
                        side,
                        held_object,
                        support_object,
                        clearance=self._float_arg(args, "high_clearance", 0.045),
                    ),
                    steps=self._int_arg(args, "high_steps", self._int_arg(args, "steps", 240)),
                    pos_tolerance=self._float_arg(args, "pos_tolerance", 0.10),
                )
                if high_result.terminated or high_result.truncated or high_result.task_success >= self.success_threshold:
                    high_result.name = f"place_{held_object}_on_{support_object}"
                    return high_result
                place_result = move(
                    self._held_object_above_object_target(
                        side,
                        held_object,
                        support_object,
                        clearance=self._float_arg(args, "place_clearance", 0.041),
                    ),
                    steps=self._int_arg(args, "place_steps", self._int_arg(args, "steps", 120)),
                    pos_tolerance=self._float_arg(args, "pos_tolerance", 0.10),
                )
                place_result.name = f"place_{held_object}_on_{support_object}"
                place_result.steps += high_result.steps
                if not high_result.success and not place_result.success:
                    place_result.message = f"high move: {high_result.message}; place move: {place_result.message}"
                return place_result
            if primitive == "move_held_object_above_object":
                side = self._required(args, "side")
                target_pose = self._held_object_above_object_target(
                    side,
                    str(self._required(args, "held_object")),
                    str(self._required(args, "support_object")),
                    clearance=self._float_arg(args, "clearance", 0.04),
                )
                side_enum = self.resolver._parse_side(side)
                move = self.controller.move_left_ee if side_enum == HandSide.LEFT else self.controller.move_right_ee
                return move(
                    target_pose,
                    steps=self._int_arg(args, "steps", 120),
                    pos_tolerance=self._float_arg(args, "pos_tolerance", 0.08),
                )
            if primitive == "release_object":
                return self.controller.release_object(
                    self._required(args, "side"),
                    steps=self._int_arg(args, "steps", 25),
                )
            if primitive == "rotate_tool_or_object":
                return self.controller.rotate_tool_or_object(
                    self._required(args, "side"),
                    yaw_delta=self._float_arg(args, "yaw_delta", 0.4),
                    steps=self._int_arg(args, "steps", 60),
                )
            if primitive == "finish":
                task_success = self._task_success()
                return self._result(
                    "finish",
                    task_success >= self.success_threshold,
                    0,
                    "task success metric reached" if task_success >= self.success_threshold else "finish requested before success",
                    "Continue planning until the success metric is reached.",
                )
        except Exception as exc:  # noqa: BLE001 - convert planner mistakes into feedback.
            return self._failure(
                action.primitive,
                f"{type(exc).__name__}: {exc}",
                "Repair the primitive arguments and retry from the latest state.",
            )

        return self._failure(action.primitive, "Unhandled primitive.", "Use an allowed primitive.")

    def _validate_action(self, action: ActionPlan) -> str | None:
        if action.primitive not in PRIMITIVE_SCHEMAS:
            return f"Unknown primitive {action.primitive!r}."
        forbidden = _find_forbidden_arg(action.args)
        if forbidden:
            return f"LLM tried to use low-level control argument {forbidden!r}; use primitives only."
        return None

    def _held_object_above_object_target(
        self,
        side: Any,
        held_object: str,
        support_object: str,
        *,
        clearance: float,
    ) -> np.ndarray:
        objects = get_task_objects(self.env)
        missing = [name for name in (held_object, support_object) if name not in objects]
        if missing:
            raise ValueError(f"Objects not available for placement target: {missing}")

        side_enum = self.resolver._parse_side(side)
        pose = self.controller.current_ee_pose()
        pose_slice = self.resolver._pose_slice(side_enum)
        target = pose[pose_slice].copy()
        held_pos = get_object_position(objects[held_object]).copy()
        support_pos = get_object_position(objects[support_object]).copy()
        ee_to_held = target[:3] - held_pos
        desired_held = support_pos.copy()
        desired_held[2] = support_pos[2] + clearance
        target[:3] = desired_held + ee_to_held
        return target
    def _object_and_offset(self, args: PrimitiveArgs, side: Any) -> tuple[str, np.ndarray]:
        explicit_offset = args.get("ee_offset", args.get("offset"))
        if "target" in args:
            return self.resolver.resolve_object_offset(str(args["target"]), side, explicit_offset)
        object_name = args.get("object", args.get("object_name"))
        if object_name is None:
            object_name = self.primary_object
        return self.resolver.resolve_object_offset(str(object_name), side, explicit_offset)

    def _required(self, args: PrimitiveArgs, key: str) -> Any:
        if key not in args:
            raise ValueError(f"Missing required arg {key!r}.")
        return args[key]

    def _int_arg(self, args: PrimitiveArgs, key: str, default: int) -> int:
        return int(args.get(key, default))

    def _float_arg(self, args: PrimitiveArgs, key: str, default: float) -> float:
        return float(args.get(key, default))

    def _bool_arg(self, args: PrimitiveArgs, key: str, default: bool) -> bool:
        value = args.get(key, default)
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _task_success(self) -> float:
        return float(self.env.get_info().get("task_success", 0.0))

    def _result(
        self,
        name: str,
        success: bool,
        steps: int,
        message: str,
        next_suggestion: str,
    ) -> PrimitiveResult:
        info = self.env.get_info()
        return PrimitiveResult(
            name=name,
            success=success,
            steps=steps,
            message=message,
            task_success=float(info.get("task_success", 0.0)),
            reward=0.0,
            terminated=False,
            truncated=False,
            collisions={
                "env_collision_count": int(info.get("env_collision_count", 0)),
                "self_collision_count": int(info.get("self_collision_count", 0)),
            },
            next_suggestion=next_suggestion,
        )

    def _failure(self, name: str, message: str, next_suggestion: str) -> PrimitiveResult:
        return self._result(name, False, 0, message, next_suggestion)

def unfinished_cleanup_needed(task_key: str, records: list[AgentStepRecord], state: dict[str, Any]) -> bool:
    """Return whether success should continue into a meaningful cleanup step."""

    recommended = recommended_playbook_step(task_key, records)
    if recommended is None:
        return False
    _, next_step = recommended
    primitive = next_step.get("primitive")
    if primitive == "finish":
        return False
    if primitive == "release_object":
        side = str(next_step.get("args", {}).get("side", "")).lower()
        grippers = state.get("robot", {}).get("grippers", {})
        holding = grippers.get(side, {}).get("holding", {})
        return any(bool(value) for value in holding.values())
    return True


class LLMAgent:
    """Closed-loop LLM planner that can only call RoboEval primitives."""

    def __init__(
        self,
        task_key: str,
        env: Any,
        controller: PrimitiveController,
        planner: PlannerClient,
        *,
        execute_primitives: bool = True,
        success_threshold: float = 1.0,
    ) -> None:
        self.task_key = task_key
        self.spec = TASK_SPECS[task_key]
        self.env = env
        self.controller = controller
        self.planner = planner
        self.executor = PrimitiveExecutor(
            env,
            controller,
            task_key=task_key,
            success_threshold=success_threshold,
        )
        self.execute_primitives = execute_primitives
        self.success_threshold = success_threshold

    def run(self, *, max_steps: int = 8, step_callback: Any | None = None) -> AgentRunResult:
        records: list[AgentStepRecord] = []
        previous_state: dict[str, Any] | None = None
        last_result: PrimitiveResult | None = None
        feedback: list[str] = []

        for step_idx in range(max_steps):
            state = collect_env_state(self.env)
            state_summary = summarize_env_state(
                state,
                self.spec,
                last_result=last_result,
                feedback=feedback,
            )
            system_prompt, user_prompt = build_task_prompt(
                self.spec,
                state_summary,
                history=records,
            )
            repair_feedback: list[str] = []
            raw_response = self.planner.plan(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                history=records,
            )
            try:
                action = ActionPlan.from_llm_response(raw_response)
                action, repair_feedback = fill_missing_playbook_args(self.task_key, records, action)
                validation_issue = playbook_validation_issue(self.task_key, records, action)
                if validation_issue:
                    info = self.env.get_info()
                    result = PrimitiveResult(
                        name="playbook_validation",
                        success=False,
                        steps=0,
                        message=validation_issue,
                        task_success=float(info.get("task_success", 0.0)),
                        collisions={
                            "env_collision_count": int(info.get("env_collision_count", 0)),
                            "self_collision_count": int(info.get("self_collision_count", 0)),
                        },
                        next_suggestion="Choose next_recommended_playbook_step exactly unless task_success is already 1.0.",
                    )
                else:
                    result = (
                        self.executor.execute(action)
                        if self.execute_primitives
                        else self.executor.preview(action)
                    )
            except Exception as exc:  # noqa: BLE001 - bad LLM text becomes loop feedback.
                action = ActionPlan(
                    thought="Could not parse planner output.",
                    primitive="parse_error",
                    args={},
                )
                result = PrimitiveResult(
                    name="parse_error",
                    success=False,
                    steps=0,
                    message=f"{type(exc).__name__}: {exc}",
                    next_suggestion="Return exactly one JSON object with thought, primitive, and args.",
                )

            next_state = collect_env_state(self.env)
            if self.execute_primitives:
                feedback = repair_feedback + make_recovery_feedback(
                    self.spec,
                    previous_state=previous_state or state,
                    current_state=next_state,
                    action=action,
                    result=result,
                )
            else:
                feedback = repair_feedback + ["dry_run: primitive parsed but not executed; environment state is unchanged."]
            record = AgentStepRecord(
                index=step_idx,
                state_summary=state_summary,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=raw_response,
                action=action.to_dict(),
                result=result.to_dict(),
                feedback=feedback,
            )
            records.append(record)
            if step_callback is not None:
                step_callback(record, next_state)

            previous_state = next_state
            last_result = result
            if result.task_success >= self.success_threshold and not unfinished_cleanup_needed(self.task_key, records, next_state):
                break
            if action.primitive == "finish":
                break

        final_success = float(collect_env_state(self.env)["metrics"].get("task_success", 0.0))
        return AgentRunResult(
            task_key=self.task_key,
            provider=self.planner.provider,
            model=self.planner.model,
            completed=final_success >= self.success_threshold,
            final_task_success=final_success,
            steps=records,
        )


def recommended_playbook_step(
    task_key: str,
    history: list[AgentStepRecord] | None,
) -> tuple[int, dict[str, Any]] | None:
    """Return the next playbook step, counting only executed successful steps."""

    playbook = TASK_PLAYBOOKS.get(task_key, [])
    if not playbook:
        return None

    progress = 0
    for record in history or []:
        if progress >= len(playbook):
            break
        result = record.result
        if result.get("name") in {"playbook_validation", "parse_error"}:
            continue
        task_success = float(result.get("task_success") or 0.0)
        primitive_success = result.get("success") is True or task_success >= 1.0
        if not primitive_success:
            continue
        expected = playbook[progress]
        action = record.action
        if action.get("primitive") == expected.get("primitive"):
            progress += 1

    index = min(progress, len(playbook) - 1)
    return index, playbook[index]

def fill_missing_playbook_args(
    task_key: str,
    history: list[AgentStepRecord],
    action: ActionPlan,
) -> tuple[ActionPlan, list[str]]:
    """Fill omitted primitive args when the LLM chose the recommended playbook primitive."""

    recommended = recommended_playbook_step(task_key, history)
    if recommended is None:
        return action, []

    index, step = recommended
    if action.primitive != step.get("primitive"):
        return action, [
            f"planner_deviated_from_recommended_step:{index}:{step.get('primitive')}"
        ]

    expected_args = dict(step.get("args", {}))
    missing = [key for key in expected_args if key not in action.args]
    if not missing:
        return action, []

    repaired_args = dict(expected_args)
    repaired_args.update(action.args)
    repaired = ActionPlan(
        thought=action.thought,
        primitive=action.primitive,
        args=repaired_args,
    )
    return repaired, [f"filled_missing_playbook_args:{missing}"]

def playbook_validation_issue(
    task_key: str,
    history: list[AgentStepRecord],
    action: ActionPlan,
) -> str | None:
    """Reject out-of-order playbook actions before they disturb the simulator."""

    recommended = recommended_playbook_step(task_key, history)
    if recommended is None:
        return None

    index, step = recommended
    expected_primitive = step.get("primitive")
    if action.primitive != expected_primitive:
        return (
            f"planner chose {action.primitive!r}, but recommended step {index} "
            f"requires {expected_primitive!r}"
        )

    for key, expected_value in dict(step.get("args", {})).items():
        if key in action.args and action.args[key] != expected_value:
            return (
                f"planner changed arg {key!r} for recommended step {index}: "
                f"expected {expected_value!r}, got {action.args[key]!r}"
            )
    return None

def make_planner(
    provider: str,
    task_key: str,
    model: str | None = None,
    *,
    reasoning_effort: str | None = None,
    max_output_tokens: int = 600,
) -> PlannerClient:
    """Build a planner provider by name."""

    provider = provider.lower()
    if provider == "mock":
        return MockPlanner(task_key)
    if provider == "openai":
        model = model or os.getenv("OPENAI_MODEL") or "gpt-5.6-terra"
        return OpenAIPlanner(
            model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        )
    if provider == "anthropic":
        model = model or os.getenv("ANTHROPIC_MODEL")
        if not model:
            raise ValueError("Set --model or ANTHROPIC_MODEL for the Anthropic planner.")
        return AnthropicPlanner(model)
    raise ValueError(f"Unknown planner provider: {provider}")

def summarize_env_state(
    state: dict[str, Any],
    spec: TaskSpec,
    *,
    last_result: PrimitiveResult | None = None,
    feedback: list[str] | None = None,
) -> dict[str, Any]:
    """Make a compact JSON summary suitable for LLM context."""

    metrics = state.get("metrics", {})
    robot = state.get("robot", {})
    objects = state.get("objects", {})
    grippers = robot.get("grippers", {})
    summary = {
        "task": {
            "key": spec.key,
            "name": state.get("task_name"),
            "goal": spec.success_condition,
            "primary_object": spec.primary_object,
            "stage_meaning": spec.stage_meaning,
        },
        "metrics": {
            "task_success": metrics.get("task_success", metrics.get("success", 0.0)),
            "subtask_progress": metrics.get("subtask_progress", 0.0),
            "task_stage_reached": metrics.get("task_stage_reached", {}),
            "env_collision_count": metrics.get("env_collision_count", 0),
            "self_collision_count": metrics.get("self_collision_count", 0),
            "target_distance": metrics.get("target_distance", {}),
        },
        "grippers": {
            side: {
                "qpos": values.get("qpos"),
                "pinch_position": values.get("pinch_position"),
                "controlled_ee_pose": values.get("controlled_ee_pose"),
                "holding": values.get("holding", {}),
            }
            for side, values in grippers.items()
        },
        "objects": {
            name: {
                "position": values.get("position"),
                "euler_xyz": values.get("euler_xyz"),
                "aabb_size": values.get("aabb_size"),
            }
            for name, values in objects.items()
        },
        "object_distances": state.get("object_distances", {}),
        "available_symbolic_targets": list(SYMBOLIC_TARGETS) + ["left_safe_parking", "held_block0_above_block1_high", "held_block0_on_block1"],
        "last_result": last_result.to_dict() if last_result else None,
        "recovery_feedback": feedback or [],
    }
    return summary


def build_task_prompt(
    spec: TaskSpec,
    state_summary: dict[str, Any],
    *,
    history: list[AgentStepRecord] | None = None,
) -> tuple[str, str]:
    """Build the system and user prompts for one planner step."""

    compact_history = []
    for item in (history or [])[-4:]:
        compact_history.append(
            {
                "step": item.index,
                "action": item.action,
                "result": {
                    "success": item.result.get("success"),
                    "message": item.result.get("message"),
                    "task_success": item.result.get("task_success"),
                    "distances": item.result.get("distances"),
                    "collisions": item.result.get("collisions"),
                    "next_suggestion": item.result.get("next_suggestion"),
                },
                "feedback": item.feedback,
            }
        )

    next_playbook_step = None
    recommended = recommended_playbook_step(spec.key, history or [])
    if recommended is not None:
        next_index, step = recommended
        next_playbook_step = dict(step)
        next_playbook_step["index"] = next_index
    system_prompt = (
        "You are a bimanual RoboEval task planner. Choose exactly one high-level "
        "primitive for the next control step. You must never output robot joint "
        "values, torques, raw MuJoCo actions, or policy weights. Use only the "
        "allowed primitive names and JSON arguments. Return exactly one JSON "
        "object with keys: thought, primitive, args. Do not wrap the JSON in "
        "Markdown. Usually copy the primitive and args exactly from "
        "next_recommended_playbook_step. Do not omit numeric args such as "
        "steps, pos_tolerance, height, or clearances. Only deviate when "
        "the latest observation shows that step is already complete or recovery "
        "is needed. If the task_success metric is already satisfied and no "
        "non-finish cleanup step remains, choose finish. If a non-finish "
        "next_recommended_playbook_step remains, choose it exactly."
    )
    user_payload = {
        "task_key": spec.key,
        "success_condition": spec.success_condition,
        "task_playbook": TASK_PLAYBOOKS.get(spec.key, []),
        "next_recommended_playbook_step": next_playbook_step,
        "selection_rules": [
            "Usually choose next_recommended_playbook_step and copy its primitive and args exactly.",
            "Do not omit numeric args such as steps, pos_tolerance, height, or clearances.",
            "Only deviate from the recommended step when observation or recovery_feedback proves it is unsafe or already complete.",
            "If task_success is already 1.0 but next_recommended_playbook_step is not finish, run that cleanup primitive before finish.",
        ],
        "allowed_primitives": PRIMITIVE_SCHEMAS,
        "symbolic_targets": SYMBOLIC_TARGETS,
        "state": state_summary,
        "recent_history": compact_history,
        "required_response_format": {
            "thought": "short reason for the next primitive",
            "primitive": "one key from allowed_primitives",
            "args": "JSON object containing only primitive-level arguments",
        },
        "example": {
            "thought": "The left gripper should approach the left pot handle.",
            "primitive": "move_left_ee",
            "args": {"target": "left_pot_handle"},
        },
    }
    return system_prompt, json.dumps(user_payload, indent=2)


def make_recovery_feedback(
    spec: TaskSpec,
    *,
    previous_state: dict[str, Any],
    current_state: dict[str, Any],
    action: ActionPlan,
    result: PrimitiveResult,
) -> list[str]:
    """Generate concise recovery hints for the next LLM prompt."""

    hints: list[str] = []
    if not result.success:
        hints.append(f"last_primitive_failed: {result.message}")
        if result.next_suggestion:
            hints.append(f"suggestion: {result.next_suggestion}")

    previous_metrics = previous_state.get("metrics", {})
    current_metrics = current_state.get("metrics", {})
    if int(current_metrics.get("self_collision_count", 0)) > int(previous_metrics.get("self_collision_count", 0)):
        hints.append("self_collision_increased: choose a safer waypoint or separate the arms.")
    if int(current_metrics.get("env_collision_count", 0)) > int(previous_metrics.get("env_collision_count", 0)):
        hints.append("env_collision_increased: raise the approach height or reduce lateral motion.")

    if action.primitive in {"align_to_object", "grasp_object"}:
        before = _min_distance_to_primary(previous_state, spec.primary_object)
        after = _min_distance_to_primary(current_state, spec.primary_object)
        if before is not None and after is not None and after > before - 0.005:
            hints.append("distance_not_improved: retry with a closer symbolic target or larger object offset.")

    if action.primitive == "grasp_object" and not result.success:
        hints.append("grasp_not_detected: align closer to the object's top/handle and close again.")

    return hints


def _min_distance_to_primary(state: dict[str, Any], primary_object: str) -> float | None:
    distances = [
        float(value)
        for key, value in state.get("object_distances", {}).items()
        if key.startswith(f"{primary_object}_to_")
    ]
    if not distances:
        return None
    return min(distances)


def _find_forbidden_arg(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, subvalue in value.items():
            key_lower = str(key).lower()
            for fragment in FORBIDDEN_ARG_FRAGMENTS:
                if fragment in key_lower:
                    return key
            found = _find_forbidden_arg(subvalue)
            if found:
                return found
    if isinstance(value, list):
        for subvalue in value:
            found = _find_forbidden_arg(subvalue)
            if found:
                return found
    return None


def _parse_first_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])
