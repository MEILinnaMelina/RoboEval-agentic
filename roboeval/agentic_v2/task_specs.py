"""Task registry and fixed environment construction for Agentic v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from roboeval.action_modes import JointPositionActionMode
from roboeval.envs.lift_pot import LiftPot
from roboeval.envs.manipulation import CubeHandover, StackTwoBlocks
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.utils.observation_config import CameraConfig, ObservationConfig


@dataclass(frozen=True)
class TaskSpec:
    key: str
    env_cls: type
    objects: tuple[str, ...]
    success_condition: str
    stage_meaning: dict[int, str]


TASK_SPECS: dict[str, TaskSpec] = {
    "lift_pot": TaskSpec(
        key="lift_pot",
        env_cls=LiftPot,
        objects=("kitchenpot",),
        success_condition=(
            "Lift the pot at least 0.10 m without cabinet/floor collision while "
            "both grippers hold it and its orientation remains valid."
        ),
        stage_meaning={
            1: "left gripper holds the pot",
            2: "right gripper holds the pot",
            3: "pot exceeds the lift threshold",
            4: "both holds, lift, and orientation are valid",
        },
    ),
    "cube_handover": TaskSpec(
        key="cube_handover",
        env_cls=CubeHandover,
        objects=("cube",),
        success_condition=(
            "Establish an initial holder, transfer the rod to the opposite "
            "gripper, and release the initial gripper without dropping it."
        ),
        stage_meaning={
            1: "one gripper has held the rod",
            2: "the opposite gripper has held the rod",
        },
    ),
    "stack_two_blocks": TaskSpec(
        key="stack_two_blocks",
        env_cls=StackTwoBlocks,
        objects=("block_0", "block_1"),
        success_condition=(
            "One block rests on the table, the other contacts only that block, "
            "and neither gripper holds a block."
        ),
        stage_meaning={
            1: "at least one block has been held",
            2: "the other block has also been held",
            3: "blocks have stacked contact without upper-table contact",
        },
    ),
}

BASE_TASK_KEYS = ("cube_handover", "lift_pot", "stack_two_blocks")


def task_key_from_env(env: Any) -> str:
    for key, spec in TASK_SPECS.items():
        if isinstance(env, spec.env_cls):
            return key
    raise ValueError(f"unsupported Agentic v2 environment: {type(env).__name__}")


def make_task_env(
    task_key: str,
    *,
    render_mode: str | None = None,
    control_frequency: int = 20,
    include_camera: bool = False,
    camera_resolution: tuple[int, int] = (256, 256),
) -> Any:
    """Construct a base task with v2's fixed absolute joint action mode."""

    spec = TASK_SPECS[task_key]
    cameras = []
    if include_camera:
        cameras.append(
            CameraConfig(
                name="external",
                rgb=True,
                depth=False,
                resolution=camera_resolution,
            )
        )
    return spec.env_cls(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            ee=False,
            floating_dofs=[],
        ),
        render_mode=render_mode,
        control_frequency=control_frequency,
        robot_cls=BimanualPanda,
        observation_config=ObservationConfig(cameras=cameras),
    )
