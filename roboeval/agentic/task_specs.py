"""Task registry for the first RoboEval agentic experiments."""

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
    """Static information an LLM planner should know about a task."""

    key: str
    env_cls: type
    primary_object: str
    success_condition: str
    stage_meaning: dict[int, str]


TASK_SPECS: dict[str, TaskSpec] = {
    "lift_pot": TaskSpec(
        key="lift_pot",
        env_cls=LiftPot,
        primary_object="kitchenpot",
        success_condition=(
            "Both grippers should grasp the pot, lift it above the table, avoid "
            "cabinet/floor collision, and keep the pot orientation stable."
        ),
        stage_meaning={
            1: "left gripper contacts/grips the pot",
            2: "right gripper contacts/grips the pot",
            3: "pot has been lifted above the height threshold",
            4: "pot is lifted while both grippers hold it with acceptable orientation",
        },
    ),
    "cube_handover": TaskSpec(
        key="cube_handover",
        env_cls=CubeHandover,
        primary_object="cube",
        success_condition=(
            "One gripper should hold the rod/cube first, transfer it to the opposite "
            "gripper, and release it from the initial gripper without dropping it."
        ),
        stage_meaning={
            1: "some gripper is holding the object",
            2: "object has been transferred to the opposite gripper",
        },
    ),
    "stack_two_blocks": TaskSpec(
        key="stack_two_blocks",
        env_cls=StackTwoBlocks,
        primary_object="block_0",
        success_condition=(
            "The two blocks should be stacked, the lower block should remain on the "
            "table, the upper block should contact only the lower block, and both "
            "grippers should release the blocks."
        ),
        stage_meaning={
            1: "at least one block has been grasped",
            2: "the other block has also been grasped at some point",
            3: "blocks are stacked without the upper block touching the table",
        },
    ),
}


def make_task_env(
    task_key: str,
    *,
    render_mode: str | None = None,
    control_frequency: int = 20,
    ee: bool = False,
    include_camera: bool = True,
    camera_resolution: tuple[int, int] = (256, 256),
) -> Any:
    """Create one of the initial RoboEval tasks with consistent defaults."""

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
            ee=ee,
            floating_dofs=[],
        ),
        render_mode=render_mode,
        control_frequency=control_frequency,
        robot_cls=BimanualPanda,
        observation_config=ObservationConfig(cameras=cameras),
    )
