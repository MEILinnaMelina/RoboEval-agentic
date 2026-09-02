"""Smoke-test RoboEval task creation, stepping, and rendering."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from roboeval.action_modes import JointPositionActionMode
from roboeval.envs.lift_pot import LiftPot
from roboeval.envs.manipulation import CubeHandover, StackTwoBlocks
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.utils.observation_config import CameraConfig, ObservationConfig


TASKS = {
    "lift_pot": LiftPot,
    "cube_handover": CubeHandover,
    "stack_two_blocks": StackTwoBlocks,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=TASKS,
        default="lift_pot",
        help="RoboEval task to load.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of simulation steps to render.",
    )
    parser.add_argument(
        "--control-frequency",
        type=int,
        default=20,
        help="Environment control frequency.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open the MuJoCo window; only save an image.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=Path("outputs") / "smoke_render.png",
        help="Where to save the rendered camera image.",
    )
    return parser.parse_args()


def make_env(task_name: str, render_mode: str | None, control_frequency: int):
    task_cls = TASKS[task_name]
    return task_cls(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            floating_dofs=[],
        ),
        render_mode=render_mode,
        control_frequency=control_frequency,
        robot_cls=BimanualPanda,
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(
                    name="external",
                    rgb=True,
                    depth=False,
                    resolution=(256, 256),
                )
            ],
        ),
    )


def main() -> None:
    args = parse_args()
    render_mode = None if args.no_window else "human"
    env = make_env(args.task, render_mode, args.control_frequency)

    try:
        obs, info = env.reset()
        rgb = np.moveaxis(obs["rgb_external"], 0, -1)
        args.image_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(args.image_path, rgb)

        action = np.clip(
            np.zeros(env.action_space.shape, dtype=np.float32),
            env.action_space.low,
            env.action_space.high,
        )

        print(f"task: {env.task_name}")
        print(f"action_space: {env.action_space.shape}")
        print(f"initial_info: {info}")
        print(f"saved_image: {args.image_path.resolve()}")

        for step_idx in range(args.steps):
            if env.render_mode:
                env.render()
            _, reward, terminated, truncated, info = env.step(action, fast=False)
            if step_idx % 50 == 0:
                print(
                    f"step={step_idx:04d} reward={reward:.3f} "
                    f"terminated={terminated} truncated={truncated} "
                    f"task_success={info.get('task_success')}"
                )
            if terminated or truncated:
                break
            time.sleep(0.02)

        print("smoke render finished")
    finally:
        env.close()


if __name__ == "__main__":
    main()
