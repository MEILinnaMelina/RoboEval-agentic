"""Run low-level RoboEval primitives without any LLM planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from absl import logging as absl_logging

from roboeval.agentic.primitives import PrimitiveController
from roboeval.agentic.task_specs import TASK_SPECS, make_task_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_SPECS, default="cube_handover")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "primitive_smoke")
    return parser.parse_args()


def object_offsets(task_key: str) -> dict[str, np.ndarray]:
    if task_key == "lift_pot":
        return {
            "left": np.array([0.0, 0.16, 0.1]),
            "right": np.array([0.0, -0.16, 0.1]),
        }
    if task_key == "stack_two_blocks":
        return {
            "left": np.array([0.0, 0.0, 0.1]),
            "right": np.array([0.0, 0.0, 0.1]),
        }
    return {
        "left": np.array([0.0, 0.0, 0.1]),
        "right": np.array([0.0, 0.0, 0.1]),
    }


def smoke_side(task_key: str) -> str:
    if task_key == "stack_two_blocks":
        return "right"
    return "left"


def smoke_lift_height(task_key: str) -> float:
    if task_key in {"lift_pot", "stack_two_blocks"}:
        return 0.05
    return 0.08


def save_frame(env, path: Path) -> None:
    obs = env.get_observation()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.moveaxis(obs["rgb_external"], 0, -1))


def main() -> None:
    absl_logging.set_verbosity(absl_logging.ERROR)
    args = parse_args()
    env = make_task_env(
        args.task,
        render_mode="human" if args.window else None,
        ee=True,
        include_camera=True,
    )

    try:
        env.reset()
        controller = PrimitiveController(env, render=args.window, sleep_s=0.02 if args.window else 0.0)
        spec = TASK_SPECS[args.task]
        offsets = object_offsets(args.task)
        side = smoke_side(args.task)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_frame(env, args.output_dir / f"{args.task}_before.png")

        results = []
        results.append(controller.open_gripper(side).to_dict())
        results.append(
            controller.align_to_object(
                side,
                spec.primary_object,
                ee_offset=offsets[side],
                steps=140,
                pos_tolerance=0.06,
            ).to_dict()
        )
        results.append(controller.close_gripper(side).to_dict())
        results.append(
            controller.lift_object(
                side,
                height=smoke_lift_height(args.task),
                steps=100,
                pos_tolerance=0.08,
            ).to_dict()
        )
        results.append(controller.release_object(side).to_dict())

        save_frame(env, args.output_dir / f"{args.task}_after.png")
        print(json.dumps(results, indent=2))
        print(f"saved_images: {args.output_dir.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
