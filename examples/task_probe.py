"""Print RoboEval task observations and save a rendered image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from roboeval.agentic.state import collect_env_state
from roboeval.agentic.task_specs import TASK_SPECS, make_task_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASK_SPECS, "all"], default="all")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "task_probe")
    return parser.parse_args()


def run_probe(task_key: str, args: argparse.Namespace) -> dict:
    env = make_task_env(
        task_key,
        render_mode="human" if args.window else "rgb_array",
        ee=False,
        include_camera=True,
    )
    try:
        obs, info = env.reset()
        action = env.robot.qpos_actuated.copy()
        action[-len(env.robot.grippers) :] = 1.0 - env.robot.qpos_grippers
        action = np.clip(action.astype(np.float32), env.action_space.low, env.action_space.high)
        for _ in range(args.steps):
            obs, _, _, _, info = env.step(action, fast=False)
            if args.window:
                env.render()

        state = collect_env_state(env, info)
        spec = TASK_SPECS[task_key]
        state["task_spec"] = {
            "key": spec.key,
            "primary_object": spec.primary_object,
            "success_condition": spec.success_condition,
            "stage_meaning": spec.stage_meaning,
        }

        image_path = args.output_dir / f"{task_key}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(image_path, np.moveaxis(obs["rgb_external"], 0, -1))
        state["saved_image"] = str(image_path.resolve())

        return state
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    task_keys = list(TASK_SPECS) if args.task == "all" else [args.task]
    report = {task_key: run_probe(task_key, args) for task_key in task_keys}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "task_probe_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"saved_report: {json_path.resolve()}")


if __name__ == "__main__":
    main()
