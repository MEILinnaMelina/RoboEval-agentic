"""Run Phase 5 task-level primitive solvers for RoboEval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from absl import logging as absl_logging

from roboeval.agentic.task_solvers import TaskSolver
from roboeval.agentic.primitives import PrimitiveController
from roboeval.agentic.task_specs import TASK_SPECS, make_task_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASK_SPECS, "all"], default="all")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "phase5_solvers")
    return parser.parse_args()


def save_frame(env, path: Path) -> None:
    obs = env.get_observation()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.moveaxis(obs["rgb_external"], 0, -1))


def compact_report(report: dict) -> dict:
    return {
        "task_key": report["task_key"],
        "success": report["success"],
        "final_task_success": report["final_task_success"],
        "failure_reasons": report["failure_reasons"],
        "steps": [
            {
                "index": step["index"],
                "label": step["label"],
                "primitive": step["action"]["primitive"],
                "success": step["result"]["success"],
                "task_completed": step["result"].get("task_completed", False),
                "task_success": step["result"]["task_success"],
                "message": step["result"]["message"],
                "distances": step["result"]["distances"],
                "collisions": step["result"]["collisions"],
                "holding": step["holding"],
            }
            for step in report["steps"]
        ],
    }


def run_task(task_key: str, args: argparse.Namespace) -> dict:
    env = make_task_env(
        task_key,
        render_mode="human" if args.window else "rgb_array",
        ee=True,
        include_camera=True,
    )
    try:
        env.reset()
        save_frame(env, args.output_dir / f"{task_key}_before.png")

        controller = PrimitiveController(env, render=args.window, sleep_s=0.02 if args.window else 0.0)
        solver = TaskSolver(task_key, env, controller)
        report = solver.solve().to_dict()

        save_frame(env, args.output_dir / f"{task_key}_after.png")
        return report
    finally:
        env.close()


def main() -> None:
    absl_logging.set_verbosity(absl_logging.ERROR)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_keys = list(TASK_SPECS) if args.task == "all" else [args.task]
    reports = {task_key: run_task(task_key, args) for task_key in task_keys}

    report_path = args.output_dir / "phase5_solver_report.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    compact = {task_key: compact_report(report) for task_key, report in reports.items()}
    print(json.dumps(compact, indent=2))
    print(f"saved_report: {report_path.resolve()}")
    print(f"saved_images: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
