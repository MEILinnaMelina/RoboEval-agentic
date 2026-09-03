"""Run real online LLM-agent RoboEval rollouts.

Phase 7 differs from the deterministic Phase 5 solver: every control step is
chosen by an LLM from the current observation, recent history, and the allowed
primitive API. The LLM never outputs joint values or MuJoCo actions directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from absl import logging as absl_logging

from roboeval.agentic.llm_agent import LLMAgent, make_planner
from roboeval.agentic.primitives import PrimitiveController
from roboeval.agentic.state import collect_env_state
from roboeval.agentic.task_specs import TASK_SPECS, make_task_env


TASKS = ("lift_pot", "cube_handover", "stack_two_blocks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--provider", choices=("openai", "anthropic", "mock"), default="openai")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-output-tokens", type=int, default=600)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--record-gif", action="store_true")
    parser.add_argument("--gif-every", type=int, default=15)
    parser.add_argument("--gif-duration", type=float, default=0.12)
    parser.add_argument("--kinematic-attachments", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "phase7_llm")
    return parser.parse_args()


def save_frame(env: Any, path: Path) -> None:
    obs = env.get_observation()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.moveaxis(obs["rgb_external"], 0, -1))


def sanitize_label(label: str) -> str:
    keep = []
    for char in label.lower():
        keep.append(char if char.isalnum() else "_")
    return "".join(keep).strip("_") or "frame"


def write_gif(frame_paths: list[Path], path: Path, duration_s: float) -> str | None:
    if not frame_paths:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [imageio.imread(frame_path) for frame_path in frame_paths]
    imageio.mimsave(path, frames, duration=max(0.01, duration_s))
    return str(path)


def visual_diagnostics_from_state(state: dict[str, Any]) -> dict[str, Any]:
    grippers = state.get("robot", {}).get("grippers", {})
    return {
        "gripper_qpos": {side: data.get("qpos") for side, data in grippers.items()},
        "holding": {side: data.get("holding", {}) for side, data in grippers.items()},
        "pinch_position": {side: data.get("pinch_position") for side, data in grippers.items()},
        "object_positions": {
            name: value.get("position")
            for name, value in state.get("objects", {}).items()
        },
        "object_distances": state.get("object_distances", {}),
        "metrics": state.get("metrics", {}),
    }


def visual_diagnostics(env: Any) -> dict[str, Any]:
    return visual_diagnostics_from_state(collect_env_state(env))


def task_keys(selection: str) -> list[str]:
    return list(TASKS) if selection == "all" else [selection]


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def final_metrics(env: Any) -> dict[str, Any]:
    state = collect_env_state(env)
    metrics = state.get("metrics", {})
    raw_task_success = metrics.get("task_success", metrics.get("success", 0.0))
    return {
        "task_success": numeric(raw_task_success) or 0.0,
        "subtask_progress": numeric(metrics.get("subtask_progress")),
        "env_collision_count": numeric(metrics.get("env_collision_count")),
        "self_collision_count": numeric(metrics.get("self_collision_count")),
        "slip_count": numeric(metrics.get("slip_count")),
        "total_cartesian_path_length": numeric(metrics.get("total_cartesian_path_length")),
        "avg_cartesian_path_length": numeric(metrics.get("avg_cartesian_path_length")),
        "bimanual_arm_velocity_difference": numeric(metrics.get("bimanual_arm_velocity_difference")),
        "bimanual_gripper_vertical_difference": numeric(metrics.get("bimanual_gripper_vertical_difference")),
        "raw_metrics": metrics,
    }


def compact_step(step: dict[str, Any]) -> dict[str, Any]:
    result = step.get("result", {})
    action = step.get("action", {})
    return {
        "index": step.get("index"),
        "primitive": action.get("primitive"),
        "args": action.get("args"),
        "primitive_success": result.get("success"),
        "message": result.get("message"),
        "task_success": result.get("task_success"),
        "collisions": result.get("collisions"),
        "feedback": step.get("feedback"),
    }


def first_failed_step(report: dict[str, Any]) -> dict[str, Any] | None:
    for step in report.get("steps", []):
        result = step.get("result", {})
        if result.get("success") is False and numeric(result.get("task_success")) != 1.0:
            return compact_step(step)
    if report.get("steps"):
        return compact_step(report["steps"][-1])
    return None


def run_one(
    *,
    task_key: str,
    trial_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trial_dir = args.output_dir / task_key / f"trial_{trial_index:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    env = make_task_env(
        task_key,
        render_mode="human" if args.window else "rgb_array",
        ee=True,
        include_camera=True,
    )
    try:
        env.reset()
        save_frame(env, trial_dir / "before.png")

        frame_paths: list[Path] = []
        step_diagnostics: list[dict[str, Any]] = []

        def capture_frame(label: str) -> None:
            if not args.record_gif:
                return
            frame_path = trial_dir / "frames" / f"{len(frame_paths):04d}_{sanitize_label(label)}.png"
            save_frame(env, frame_path)
            frame_paths.append(frame_path)

        capture_frame("before")

        controller = PrimitiveController(
            env,
            render=args.window,
            sleep_s=0.02 if args.window else 0.0,
            frame_callback=(lambda _env, step: capture_frame(f"sim_step_{step:06d}")) if args.record_gif else None,
            frame_every=args.gif_every,
            kinematic_attachments=args.kinematic_attachments,
        )
        planner = make_planner(
            args.provider,
            task_key,
            args.model,
            reasoning_effort=args.reasoning_effort if args.provider == "openai" else None,
            max_output_tokens=args.max_output_tokens,
        )
        agent = LLMAgent(task_key, env, controller, planner, execute_primitives=True)

        def after_agent_step(record: Any, next_state: dict[str, Any]) -> None:
            step_diagnostics.append(
                {
                    "index": record.index,
                    "primitive": record.action.get("primitive"),
                    "args": record.action.get("args"),
                    "result": record.result,
                    "state_after": visual_diagnostics_from_state(next_state),
                }
            )
            capture_frame(f"agent_step_{record.index:02d}_{record.action.get('primitive', 'unknown')}")

        result = agent.run(max_steps=args.max_steps, step_callback=after_agent_step)
        save_frame(env, trial_dir / "after.png")
        capture_frame("after")
        trajectory_gif = write_gif(frame_paths, trial_dir / "trajectory.gif", args.gif_duration) if args.record_gif else None

        report = result.to_dict()
        metrics = final_metrics(env)
        diagnostics = {
            "final": visual_diagnostics(env),
            "per_agent_step": step_diagnostics,
            "trajectory_gif": trajectory_gif,
        }
        report["phase"] = 7
        report["kinematic_attachments"] = args.kinematic_attachments
        report["trial_index"] = trial_index
        report["trial_dir"] = str(trial_dir)
        report["final_metrics"] = metrics
        report["screenshots"] = {
            "before": str(trial_dir / "before.png"),
            "after": str(trial_dir / "after.png"),
            "trajectory_gif": trajectory_gif,
        }
        report["visual_diagnostics"] = diagnostics
        report["compact_steps"] = [compact_step(step) for step in report.get("steps", [])]
        report_path = trial_dir / "llm_agent_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        success = bool(result.completed)
        case_path = trial_dir / ("success_trajectory_log.json" if success else "failure_case.json")
        case_payload = {
            "task_key": task_key,
            "trial_index": trial_index,
            "success": success,
            "provider": args.provider,
            "model": args.model,
            "kinematic_attachments": args.kinematic_attachments,
            "final_task_success": result.final_task_success,
            "final_metrics": metrics,
            "report_path": str(report_path),
            "planning_trace": report["compact_steps"],
            "visual_diagnostics": diagnostics,
            "trajectory_gif": trajectory_gif,
            "failed_step": None if success else first_failed_step(report),
        }
        case_path.write_text(json.dumps(case_payload, indent=2), encoding="utf-8")

        return {
            "task_key": task_key,
            "trial_index": trial_index,
            "success": success,
            "provider": args.provider,
            "model": args.model,
            "kinematic_attachments": args.kinematic_attachments,
            "final_task_success": result.final_task_success,
            "steps": len(result.steps),
            "final_metrics": metrics,
            "trial_dir": str(trial_dir),
            "report_path": str(report_path),
            "case_log_path": str(case_path),
            "trajectory_gif": trajectory_gif,
            "failed_step": None if success else first_failed_step(report),
        }
    finally:
        env.close()


def mean_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for task_key in sorted({record["task_key"] for record in records}):
        task_records = [record for record in records if record["task_key"] == task_key]
        metrics = [record["final_metrics"] for record in task_records]
        trials = len(task_records)
        successes = sum(1 for record in task_records if record["success"])
        summaries.append(
            {
                "task_key": task_key,
                "trials": trials,
                "successes": successes,
                "success_rate": successes / trials if trials else 0.0,
                "mean_steps": mean_present([numeric(record.get("steps")) for record in task_records]),
                "mean_task_success": mean_present([metric.get("task_success") for metric in metrics]),
                "mean_subtask_progress": mean_present([metric.get("subtask_progress") for metric in metrics]),
                "mean_env_collision_count": mean_present([metric.get("env_collision_count") for metric in metrics]),
                "mean_self_collision_count": mean_present([metric.get("self_collision_count") for metric in metrics]),
                "mean_slip_count": mean_present([metric.get("slip_count") for metric in metrics]),
                "mean_total_cartesian_path_length": mean_present([metric.get("total_cartesian_path_length") for metric in metrics]),
                "mean_bimanual_arm_velocity_difference": mean_present([metric.get("bimanual_arm_velocity_difference") for metric in metrics]),
                "mean_bimanual_gripper_vertical_difference": mean_present([metric.get("bimanual_gripper_vertical_difference") for metric in metrics]),
                "failure_count": trials - successes,
            }
        )
    return summaries


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "task_key",
        "trials",
        "successes",
        "success_rate",
        "mean_steps",
        "mean_task_success",
        "mean_subtask_progress",
        "mean_env_collision_count",
        "mean_self_collision_count",
        "mean_slip_count",
        "mean_total_cartesian_path_length",
        "mean_bimanual_arm_velocity_difference",
        "mean_bimanual_gripper_vertical_difference",
        "failure_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})


def main() -> None:
    absl_logging.set_verbosity(absl_logging.ERROR)
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for task_key in task_keys(args.task):
        for trial_index in range(1, args.trials + 1):
            print(f"[phase7] {task_key} trial {trial_index}/{args.trials}", flush=True)
            record = run_one(task_key=task_key, trial_index=trial_index, args=args)
            records.append(record)
            status = "success" if record["success"] else "failure"
            print(
                f"[phase7] {task_key} trial {trial_index}: {status}, "
                f"task_success={record['final_task_success']}, steps={record['steps']}",
                flush=True,
            )

    summaries = summarize(records)
    report = {
        "phase": 7,
        "provider": args.provider,
        "model": args.model,
        "trials_per_task": args.trials,
        "summaries": summaries,
        "records": records,
        "failure_cases": [record for record in records if not record["success"]],
    }
    report_path = args.output_dir / "phase7_llm_eval_report.json"
    summary_path = args.output_dir / "phase7_llm_summary.csv"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary_csv(summary_path, summaries)

    print("\nPhase 7 LLM summary")
    for summary in summaries:
        print(
            f"- {summary['task_key']}: {summary['successes']}/{summary['trials']} "
            f"success_rate={summary['success_rate']:.2f}, "
            f"mean_task_success={summary['mean_task_success']}"
        )
    print(f"saved_report: {report_path.resolve()}")
    print(f"saved_summary_csv: {summary_path.resolve()}")


if __name__ == "__main__":
    main()