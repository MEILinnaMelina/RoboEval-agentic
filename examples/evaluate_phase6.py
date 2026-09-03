"""Run Phase 6 repeated evaluations for the agentic RoboEval solvers.

The script intentionally reuses examples/solve_tasks_demo.py as the execution
entrypoint. That keeps Phase 6 aligned with the same primitive-only task
solvers used in Phase 5, while adding repeated trials, metric aggregation, and
failure/success trajectory records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


TASKS = ("lift_pot", "cube_handover", "stack_two_blocks")
TASK_MAX_STAGE = {
    "lift_pot": 3,
    "cube_handover": 3,
    "stack_two_blocks": 3,
}


@dataclass
class TrialRecord:
    task: str
    trial_index: int
    success: bool
    return_code: int
    trial_dir: Path
    report_path: Path | None
    final_task_success: float | None
    subtask_progress: float | None
    metrics: dict[str, Any]
    failed_step: dict[str, Any] | None
    success_log_path: Path | None
    failure_log_path: Path | None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _task_selection(value: str) -> list[str]:
    return list(TASKS) if value == "all" else [value]


def _run_trial(
    *,
    task: str,
    trial_index: int,
    output_dir: Path,
    window: bool,
    timeout_sec: int,
) -> tuple[int, Path, Path | None]:
    trial_dir = output_dir / task / f"trial_{trial_index:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(_repo_root() / "examples" / "solve_tasks_demo.py"),
        "--task",
        task,
        "--output-dir",
        str(trial_dir),
    ]
    if window:
        command.append("--window")

    _write_json(
        trial_dir / "command.json",
        {
            "command": command,
            "task": task,
            "trial_index": trial_index,
            "timeout_sec": timeout_sec,
        },
    )

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        completed = subprocess.run(
            command,
            cwd=str(_repo_root()),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_sec,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTIMEOUT after {timeout_sec} seconds\n"

    (trial_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (trial_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    (trial_dir / "returncode.txt").write_text(str(return_code), encoding="utf-8")

    report_path = trial_dir / "phase5_solver_report.json"
    return return_code, trial_dir, report_path if report_path.exists() else None


def _extract_task_report(raw: Any, task: str) -> dict[str, Any]:
    if isinstance(raw, dict):
        if task in raw and isinstance(raw[task], dict):
            return raw[task]
        for key in ("reports", "tasks", "results"):
            value = raw.get(key)
            if isinstance(value, dict) and task in value and isinstance(value[task], dict):
                return value[task]
            if isinstance(value, list):
                for item in value:
                    if _looks_like_task_report(item, task):
                        return item
        if _looks_like_task_report(raw, task):
            return raw
    if isinstance(raw, list):
        for item in raw:
            if _looks_like_task_report(item, task):
                return item
    return {}


def _looks_like_task_report(value: Any, task: str) -> bool:
    if not isinstance(value, dict):
        return False
    task_values = {
        value.get("task"),
        value.get("task_key"),
        value.get("key"),
        value.get("name"),
    }
    return task in task_values or bool(value.get("steps"))


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def _last_number_for_keys(value: Any, keys: Iterable[str]) -> float | None:
    key_set = set(keys)
    found: float | None = None
    for node in _walk(value):
        if isinstance(node, dict):
            for key in key_set:
                number = _to_float(node.get(key))
                if number is not None:
                    found = number
    return found


def _max_number_for_keys(value: Any, keys: Iterable[str]) -> float | None:
    key_set = set(keys)
    found: list[float] = []
    for node in _walk(value):
        if isinstance(node, dict):
            for key in key_set:
                number = _to_float(node.get(key))
                if number is not None:
                    found.append(number)
    return max(found) if found else None


def _find_metric_dict(report: dict[str, Any]) -> dict[str, Any]:
    for key in ("metrics", "final_metrics", "roboeval_metrics", "eval_metrics"):
        value = report.get(key)
        if isinstance(value, dict):
            return value

    best: dict[str, Any] = {}
    metric_markers = {
        "task_success",
        "subtask_progress",
        "collision_count",
        "env_collision_count",
        "self_collision_count",
        "slip_count",
        "path_length",
        "bimanual_arm_velocity_difference",
        "bimanual_gripper_vertical_difference",
    }
    for node in _walk(report):
        if isinstance(node, dict) and metric_markers.intersection(node.keys()):
            best = node
    return best


def _flatten_numeric(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            clean_key = str(key).replace(" ", "_")
            child_prefix = f"{prefix}.{clean_key}" if prefix else clean_key
            _flatten_numeric(child_prefix, child, output)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _flatten_numeric(f"{prefix}.{index}", child, output)
    else:
        number = _to_float(value)
        if number is not None:
            output[prefix] = number


def _selected_metrics(task: str, report: dict[str, Any]) -> dict[str, Any]:
    raw_metrics = _find_metric_dict(report)
    final_task_success = _last_number_for_keys(
        report,
        ("task_success", "final_task_success", "success_rate"),
    )
    stage = _max_number_for_keys(report, ("stage", "task_stage", "subtask_stage"))
    explicit_progress = _last_number_for_keys(
        report,
        ("subtask_progress", "progress", "task_progress"),
    )
    if explicit_progress is not None:
        subtask_progress = explicit_progress
    elif stage is not None:
        subtask_progress = min(stage / TASK_MAX_STAGE.get(task, 3), 1.0)
    else:
        subtask_progress = final_task_success

    env_collisions = _last_number_for_keys(report, ("env_collision_count",))
    self_collisions = _last_number_for_keys(report, ("self_collision_count",))
    collision_count = _last_number_for_keys(report, ("collision_count", "collisions"))
    if collision_count is None:
        collision_count = (env_collisions or 0.0) + (self_collisions or 0.0)

    slip_count = _last_number_for_keys(report, ("slip_count", "slips"))
    path_length = _last_number_for_keys(
        report,
        (
            "path_length",
            "total_cartesian_path_length",
            "overall_path_length",
            "cartesian_path_length",
            "avg_cartesian_path_length",
            "ee_path_length",
        ),
    )

    bimanual_coordination = {
        "arm_velocity_difference": _last_number_for_keys(
            report,
            ("bimanual_arm_velocity_difference",),
        ),
        "gripper_vertical_difference": _last_number_for_keys(
            report,
            ("bimanual_gripper_vertical_difference",),
        ),
    }

    flattened_raw: dict[str, float] = {}
    _flatten_numeric("", raw_metrics, flattened_raw)

    return {
        "task_success": final_task_success,
        "subtask_progress": subtask_progress,
        "collision_count": collision_count,
        "env_collision_count": env_collisions,
        "self_collision_count": self_collisions,
        "slip_count": slip_count,
        "path_length": path_length,
        "bimanual_coordination": bimanual_coordination,
        "raw_numeric_metrics": flattened_raw,
    }


def _report_success(report: dict[str, Any], return_code: int) -> bool:
    if return_code != 0:
        return False
    direct_success = report.get("success")
    if isinstance(direct_success, bool) and direct_success:
        return True
    task_success = _last_number_for_keys(report, ("task_success", "final_task_success"))
    return task_success is not None and task_success >= 1.0


def _steps(report: dict[str, Any]) -> list[dict[str, Any]]:
    value = report.get("steps")
    if isinstance(value, list):
        return [step for step in value if isinstance(step, dict)]
    for node in _walk(report):
        if isinstance(node, dict):
            value = node.get("steps")
            if isinstance(value, list):
                return [step for step in value if isinstance(step, dict)]
    return []


def _step_result(step: dict[str, Any]) -> dict[str, Any]:
    result = step.get("result")
    if isinstance(result, dict):
        return result
    return step


def _primitive_name(step: dict[str, Any]) -> str:
    plan = step.get("plan")
    if isinstance(plan, dict) and plan.get("primitive"):
        return str(plan["primitive"])
    if step.get("primitive"):
        return str(step["primitive"])
    result = _step_result(step)
    if result.get("name"):
        return str(result["name"])
    return ""


def _failed_step(report: dict[str, Any]) -> dict[str, Any] | None:
    steps = _steps(report)
    for index, step in enumerate(steps):
        result = _step_result(step)
        primitive_success = result.get("success")
        task_completed = result.get("task_completed")
        task_success = _to_float(result.get("task_success"))
        if primitive_success is False and not task_completed and task_success != 1.0:
            return _format_step(index, step)
    if steps:
        return _format_step(len(steps) - 1, steps[-1])
    return None


def _format_step(index: int, step: dict[str, Any]) -> dict[str, Any]:
    result = _step_result(step)
    plan = step.get("plan") if isinstance(step.get("plan"), dict) else step
    return {
        "step_index": index,
        "thought": plan.get("thought"),
        "primitive": _primitive_name(step),
        "args": plan.get("args"),
        "primitive_success": result.get("success"),
        "task_completed": result.get("task_completed"),
        "task_success": result.get("task_success"),
        "message": result.get("message"),
        "distances": result.get("distances"),
        "collisions": result.get("collisions"),
        "next_suggestion": result.get("next_suggestion"),
    }


def _planning_trace(report: dict[str, Any]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for index, step in enumerate(_steps(report)):
        result = _step_result(step)
        plan = step.get("plan") if isinstance(step.get("plan"), dict) else step
        trace.append(
            {
                "step_index": index,
                "thought": plan.get("thought"),
                "primitive": _primitive_name(step),
                "args": plan.get("args"),
                "primitive_success": result.get("success"),
                "task_completed": result.get("task_completed"),
                "task_success": result.get("task_success"),
                "message": result.get("message"),
            }
        )
    return trace


def _build_trial_record(
    *,
    task: str,
    trial_index: int,
    return_code: int,
    trial_dir: Path,
    report_path: Path | None,
) -> TrialRecord:
    raw_report: Any = {}
    report: dict[str, Any] = {}
    if report_path is not None:
        raw_report = _read_json(report_path)
        report = _extract_task_report(raw_report, task)

    metrics = _selected_metrics(task, report)
    success = _report_success(report, return_code)
    failed_step = None if success else _failed_step(report)
    final_task_success = metrics.get("task_success")
    subtask_progress = metrics.get("subtask_progress")

    common_log = {
        "task": task,
        "trial_index": trial_index,
        "success": success,
        "return_code": return_code,
        "trial_dir": str(trial_dir),
        "report_path": str(report_path) if report_path else None,
        "final_task_success": final_task_success,
        "subtask_progress": subtask_progress,
        "metrics": metrics,
        "planner_type": "deterministic_agentic_primitive_plan",
        "planning_trace": _planning_trace(report),
    }

    success_log_path: Path | None = None
    failure_log_path: Path | None = None
    if success:
        success_log_path = trial_dir / "success_trajectory_log.json"
        _write_json(success_log_path, common_log)
    else:
        failure_log_path = trial_dir / "failure_case.json"
        failure_payload = dict(common_log)
        failure_payload["failed_step"] = failed_step
        failure_payload["stdout_path"] = str(trial_dir / "stdout.txt")
        failure_payload["stderr_path"] = str(trial_dir / "stderr.txt")
        _write_json(failure_log_path, failure_payload)

    return TrialRecord(
        task=task,
        trial_index=trial_index,
        success=success,
        return_code=return_code,
        trial_dir=trial_dir,
        report_path=report_path,
        final_task_success=final_task_success,
        subtask_progress=subtask_progress,
        metrics=metrics,
        failed_step=failed_step,
        success_log_path=success_log_path,
        failure_log_path=failure_log_path,
    )


def _numeric_values(records: list[TrialRecord], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for record in records:
        value: Any = record.metrics
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        number = _to_float(value)
        if number is not None:
            values.append(number)
    return values


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _aggregate_task(task: str, records: list[TrialRecord]) -> dict[str, Any]:
    successes = sum(1 for record in records if record.success)
    trials = len(records)
    raw_metric_keys = sorted(
        {
            key
            for record in records
            for key in record.metrics.get("raw_numeric_metrics", {}).keys()
        }
    )
    raw_metric_means = {}
    for key in raw_metric_keys:
        values = [
            value
            for record in records
            if (value := _to_float(record.metrics.get("raw_numeric_metrics", {}).get(key)))
            is not None
        ]
        if values:
            raw_metric_means[key] = mean(values)

    return {
        "task": task,
        "trials": trials,
        "successes": successes,
        "success_rate": successes / trials if trials else 0.0,
        "mean_task_success": _mean_or_none(_numeric_values(records, ("task_success",))),
        "mean_subtask_progress": _mean_or_none(
            _numeric_values(records, ("subtask_progress",))
        ),
        "mean_collision_count": _mean_or_none(_numeric_values(records, ("collision_count",))),
        "mean_env_collision_count": _mean_or_none(
            _numeric_values(records, ("env_collision_count",))
        ),
        "mean_self_collision_count": _mean_or_none(
            _numeric_values(records, ("self_collision_count",))
        ),
        "mean_slip_count": _mean_or_none(_numeric_values(records, ("slip_count",))),
        "mean_path_length": _mean_or_none(_numeric_values(records, ("path_length",))),
        "mean_bimanual_arm_velocity_difference": _mean_or_none(
            _numeric_values(records, ("bimanual_coordination", "arm_velocity_difference"))
        ),
        "mean_bimanual_gripper_vertical_difference": _mean_or_none(
            _numeric_values(records, ("bimanual_coordination", "gripper_vertical_difference"))
        ),
        "raw_metric_means": raw_metric_means,
        "failure_count": trials - successes,
        "failure_logs": [
            str(record.failure_log_path)
            for record in records
            if record.failure_log_path is not None
        ],
        "success_logs": [
            str(record.success_log_path)
            for record in records
            if record.success_log_path is not None
        ],
    }


def _write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "trials",
        "successes",
        "success_rate",
        "mean_task_success",
        "mean_subtask_progress",
        "mean_collision_count",
        "mean_env_collision_count",
        "mean_self_collision_count",
        "mean_slip_count",
        "mean_path_length",
        "mean_bimanual_arm_velocity_difference",
        "mean_bimanual_gripper_vertical_difference",
        "failure_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})


def run_evaluation(
    *,
    tasks: list[str],
    trials: int,
    output_dir: Path,
    window: bool,
    timeout_sec: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[TrialRecord] = []
    for task in tasks:
        for trial_index in range(1, trials + 1):
            print(f"[phase6] {task} trial {trial_index}/{trials}", flush=True)
            return_code, trial_dir, report_path = _run_trial(
                task=task,
                trial_index=trial_index,
                output_dir=output_dir,
                window=window,
                timeout_sec=timeout_sec,
            )
            record = _build_trial_record(
                task=task,
                trial_index=trial_index,
                return_code=return_code,
                trial_dir=trial_dir,
                report_path=report_path,
            )
            records.append(record)
            status = "success" if record.success else "failure"
            print(
                f"[phase6] {task} trial {trial_index}: {status}, "
                f"task_success={record.final_task_success}",
                flush=True,
            )

    summaries = [
        _aggregate_task(task, [record for record in records if record.task == task])
        for task in tasks
    ]
    failure_cases = [
        {
            "task": record.task,
            "trial_index": record.trial_index,
            "trial_dir": str(record.trial_dir),
            "failure_log_path": str(record.failure_log_path)
            if record.failure_log_path
            else None,
            "final_task_success": record.final_task_success,
            "subtask_progress": record.subtask_progress,
            "failed_step": record.failed_step,
        }
        for record in records
        if not record.success
    ]

    report = {
        "phase": 6,
        "tasks": tasks,
        "trials_per_task": trials,
        "output_dir": str(output_dir),
        "summaries": summaries,
        "failure_cases": failure_cases,
        "trial_records": [
            {
                "task": record.task,
                "trial_index": record.trial_index,
                "success": record.success,
                "return_code": record.return_code,
                "trial_dir": str(record.trial_dir),
                "report_path": str(record.report_path) if record.report_path else None,
                "final_task_success": record.final_task_success,
                "subtask_progress": record.subtask_progress,
                "metrics": record.metrics,
                "success_log_path": str(record.success_log_path)
                if record.success_log_path
                else None,
                "failure_log_path": str(record.failure_log_path)
                if record.failure_log_path
                else None,
            }
            for record in records
        ],
    }

    _write_json(output_dir / "phase6_eval_report.json", report)
    _write_json(output_dir / "failure_cases.json", failure_cases)
    _write_csv(output_dir / "phase6_eval_summary.csv", summaries)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated Phase 6 RoboEval task-solver evaluations."
    )
    parser.add_argument(
        "--task",
        choices=("all", *TASKS),
        default="all",
        help="Task to evaluate. Default: all.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per task. Default: 10.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "phase6_eval",
        help="Directory for per-trial logs and aggregate reports.",
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="Open the RoboEval viewer during each trial.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=180,
        help="Timeout for each trial subprocess. Default: 180.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")

    report = run_evaluation(
        tasks=_task_selection(args.task),
        trials=args.trials,
        output_dir=args.output_dir,
        window=args.window,
        timeout_sec=args.timeout_sec,
    )
    print("\nPhase 6 summary")
    for summary in report["summaries"]:
        print(
            f"- {summary['task']}: {summary['successes']}/{summary['trials']} "
            f"success_rate={summary['success_rate']:.2f}, "
            f"mean_task_success={summary['mean_task_success']}"
        )
    print(f"saved_report: {args.output_dir / 'phase6_eval_report.json'}")
    print(f"saved_summary_csv: {args.output_dir / 'phase6_eval_summary.csv'}")
    print(f"saved_failure_cases: {args.output_dir / 'failure_cases.json'}")


if __name__ == "__main__":
    main()


