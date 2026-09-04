"""Run one immutable historical v1 trial and normalize its report."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import runpy
import subprocess
import sys


BASELINE_COMMITS = {
    "v1-p22-independent": "d9198fa",
    "v1-p23-memory": "337d0f0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(BASELINE_COMMITS), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("cube_handover", "lift_pot", "stack_two_blocks"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--memory-file", type=Path)
    parser.add_argument("--record-gif", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def checked_commit(root: Path, expected: str) -> str:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not commit.startswith(expected):
        raise RuntimeError(
            f"{root} is at {commit}; {expected} is required for this baseline"
        )
    return commit


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    expected = BASELINE_COMMITS[args.method]
    commit = checked_commit(source_root, expected)
    script = source_root / "examples" / "run_phase7_llm_agent.py"
    if not script.exists():
        raise FileNotFoundError(script)

    sys.path.insert(0, str(source_root))
    from roboeval.roboeval_env import RoboEvalEnv

    original_reset = RoboEvalEnv.reset

    def reset_with_requested_seed(self, *reset_args, **reset_kwargs):
        if reset_kwargs.get("seed") is None:
            reset_kwargs["seed"] = args.seed
        return original_reset(self, *reset_args, **reset_kwargs)

    RoboEvalEnv.reset = reset_with_requested_seed
    namespace = runpy.run_path(str(script), run_name="immutable_v1_baseline")
    legacy_root = args.output_dir / "legacy"
    legacy_args = argparse.Namespace(
        task=args.task,
        provider=args.provider,
        model=args.model,
        trials=1,
        max_steps=args.max_steps,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        window=False,
        record_gif=args.record_gif,
        gif_every=15,
        gif_duration=0.12,
        output_dir=legacy_root,
    )
    run_one = namespace["run_one"]
    kwargs = {
        "task_key": args.task,
        "trial_index": 1,
        "args": legacy_args,
    }
    memory_notes = []
    if args.memory_file and args.memory_file.exists():
        memory_notes = json.loads(args.memory_file.read_text(encoding="utf-8"))
    if "prior_trial_notes" in inspect.signature(run_one).parameters:
        kwargs["prior_trial_notes"] = memory_notes
    record = run_one(**kwargs)

    final_metrics = dict(record.get("final_metrics", {}))
    raw_metrics = dict(final_metrics.get("raw_metrics") or {})
    for key, value in final_metrics.items():
        if key != "raw_metrics" and value is not None:
            raw_metrics.setdefault(key, value)
    benchmark_success = float(
        record.get("final_task_success", raw_metrics.get("task_success", 0.0))
        or 0.0
    )
    normalized = {
        "task_key": args.task,
        "seed": args.seed,
        "method": args.method,
        "benchmark_success": benchmark_success,
        "subtask_progress": float(raw_metrics.get("subtask_progress", 0.0) or 0.0),
        "behavior_quality": record.get("quality_assessment", {"passed": False}),
        "skill_results": [],
        "failure_code": None if benchmark_success >= 1.0 else "POSTCONDITION_FAILED",
        "metrics": {
            **raw_metrics,
            "llm_calls": record.get("steps"),
            "llm_planning_seconds": None,
            "llm_cost_usd": None,
            "replan_count": None,
        },
        "artifacts": {
            "legacy_report": record.get("report_path"),
            "legacy_case_log": record.get("case_log_path"),
            "trajectory_gif": record.get("trajectory_gif"),
        },
        "metadata": {
            "baseline_commit": commit,
            "immutable_source_root": str(source_root),
            "legacy_record": record,
            "memory_note": record.get("memory_note"),
            "unavailable_metrics": [
                "token_usage",
                "llm_latency",
                "llm_cost",
            ],
        },
    }
    write_json(args.output_dir / "trial_report.json", normalized)
    write_json(
        args.output_dir / "run_config.json",
        {
            "task": args.task,
            "seed": args.seed,
            "method": args.method,
            "provider": args.provider,
            "model": args.model,
            "baseline_commit": commit,
        },
    )
    print(
        json.dumps(
            {
                "method": args.method,
                "task": args.task,
                "seed": args.seed,
                "benchmark_success": benchmark_success,
                "report": str((args.output_dir / "trial_report.json").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

