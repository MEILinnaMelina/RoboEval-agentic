"""Run a closed-loop LLM planner that can only call RoboEval primitives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from absl import logging as absl_logging

from roboeval.agentic.llm_agent import LLMAgent, build_task_prompt, make_planner, summarize_env_state
from roboeval.agentic.primitives import PrimitiveController
from roboeval.agentic.state import collect_env_state
from roboeval.agentic.task_specs import TASK_SPECS, make_task_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_SPECS, default="cube_handover")
    parser.add_argument("--provider", choices=["mock", "openai", "anthropic"], default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "llm_agent")
    return parser.parse_args()


def save_frame(env, path: Path) -> None:
    obs = env.get_observation()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.moveaxis(obs["rgb_external"], 0, -1))


def compact_console_summary(report: dict) -> dict:
    steps = []
    for item in report["steps"]:
        result = item["result"]
        steps.append(
            {
                "index": item["index"],
                "primitive": item["action"]["primitive"],
                "args": item["action"]["args"],
                "success": result["success"],
                "message": result["message"],
                "task_success": result["task_success"],
                "distances": result["distances"],
                "collisions": result["collisions"],
                "feedback": item["feedback"],
            }
        )
    return {
        "task_key": report["task_key"],
        "provider": report["provider"],
        "model": report["model"],
        "completed": report["completed"],
        "final_task_success": report["final_task_success"],
        "steps": steps,
    }


def main() -> None:
    absl_logging.set_verbosity(absl_logging.ERROR)
    args = parse_args()
    env = make_task_env(
        args.task,
        render_mode="human" if args.window else "rgb_array",
        ee=True,
        include_camera=True,
    )

    try:
        env.reset()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_frame(env, args.output_dir / f"{args.task}_before.png")

        controller = PrimitiveController(env, render=args.window, sleep_s=0.02 if args.window else 0.0)
        planner = make_planner(args.provider, args.task, args.model)
        agent = LLMAgent(
            args.task,
            env,
            controller,
            planner,
            execute_primitives=not args.dry_run,
        )

        if args.print_prompt:
            state = collect_env_state(env)
            summary = summarize_env_state(state, TASK_SPECS[args.task])
            system_prompt, user_prompt = build_task_prompt(TASK_SPECS[args.task], summary)
            print("SYSTEM PROMPT:")
            print(system_prompt)
            print("\nUSER PROMPT:")
            print(user_prompt)

        result = agent.run(max_steps=args.max_steps)
        save_frame(env, args.output_dir / f"{args.task}_after.png")

        report = result.to_dict()
        report_path = args.output_dir / f"{args.task}_{args.provider}_agent_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(json.dumps(compact_console_summary(report), indent=2))
        print(f"saved_report: {report_path.resolve()}")
        print(f"saved_images: {args.output_dir.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
