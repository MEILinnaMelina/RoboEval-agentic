# Formal experiment results

This directory holds small, git-tracked summaries of formal `agentic_v2`
experiment runs - nothing else belongs here. Raw per-trial artifacts
(`planning_trace.json`, per-skill JSON, `trials.jsonl`, GIFs, frames) stay
in local `outputs/` (gitignored) and are never committed - a single
`trials.jsonl` from one 10-seed run is already ~2.4 MB, and per-trial
traces multiply that by every task/method/seed combination.

## What goes in a results file

One JSON file per formal run (or comparison), containing only the
aggregate-level fields already produced by
`roboeval/agentic_v2/evaluation.py`'s `aggregate_reports()`:

- `git_commit` (+ whether the tree was dirty)
- `method`, `model`, `task_key`, `seeds`
- `success_rate`, `success_ci95_low/high`, `mean_subtask_progress`
- `terminal_failures` (failure code histogram)
- `mean_env_collision_count`, `mean_self_collision_count`, `mean_slip_count`
- `mean_replans`, `mean_llm_calls`, `mean_llm_cost_usd` (if applicable)

Not raw state dumps, not prompts/responses, not per-step traces. If you need
those to debug a specific run, they're in your local `outputs/` - point
someone at the run's `output_dir` instead of committing it here.

## Naming convention

`{method}_{commit_short}_seeds{range}[_{date}].json`, e.g.:

```
results/v2-fixed_23a6acd_seeds0-9_20260904.json
results/openai-full_23a6acd_seeds0-9_20260910.json
```

Not `v2`/`v3`/`v4` - the commit SHA is what actually disambiguates one run's
results from another's; a version-number suffix stops meaning anything once
there have been several rounds of fixes. See `docs/agentic_v2_status.md` for
the single source of truth on what's currently verified.

Formal runs recorded so far: `openai-full_42eb33b_seeds0-9_20260905.json`
(first three base tasks, 30/30) and `openai-full_b93f1bf_seeds0-9_20260905.json` (six new base tasks, 60/60).
