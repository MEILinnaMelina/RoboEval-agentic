# Agentic v2 - Current Verified Status

Single source of truth for what is actually verified to work right now. If
this file disagrees with `agentic_v2_change_log.md`, `agentic_v2_runbook.md`,
or any commit message, **this file wins** - the others describe history,
intent, or a snapshot from an earlier commit, not necessarily current fact.
Update this file whenever a claim below stops being true.

## Environment

Use the `roboeval` conda env, not base Anaconda - the base env is missing
`mujoco`/`dm_control`/`imageio` and its `pytest` hangs at plugin
auto-discovery, which looks like a stuck test run:

```
C:\Users\melin\.conda\envs\roboeval\python.exe
```

`PYTHONPATH=<repo root>` (or run from repo root) is required when invoking
`examples/*.py` directly - `roboeval` is not pip-installed in this env.

## Unit tests

`pytest tests/test_agentic_v2_*.py` in the `roboeval` env: **44/44 passed**
(last verified run, this commit's code).

## Phase 9 deterministic gate (`--planner fixed`, no LLM)

| Task | Result | Coverage |
|---|---|---|
| `lift_pot` | 10/10 `benchmark_success` | full seeds 0-9, verified at an earlier commit |
| `cube_handover` | not passing | single-seed (seed 0) spot check only, after the latest `handover.py` fixes |
| `stack_two_blocks` | not passing | single-seed (seed 0) spot check only, after the latest `handover.py` fixes |

`lift_pot`'s grasp path (`kitchenpot` handles) goes through
`CandidateGenerator._pot_handle_candidates`, which the recent aperture/frame
fixes below did not touch, so the 10/10 result should still hold - but it
has not been re-run against the current commit to confirm.

**A full 10-seed rerun of all three tasks against the current commit has not
been done.** The `outputs/agentic_v2_phase9_gate*` directories are gitignored
local artifacts (removed from git tracking - see `git log` for the commit
that last had them) and were generated at earlier, different commits; do not
treat them as representative of the current code. `cube_handover`'s v3 run in
particular recorded `git_commit: b91ba11`, which predates the actual
`agentic_v2` package commit - that mismatch is exactly why those directories
were untracked.

## Known open issues (found, not yet fixed)

- **`cube_handover`**: RoboEval's own `task_stage_reached[2]` ground truth
  confirms the handover transfer does occur during some attempts
  (`behavior_quality.reached_transfer_stage: true`), but `HandoverSkill`'s
  own `set(held_by) == {donor, receiver}` verification still reports
  failure - stricter or more timing-sensitive than the benchmark's own
  check. Root cause not yet identified.
- **`stack_two_blocks`**: the staged-regrasp mechanics (place, release,
  donor clear-out, retry) work end-to-end and eliminated the earlier
  `SELF_COLLISION`, but the final regrasp close still trips
  `ENV_COLLISION` before the fingers even close - the object is very thin
  (4cm) and rests flush on the table, leaving little clearance for a
  generic top-down grasp. Needs either a grasp-height bias or a
  table-clearance-aware candidate for short objects.

## API / LLM runs

**No OpenAI or Anthropic run has ever been executed against this
codebase.** `--planner openai`/`--planner anthropic` and `OnlineReplanner`
are covered by unit tests and static review only, not an empirical run.
Do not run a full API matrix until the two open issues above are resolved
(or at least understood well enough that API failures can be attributed
correctly) - see the Phase 9 rationale in `agentic_v2_plan.md`.

## Experiment naming convention

Formal runs are named `{method}_{commit_short}_seeds{range}[_{date}]`, e.g.
`openai-full_23a6acd_seeds0-9_20260904` - not `v2`/`v3`/`v4`, which stop
meaning anything once there's a fourth revision. The commit SHA is the part
that actually disambiguates results; record it (`git rev-parse HEAD` plus a
dirty-tree flag) in every run's config, which `run_agentic_v2.py` already
does via `environment_metadata()`.

See `results/README.md` for what gets committed vs. stays in local
`outputs/`.
