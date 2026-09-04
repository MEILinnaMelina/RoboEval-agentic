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
| `lift_pot` | 10/10 `benchmark_success` | full seeds 0-9, verified at an earlier commit; grasp path untouched by any change since |
| `cube_handover` | 0/10 `benchmark_success` | full seeds 0-9 at commit `84ad0a8` (after the rendezvous-height fix) |
| `stack_two_blocks` | not passing | single-seed (seed 0) spot check only, after the receiver-drift fix |

**A full 10-seed rerun of `stack_two_blocks` against the current commit has
not been done** - only `cube_handover` got a full rerun after its latest fix.

## Known open issues (found, root-caused, not yet fixed)

Both of the previous entries here ("HandoverSkill's own held_by check is
stricter than RoboEval's ground truth", "generic top-down grasp lacks
table clearance for short objects") turned out to be imprecise - direct
reproduction (not inference) found the *actual* mechanisms below. Two real
bugs found this way (rendezvous height had no table-clearance margin on
one pose; receiver arm settled into incidental table contact while idle)
are already fixed as of `84ad0a8`. What's left is more fundamental:

- **`cube_handover`**: the rod is thin, and the receiver's grasp point is
  offset toward one end (`tip_margin=0.035`) specifically so it clears the
  donor's hand at the workspace center. Reproduced one case that closed
  successfully and passed `dual_verify`, then `SLIP_DETECTED` during the
  hold-still verification - the grip itself is marginally stable, not a
  detection bug. Across a full 10-seed run, the dominant failure varies
  seed to seed among `GRASP_FAILED` (close didn't register a hold),
  `HELD_OBJECT_COLLISION`, and this slip - consistent with a grip that's
  right at the edge of reliable, not a single deterministic cause. Needs a
  grip-stability fix (firmer close, more dwell before verifying, or a
  grip point traded further from the tip against less donor clearance),
  not another collision-geometry fix.
- **`stack_two_blocks`**: the receiver-drift `ENV_COLLISION` is fixed, but
  the regrasp close itself is still unreliable for this very thin (4cm)
  object resting flush on the table - not yet re-verified across a full
  10-seed run after the drift fix, so the current true pass rate is
  unknown.

## API / LLM runs

One informal 3-task x 3-seed OpenAI run has been made (model
`gpt-5.6-terra`, no `--input-cost-per-million`/`--output-cost-per-million`
set, so `llm_cost_usd` was not recorded). Results before any of the fixes
in this file: `lift_pot` 3/3, `cube_handover` 1/1 valid (2/3 died to
`APIConnectionError`, since fixed with a retry - see `llm_planner.py`),
`stack_two_blocks` 0/3 valid (all 3 died to the same network error). A
rerun of the network-killed trials after the retry fix landed reached
`max_skills=10` on all 5 without succeeding (`TIMEOUT`), consistent with
the grip-stability and table-clearance issues above rather than a new bug -
see `skill_results` sequences in those trial reports for exact failure
codes per step.

**No formal (10-seed) API run has been done.** Do not launch one until the
two open issues above show real improvement on the deterministic gate -
a 10-seed API matrix against a known-marginal grasp will mostly measure
the grasp's flakiness, not the LLM's planning quality, and burns real
budget doing it.

## Experiment naming convention

Formal runs are named `{method}_{commit_short}_seeds{range}[_{date}]`, e.g.
`openai-full_23a6acd_seeds0-9_20260904` - not `v2`/`v3`/`v4`, which stop
meaning anything once there's a fourth revision. The commit SHA is the part
that actually disambiguates results; record it (`git rev-parse HEAD` plus a
dirty-tree flag) in every run's config, which `run_agentic_v2.py` already
does via `environment_metadata()`.

See `results/README.md` for what gets committed vs. stays in local
`outputs/`.
