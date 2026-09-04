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
| `cube_handover` | 0/10 `benchmark_success` | full seeds 0-9 at commit `3cfd536` (hold-patience fix only); the later `stop_condition` removal (`19aaa41`) measurably reaches `dual_verify` more often in spot checks but has not had its own full 10-seed rerun |
| `stack_two_blocks` | 0/10 `benchmark_success` | full seeds 0-9 at commit `abd36e1` (drift + hold-patience fixes); not rerun after the later `stop_condition` fix (`2cee8eb`) |

## Known open issues (found, root-caused; grip-stability fix attempts exhausted for now)

Real bugs found and fixed this round (`84ad0a8`, `3cfd536`, `19aaa41`,
`2cee8eb`): rendezvous height had no table-clearance margin on one pose;
receiver arm settled into incidental table contact while idle; hold/slip
detection had no debounce unlike the tracking/velocity checks; and both
`HandoverSkill` and `GraspSkill` used `is_gripper_holding_object()` (a
pure pad-contact check, not a "gripping firmly" check - see
`roboeval/robots/gripper.py:184`) as an approach stop-condition, which
could stop the arm short of the precisely-computed grasp pose.

What's left, after those fixes:

- **`cube_handover`**: with the stop_condition fix, the real gate run
  showed the receiver reaching `dual_verify` in 10 of 60 candidate
  attempts across 10 seeds (up from near-zero before) - but **all 10**
  failed there with `SLIP_DETECTED left lost cube` (confirmed via the
  monitor's own event message, not inferred). The donor's identical
  aperture/material never once loses grip through a full monitored lift;
  only the receiver's grip does. **Tried and reverted**: switching the
  receiver from its side approach to the donor's proven top-down
  orientation - reproducibly caused `SELF_COLLISION` at the pregrasp
  stand-off on 5/5 seeds (two arms cannot share the airspace directly
  above a 20cm rod from above at once) - worse than the slip it was meant
  to fix, so `handover.py` is back to the side approach (`git checkout`
  after `19aaa41`, confirmed clean). Remaining ideas, not yet tried:
  direct visual inspection of the side-approach grip geometry (a
  screenshot approach was used successfully earlier this session for a
  different collision), or more close/settle time before verification.
- **`stack_two_blocks`**: regrasp reliably reaches `receiver_close` for one
  candidate (others `PATH_BLOCKED`) but still trips `ENV_COLLISION` there;
  not yet root-caused as precisely as the cube_handover slip.

## API / LLM runs

One informal 3-task x 3-seed OpenAI run has been made (commit `4ef7084`,
model `gpt-5.6-terra`, no cost-per-million flags set, so `llm_cost_usd`
was not recorded): `lift_pot` 3/3, `cube_handover` 1/1 valid (2/3 died to
`APIConnectionError`, since fixed with a retry - see `llm_planner.py`),
`stack_two_blocks` 0/3 valid (all 3 died to the same network error). A
rerun of the network-killed trials after the retry fix reached
`max_skills=10` on all 5 without succeeding (`TIMEOUT`), consistent with
the grip-stability and table-clearance issues above.

Explicit decision (2026-09-04): after exhausting the readily-available
grip-stability fixes above without reaching the 8/10 deterministic-gate
bar, proceed with API experiments anyway rather than continue indefinitely
- `lift_pot` is fully reliable and `cube_handover` has already
demonstrated a real success in online API play (seed 0, first informal
run) despite the deterministic gate not passing, so online replanning may
route around some of the marginal-grasp cases the fixed script cannot.
Treat `cube_handover`/`stack_two_blocks` API results accordingly: a low
success rate there is expected to partly reflect the known grip issues
above, not purely LLM planning quality, until those are fixed.

A second, clean formal 3-task x 3-seed run at commit `b8ac7dc` (all 9
trials completed without a connection error this time) confirms the
prediction above rather than contradicting it: `lift_pot` 3/3 (3.0 mean
LLM calls, 1.0 mean replans - straightforward), `cube_handover` 0/3 and
`stack_two_blocks` 0/3, both hitting `max_skills=10` every single trial
(10.0 mean LLM calls, ~7.7 mean replans) with `TIMEOUT`. Every
`cube_handover` trial shows `reached_transfer_stage: true` (the slip
issue, not a planning failure); every `stack_two_blocks` trial plateaus
at `subtask_progress=0.33` ("one block held", never further) consistent
with the regrasp `ENV_COLLISION`. The LLM is visibly working hard to
recover (~7-8 replans/trial) against a pipeline that cannot currently
deliver a successful handover for these two tasks - fixing the grip
issues remains the highest-leverage next step for these two tasks'
success rate, not further prompt/planning changes.

## Experiment naming convention

Formal runs are named `{method}_{commit_short}_seeds{range}[_{date}]`, e.g.
`openai-full_23a6acd_seeds0-9_20260904` - not `v2`/`v3`/`v4`, which stop
meaning anything once there's a fourth revision. The commit SHA is the part
that actually disambiguates results; record it (`git rev-parse HEAD` plus a
dirty-tree flag) in every run's config, which `run_agentic_v2.py` already
does via `environment_metadata()`.

See `results/README.md` for what gets committed vs. stays in local
`outputs/`.
