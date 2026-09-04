# Agentic v2 Change Log

## 2026-09-04

- Created branch `agentic-v2` from `main` with the approved plan change present.
- Verified baseline commits `d9198fa` and `337d0f0` exist locally.
- Recorded the fixed three-task, seeds `0..9` experiment matrix.
- Confirmed the action vector has 14 arm joints followed by left/right gripper
  commands; command `0` opens and command `1` closes the Panda gripper.
- Confirmed base-task initial geometry using fresh seed-0 environments.
- Reviewed the local ReKep and VoxPoser modules listed in the reference notes.
- Implemented Phase 8 transport, sampled handover, and geometry-derived place
  skills. Handover verifies the receiver before donor release; placement
  verifies release, settling, support contact, and the raw stack condition.
- Added fixed pose-free semantic plans for all three tasks, including an
  explicit handover before the cross-workspace stack placement.
- Added a versioned semantic JSON schema, strict low-level field rejection,
  OpenAI and Anthropic clients, and the online observe-decide-execute-observe
  replanner. Cross-trial memory remains opt-in.
- Added the v2 runner and artifact recorder for per-skill state, accepted and
  rejected IK candidates, checked joint paths, monitor events, prompts,
  responses, token usage, latency, run configuration, final report, and GIFs.
- Added the isolated-process Phase 11 launcher and aggregation for raw success,
  quality, failures, metrics, Wilson intervals, and paired-seed comparisons.
  The IK-only path is explicitly an ablation; the full collision gate remains
  the default.
- Added a seed-forcing, report-normalizing adapter for immutable v1 P22/P23
  worktrees. It verifies the exact baseline commit before execution and keeps
  historical source trees unchanged.
- Added an explicit Phase 12 scope guard. OpenVLA, expanded tasks, learned
  perception, and advanced motion planners remain deferred.
- Verification performed without simulation rollouts: 17 focused tests passed
  in 1.02 seconds. The 10-seed deterministic and API matrices have not yet
  been run and their plan gates remain unchecked.

### Later the same day: bug fixes, Phase 9 gate runs, handover redesign

The above verification was static/mocked only; running the actual test
suite and Phase 9 gate against the `roboeval` conda env surfaced real bugs
this entry didn't catch. Summary only - **`docs/agentic_v2_status.md` is
the current source of truth**, not this log:

- Fixed two crash-on-first-use bugs (missing import in `skills/transport.py`;
  wrong `OnlineReplanner` keyword in `run_agentic_v2.py`) that blocked every
  transport/handover/place call and every non-fixed-planner run respectively.
- Grew the test suite to 44 tests, all passing in the correct conda env.
- Ran the actual Phase 9 fixed-plan gate (not just unit tests): `lift_pot`
  10/10, `cube_handover` and `stack_two_blocks` 0/10 at that point.
- Root-caused and fixed a receiver-side grasp-axis bug, an object-size
  measurement that inflated with tilt (added `canonical_size`), a
  world/body frame mixing bug in the fix for that, missing retreat-after-
  failure recovery, a blanket failure code masking real causes, a
  misleading quality check, and a self-collision from small objects being
  forced through a dual-hold strategy (added staged place-then-regrasp).
  Both tasks progressed substantially (self-collision eliminated, transfer
  stage reached per RoboEval's own ground truth) but neither reliably
  passes yet - see `agentic_v2_status.md` for the two remaining open issues.
- No API/LLM run has been performed at any point.
