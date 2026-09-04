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
