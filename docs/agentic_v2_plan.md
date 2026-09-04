# RoboEval-Agentic v2 (Agentic-TAMP) - Implementation Plan

## Goal And Scope

Build `roboeval/agentic_v2/` for the three base tasks: `lift_pot`,
`cube_handover`, and `stack_two_blocks`.

The LLM selects **what to do**: a semantic skill, object, arm roles, and goal.
Deterministic robotics code decides **how to do it**: candidate generation,
IK, collision checking, path planning, execution, monitoring, and verification.
The LLM must never emit joint values, Cartesian offsets, yaw, controller
gains, step counts, or unchecked trajectories.

This milestone covers the three base tasks only. Position/orientation
variants, long-horizon tasks, vision-only perception, PDDLStream, and GPU
motion planners are deferred.

## Non-Negotiable Rules

- [ ] Keep `roboeval/agentic/` (v1) frozen; historical commits are baselines.
- [ ] Put all new behavior in `roboeval/agentic_v2/` and a v2 entry point.
- [ ] Use `JointPositionActionMode(ee=False, absolute=True)` in v2. Plan in
      joint space and execute bounded trajectory chunks.
- [ ] Never advance or mutate live MuJoCo state during planning.
- [ ] Check both arms, held objects, and skill-specific allowed contacts at
      every sampled trajectory state.
- [ ] Never close after a failed approach or retry without checked recovery.
- [ ] Keep raw RoboEval success separate from behavior-quality judgments.
- [ ] Disable cross-trial memory in the primary method; test it only as a
      named ablation.

## Phase 0 - Freeze Baselines And Create The Scaffold

- [ ] Verify and record immutable v1 references in the experiment manifest:
      `337d0f0` (P23 endpoint) and `d9198fa` (pre-memory P22).
- [ ] Run each v1 baseline in an independent process so planner history and
      cross-trial memory cannot leak between methods.
- [ ] Create a dedicated v2 branch before implementation.
- [ ] Create the minimal package layout:
      `agentic_v2/{types,state,evaluator,executor,monitor}.py`,
      `agentic_v2/motion/`, `agentic_v2/constraints/`, and
      `agentic_v2/skills/`.
- [ ] Create `examples/run_agentic_v2.py` with `--task`, `--seed`, `--render`,
      `--planner`, `--memory`, and `--output-dir` options.
- [ ] Fix primary evaluation seeds to integers `0..9` for every task. Save the
      task, seed, git commit, config, package versions, and model with a trial.
- [ ] Keep ReKep and VoxPoser outside the package and submission code. They are
      local references, not runtime dependencies.

**Exit gate:** v1 references and the 30-trial seed matrix are recorded, the v2
entry point imports, and no v1 source file has changed.

## Phase 1 - Typed Contracts, State, And Evaluation Semantics

New modules: `agentic_v2/types.py`, `state.py`, and `evaluator.py`.

- [ ] Define a `FailureCode` enum including `IK_UNREACHABLE`, `JOINT_LIMIT`,
      `SELF_COLLISION`, `ENV_COLLISION`, `HELD_OBJECT_COLLISION`,
      `OTHER_OBJECT_COLLISION`, `NO_VALID_GRASP`, `GRIPPER_APERTURE_MISMATCH`,
      `HANDOVER_REGION_EMPTY`, `PLACEMENT_UNREACHABLE`, `RELEASE_FAILED`,
      `PATH_BLOCKED`, `APPROACH_FAILED`, `GRASP_FAILED`, `OBJECT_DISPLACED`,
      `SLIP_DETECTED`, `CONSTRAINT_VIOLATION`, `EXECUTION_DIVERGED`, and
      `TIMEOUT`. Distinguish `HELD_OBJECT_COLLISION` (the object currently
      grasped) from `OTHER_OBJECT_COLLISION` (an unrelated movable object
      gets bumped, e.g. during transport in `stack_two_blocks`).
- [ ] Define typed `Pose`, `RobotState`, `ObjectState`, `SceneState`,
      `SkillRequest`, `AllowedContactPolicy`, `ConstraintSet`, `IKCandidate`,
      `MotionCandidate`, `FeasibilityReport`, `MotionPlan`, `ExecutionReport`,
      and `TrialReport` records.
- [ ] Adapt v1 state-reading logic without changing v1. Include both arms'
      joint state, end-effector pose, gripper aperture, and inferred holding;
      object pose, orientation, dimensions/AABB, velocity, and contacts; task
      stage; collision/slip counters; and raw RoboEval metrics.
- [ ] Treat an AABB as a current-state measurement only. Never reuse a static
      world-frame AABB for a transported object.
- [ ] Infer handover donor/receiver roles from current grasp state; do not
      permanently assign either arm as donor.
- [ ] Serialize JSON with separate top-level fields `benchmark_success`,
      `subtask_progress`, and `behavior_quality`.
- [ ] Test JSON round trips, enum validation, malformed requests, missing
      objects, and left/right role validation.

**Exit gate:** semantic requests validate, a live state round-trips through
JSON, and behavior quality cannot overwrite raw benchmark success.

## Phase 2 - Side-Effect-Free Collision And Contact Checking

New module: `agentic_v2/motion/collision_checker.py`.

- [ ] Create an independent `MjData` clone for planning. Map a candidate's arm
      joints into full `qpos`, then run the complete position update needed by
      this model before `mujoco.mj_collision()`; never call `mj_step()`.
- [ ] Add a regression test proving the planning data shares no writable state
      with live data and leaves live `qpos`, `qvel`, controls, time, contacts,
      object poses, and task counters byte-for-byte unchanged.
- [ ] Reuse verified robot/scene geom ownership from
      `roboeval/utils/metric_rollout.py`. Return typed contacts with geom names,
      owners, signed distance/penetration, and classification.
- [ ] Distinguish self collision, robot-environment collision, held-object
      collision, and allowed task contact.
- [ ] Require every skill to provide an `AllowedContactPolicy`. Examples:
      fingers-target during grasp, pot-table before lift, upper-lower block
      during place, and receiver-cube during handover. All other contact stays
      forbidden unless explicitly justified.
- [ ] While holding, propagate object pose at every candidate state using
      `T_world_object = T_world_ee @ T_ee_object`, then recompute its geometry
      before collision checking. Do not translate a stale AABB.
- [ ] Test at least 10 in-limit known-safe poses, 10 in-limit known-colliding
      poses, allowed-contact cases, forbidden-contact cases, and joint-limit
      rejection. Keep the extreme folded pose only as an additional stress
      case if it lies outside valid limits.

**Exit gate:** repeated checks are deterministic, detect the expected contact
class, and cannot alter a live episode.

## Phase 3 - Multi-Start IK Feasibility

New module: `agentic_v2/motion/ik.py`.

- [ ] Wrap the existing RoboEval IK without changing v1. Do not accept its
      returned `qpos` alone: retain convergence status, iteration count,
      position error, orientation error, seed, and termination reason.
- [ ] Generate deterministic multi-start seeds from current state, neutral
      state, and bounded perturbations. Deduplicate near-identical solutions.
- [ ] Reject non-converged solutions and candidates outside the configured
      joint limits before collision checking.
- [ ] Rank surviving candidates by pose error, distance from current joints,
      joint-limit margin, and collision clearance.
- [ ] For dual-arm goals, construct and validate combined configurations;
      independent left/right IK success is not sufficient.
- [ ] Return structured `IK_UNREACHABLE`, `JOINT_LIMIT`, `SELF_COLLISION`, or
      `ENV_COLLISION` evidence when no candidate survives.
- [ ] Test reachable, unreachable, near-limit, obstacle, and paired-arm targets
      with deterministic expected outcomes.

**Exit gate:** every accepted candidate is converged, in limits, collision
checked, reproducible, and carries enough diagnostics to explain rejection.

## Phase 4 - Collision-Aware Joint-Space Path Planning

New module: `agentic_v2/motion/path_planner.py`.

- [ ] Commit to `JointPositionActionMode(ee=False, absolute=True)` for v2.
- [ ] Interpolate `q_start -> q_goal` with sample count determined by maximum
      per-joint displacement, not by one fixed global `N`.
- [ ] At every sample check limits, combined-arm collision, allowed-contact
      policy, and propagated held-object geometry.
- [ ] Rank clear paths by length, clearance, joint-limit margin, and bimanual
      synchronization cost.
- [ ] If a straight path is blocked, try alternative IK goals first, then a
      small set of named joint-space recovery waypoints. Every segment must
      pass the same checks.
- [ ] Time-parameterize the selected path with bounded joint velocity and
      acceleration. The controller must not infer timing from an LLM value.
- [ ] Return `PATH_BLOCKED` with the blocking sample and contacts when all
      candidates fail.
- [ ] Defer RRT, CHOMP, cuRobo, and trajectory optimization until this simple
      planner passes the deterministic end-to-end gate.

**Exit gate:** standalone plans are reproducible, entirely prechecked, and
obey configured joint displacement and timing bounds.

## Phase 5 - Executor, Monitor, And Checked Recovery

New modules: `agentic_v2/executor.py` and `monitor.py`.

- [ ] Add one explicit adapter from planned full joint states to RoboEval's
      absolute joint-position action vector; test joint and gripper indexing.
- [ ] Execute paths in short bounded chunks. Re-observe between chunks and
      stop before issuing more actions when an invariant fails.
- [ ] Monitor tracking error, collisions, allowed contacts, grasp state,
      object displacement, EE-object relative transform, slip, task stage,
      termination, and truncation.
- [ ] Treat raw RoboEval success as an observed metric, not as permission to
      hide unsafe or visibly failed behavior.
- [ ] Invalidate cached plans and settle assumptions after every grasp,
      release, unexpected contact, object displacement, or recovery.
- [ ] Generalize v1 P19/P22 behavior into typed monitor events rather than
      copying task-specific breakers.
- [ ] Plan and collision-check every retreat. If no safe retreat exists, stop
      with a structured failure instead of improvising Cartesian motion.
- [ ] Produce an `ExecutionReport` containing executed samples, monitor events,
      final state, failure code, benchmark metrics, and artifact paths.
- [ ] Test interruption on tracking divergence, forbidden contact, displaced
      object, lost grasp, and truncation.

**Exit gate:** a failed approach cannot close the gripper, a failed grasp
cannot continue into transport, and bounded execution does not develop an
uninterrupted oscillation.

## Phase 6 - Grasp Candidates And The Grasp Skill

New modules: `agentic_v2/motion/candidate_generator.py` and
`agentic_v2/skills/grasp.py`.

- [ ] Generate multiple pregrasp/grasp candidates with pose, approach axis,
      required aperture, target contact set, and score.
- [ ] Derive cube, rod, and block candidates from current object geometry.
      Represent pot handles as measured environment affordances, not scripted
      trajectories or guessed world offsets.
- [ ] Include multiple valid approach orientations; do not assume one fixed
      wrist yaw works across seeds.
- [ ] Filter candidates by aperture, IK, joint limits, collision/contact policy,
      and complete pregrasp-to-grasp path before execution.
- [ ] Implement `Grasp` as preconditions -> candidate generation -> plan ->
      open -> approach -> close -> verify hold -> checked retreat/postcondition.
- [ ] Never close after a failed approach and never report success from finger
      contact alone. Verification must include stable object-relative motion
      over a short monitored lift or retreat.
- [ ] Validate each object class with controlled reachable, blocked, too-wide,
      and slip cases before running full tasks.

**Exit gate:** the skill either returns a verified stable hold or stops with a
specific reason and a valid recovery state.

## Phase 7 - Bimanual Constraints, BimanualGrasp, And Lift

New modules: `agentic_v2/constraints/bimanual.py`,
`skills/bimanual_grasp.py`, and `skills/lift.py`.

- [ ] Solve paired targets into one combined joint configuration and validate
      dual-arm self collision, environment collision, and object geometry.
- [ ] Capture both `T_ee_object` transforms after verified grasps. Preserve
      them within explicit translation/orientation tolerances during lift.
- [ ] Synchronize paired paths by time, not by blindly matching array indices;
      enforce bounded arm velocity difference and object tilt.
- [ ] Implement deterministic `BimanualGrasp` and `Lift` skills with typed
      preconditions, postconditions, and monitor invariants.
- [ ] Test asymmetric reach, one-arm grasp loss, conflicting paired IK,
      inter-arm collision, object tilt, and successful synchronized lift.

**Exit gate:** both grasps remain verified through a controlled lift, or both
arms stop safely with the failing constraint identified.

## Phase 8 - Transport, Handover, And Place Skills

New modules: `agentic_v2/skills/{transport,handover,place}.py`.

- [x] Transport held objects with propagated geometry and preserved grasp
      transforms at every path sample.
- [x] Replace the v1 handover midpoint with sampled rendezvous candidates.
      Check donor IK, receiver IK, combined-arm collision, held-object sweep,
      receiver grasp compatibility, and both approach paths.
- [x] Close and verify the receiver before releasing the donor. If verification
      fails, preserve the donor hold and return a replannable failure state.
- [x] Generate placement candidates from support geometry, object dimensions,
      target contact policy, clearance, and predicted static stability.
- [x] Verify placement only after release and settling. For stacking, require
      the upper block to contact the lower block, not the table, and require
      both grippers to release.
- [x] Return structured failures using the `FailureCode` enum: empty
      rendezvous regions as `HANDOVER_REGION_EMPTY`, blocked transport as
      `PATH_BLOCKED`, incompatible grasps as `NO_VALID_GRASP`, unstable or
      unreachable placement as `PLACEMENT_UNREACHABLE`, and a failed
      release as `RELEASE_FAILED`.
- [ ] Explicitly test the approximately 0.7 m transfer in `stack_two_blocks`
      instead of assuming one arm can carry directly across the workspace.

**Exit gate:** controlled tests can transport, transfer, and place objects
without stale geometry, premature donor release, or false placement success.

## Phase 9 - Deterministic End-To-End Gate (No LLM)

- [x] Build fixed semantic skill sequences for all three tasks. These sequences
      may choose skills and named objects, but may not contain poses, offsets,
      joint values, timing, or seed-specific branches.
- [ ] Validate in this order: `cube_handover`, `lift_pot`, then
      `stack_two_blocks`.
- [ ] Run each sequence on seeds `0..9` in fresh independent processes using
      the same settings intended for the LLM evaluation.
- [ ] Require at least 8/10 raw RoboEval successes for **each** task before
      connecting GPT or Claude. Behavior-quality failures remain visible even
      when the benchmark metric reports success.
- [x] Save per-step state, selected candidates, rejected candidates, paths,
      monitor events, metrics, final report, and both success/failure GIFs.
- [ ] Fix deterministic motion or skill failures here. Do not ask an LLM to
      compensate for an unreliable low-level stack.

**Exit gate:** all three fixed semantic plans reach at least 8/10 raw success
and failures are attributable from structured artifacts.

## Phase 10 - Attach The Online LLM Planner

New modules: `agentic_v2/llm_planner.py`, `prompts.py`, and `replanner.py`.

- [x] Expose only a versioned semantic JSON schema: skill name, object,
      left/right roles, symbolic goal, and optional strategy choice.
- [x] Reject unknown skills and all low-level fields, including joint values,
      poses, offsets, yaw, controller gains, step counts, and tolerances.
- [x] Give the model the task goal, raw success condition, current typed state,
      available skills, skill preconditions, prior action result, and compact
      structured failure evidence.
- [x] Compile every accepted semantic request through the same deterministic
      candidate -> feasibility -> path -> execute -> verify pipeline from
      Phases 2-8.
- [x] Re-observe after each skill and ask the LLM for the next semantic action.
      Replanning may change strategy, but cannot bypass safety checks.
- [x] Keep cross-trial memory off by default. Within-trial history is allowed
      and must be logged. Add memory later only as a named ablation.
- [x] Log prompts, model responses, validated requests, token usage, latency,
      failure codes, and replanning decisions without logging API keys.

**Exit gate:** GPT/Claude acts only through semantic skills in a genuinely
online observe-decide-execute-observe loop.

## Phase 11 - Evaluation, Ablations, And Artifacts

- [ ] Run every method on tasks `lift_pot`, `cube_handover`, and
      `stack_two_blocks`, seeds `0..9`, with identical simulator settings.
- [x] Start every trial in a fresh environment and every method in an
      independent process. Do not carry messages, summaries, or memory between
      trials unless the row is explicitly `+Memory`.
- [x] Report raw `benchmark_success` and `subtask_progress` separately from
      `behavior_quality`; never rename a quality pass as task success.
- [x] Record collision counts by class, slip count, path length, execution
      time, planning time, LLM calls/tokens/cost, replan count, terminal
      failure code, and bimanual coordination metrics.
- [x] Save machine-readable JSON/JSONL, aggregate CSV, run config, prompt and
      response logs, trajectory data, and representative success/failure GIFs.
- [x] Include confidence intervals and per-seed paired comparisons where the
      same seeds are used; do not present only one aggregate percentage.

Required comparison rows:

| Method | Feasibility gate | Semantic LLM | Online replan | Cross-trial memory |
| --- | --- | --- | --- | --- |
| v1-P22 independent | none | yes | text feedback only | off |
| v1-P23 + Memory | none | yes | text feedback only | on |
| v2-IK-only | IK only | yes | yes | off |
| v2-Fixed | full | no, fixed semantic plan | no | off |
| v2-Full-no-replan | full | yes | no | off |
| v2-Full | full | yes | yes | off |
| v2-Full + Memory | full | yes | yes | on |

The primary claim must compare `v1-P22 independent` with `v2-Full`.
`v2-Fixed` measures the deterministic robotics layer, and `+Memory` is an
optional ablation rather than part of the main method.

**Exit gate:** all rows are reproducible from saved manifests and reports, and
the result table can be regenerated without manually relabeling outcomes.

## Phase 12 - Explicitly Deferred Work

Do not start these items until the base-task v2 result table is complete:

- [ ] Position and orientation task variants.
- [ ] Long-horizon RoboEval tasks beyond the selected three.
- [ ] Vision-only scene parsing or learned perception.
- [ ] RRT, CHOMP, ReKep-style trajectory optimization, cuRobo, or an Isaac
      migration. cuRobo is especially a later Linux/CUDA integration.
- [ ] PDDLStream or another symbolic task planner.
- [ ] Learned grasp policies, OpenVLA, or direct low-level model control.

## Reference-Code Policy

- **ReKep** is available at `E:\djf\ReKep-main`. Read its constraint/subgoal/
  path decomposition for architectural ideas, but do not claim its modules map
  one-to-one onto RoboEval. Its local `ik_solver.py` uses Lula and its
  `path_solver.py` optimizes end-effector control points against an SDF; both
  differ from this MuJoCo joint-space implementation.
- No license file was visible in the downloaded ReKep tree during review.
  Therefore use ideas and independently implement them; do not copy source
  code into RoboEval unless licensing is clarified.
- **VoxPoser** is available at `E:\djf\VoxPoser-main`. Use its separation of
  LLM interface, planner, controller, and environment as a design reference.
  Its code is MIT-licensed; preserve the license and attribution if any code is
  actually adapted.
- Do not vendor either repository or make it a runtime dependency. Record any
  adapted algorithm and source in a dedicated attribution note.
- Treat VLM-TAMP, PDDLStream, cuRobo, and LABOR-Agent as reading material for
  later scope, not as dependencies for the base-task milestone.

## Definition Of Done

The base-task v2 milestone is complete only when all items below are true:

- [ ] v1 remains unchanged and both baseline commits are reproducible.
- [ ] A semantic `SkillRequest` can traverse candidate generation, IK,
      collision checking, planning, bounded execution, monitoring, and
      postcondition verification without an LLM.
- [ ] Planning tests prove there is no live MuJoCo mutation.
- [ ] Every trajectory sample checks the combined robot, propagated held
      objects, and a skill-specific allowed-contact policy.
- [ ] Failed approach/grasp/transfer/place operations stop or run a checked
      recovery; they cannot silently continue to the next skill.
- [ ] Fixed semantic plans obtain at least 8/10 raw successes on every selected
      task across seeds `0..9` before API evaluation begins.
- [ ] GPT or Claude performs online observation and semantic replanning, never
      low-level control, and every accepted request passes the same gate.
- [ ] The complete evaluation matrix reports raw benchmark success separately
      from quality metrics and includes logs, trajectories, configs, costs,
      representative GIFs, and attributable failure codes.

Completing only the API call, a visually plausible rollout, or a
deterministic script does not by itself satisfy the professor's agentic-task
objective.
