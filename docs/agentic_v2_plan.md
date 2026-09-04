# RoboEval-Agentic v2 (Agentic-TAMP) — Implementation Plan

Context: `roboeval/agentic/` (v1) spent an entire session (P1-P23, see
`docs/phase8_success_rate_debug_log.md`) hand-tuning heuristics for a
fundamentally mismatched architecture — asking an LLM to output continuous
geometric parameters (`ee_offset`, `yaw`, `steps`) it has no reliable way to
get right, then reactively patching each new failure mode. User proposed
restarting on a TAMP-style architecture instead: **LLM decides *what*
(semantic skill + target), a classical robotics layer decides *how*
(feasibility + motion)** — matching SayCan / VLM-TAMP / ReKep. v1 stays
untouched as a baseline; this is a new `roboeval/agentic_v2/` package.

**Feasibility of the riskiest primitive is now confirmed, not assumed.**
Verified directly against this repo's actual MuJoCo/mojo stack:
`copy.copy(mojo.data)` gives a fully independent `MjData` snapshot;
mutating its `qpos` and calling `mujoco.mj_kinematics()` +
`mujoco.mj_collision()` (kinematics + contact only, no `mj_step`, no time
advance) gives exact per-geom contact/penetration info without touching the
live simulation. Confirmed both the "no false positives" case and the
"detects a real forced self-collision" case (7 contacts on an extreme folded
pose). This de-risks the entire Feasibility Gate / collision checker design.

## Phase 0 — Freeze v1, set up v2 scaffold

- [ ] Tag/branch the current `main` as the v1 baseline (or just rely on git history — already fully committed through P23).
- [ ] Create `roboeval/agentic_v2/` package (empty `__init__.py` + subpackages per the layout below).
- [ ] Create `examples/run_agentic_v2.py` as the eventual new entry point (stub for now).
- [ ] Fix the two seeds/task set: `lift_pot`, `cube_handover`, `stack_two_blocks` (same three v1 already covers — no new tasks yet).
- [ ] Do **not** add new P24+ heuristics to v1 `roboeval/agentic/` going forward — it's frozen as the comparison baseline.

## Phase 1 — Correctness cleanup (independent of v2, do first, low risk)

These are v1 bugs worth fixing regardless of whether v2 ships, and they make
the v1-vs-v2 comparison fairer:
- [ ] `grasp_object`: never `close_gripper` after a failed `align_to_object` (currently it does — closing on a bad approach is part of why grasps look "attempted" when they shouldn't be).
- [ ] Add an automatic retreat (small vertical/backward move) after any failed grasp/align attempt, not just report failure and let the next primitive start from wherever the arm ended up.
- [ ] Standardize failure reasons as an enum-like set of string codes (`IK_UNREACHABLE`, `SELF_COLLISION`, `ENV_COLLISION`, `NO_VALID_GRASP`, `OBJECT_DISPLACED`, ...) instead of ad hoc free-text messages, in both v1 (where practical) and v2's design.
- [ ] Keep `benchmark_success` (raw RoboEval metric) and `quality`/`behavior` metrics as clearly separate fields everywhere they're reported (v1's `quality_assessment` already does this — carry the same split into v2 from day one).

## Phase 2 — Core typed data interfaces (no GPT, no real planning yet)

New module: `roboeval_v2/types.py` (or `agentic_v2/types.py`).
- [ ] `SceneState` — robot (qpos per arm, EE pose, gripper aperture, holding), objects (position, orientation, aabb, contact state), environment (table, obstacles), metrics (task_success, stage, collisions, slip). Reuse `roboeval/agentic/state.py`'s `collect_env_state()` logic as the source, just restructure into typed dataclasses instead of a raw dict.
- [ ] `SkillRequest` — `skill: str`, `object: str | None`, `roles: dict[str, str] | None` (e.g. `{"left": "left_handle", "right": "right_handle"}`), `goal: str`.
- [ ] `ConstraintSet`, `MotionCandidate`, `FeasibilityReport`, `MotionPlan`, `ExecutionReport` — per the user's original spec.
- [ ] Hand-write a few `SkillRequest` instances and confirm they construct/validate correctly. No pipeline yet — just prove the types are usable.

## Phase 3 — Collision checker (the now-de-risked piece — build this first among the "hard" phases)

New module: `agentic_v2/motion/collision_checker.py`.
- [ ] `check_collision(mojo, candidate_qpos) -> list[Contact]` using the verified `copy.copy(data)` + `mj_kinematics` + `mj_collision` pattern.
- [ ] Split into `check_self_collision` (both `geom1`/`geom2` belong to the robot) and `check_env_collision` (one side is scene, one is robot) and `check_held_object_collision` (checking a specific held object's geoms against the scene) — reuse the geom-ownership logic already in `roboeval/utils/metric_rollout.py` (`_robot_geoms`/`_scene_geoms`) rather than re-deriving it.
- [ ] Test against 10 known-safe poses (e.g. the robot's neutral/reset qpos, small perturbations) and 10 known-bad poses (e.g. the extreme-fold pose already proven to trigger 7 contacts) — assert stable, correct results both ways.

## Phase 4 — IK feasibility / candidate generation

New module: `agentic_v2/motion/ik.py`.
- [ ] `solve_ik_candidates(target_pose, side, n=5) -> list[IKCandidate]` — random-restart around the existing `roboeval/ik/base_ik.py` `qpos_from_site_pose` call (it only returns one local solution seeded from current qpos; get diversity by seeding from several perturbed starting qpos, keep only converged solutions within joint limits).
- [ ] Check each candidate against `roboeval/robots/config.py`'s `joint_limits`.
- [ ] Label each candidate `IK_FEASIBLE` / `IK_UNREACHABLE` per the enum.
- [ ] This is the first real "Feasibility Gate" output — test it standalone against a few known reachable/unreachable targets before wiring anything else to it.

## Phase 5 — Collision-aware path planner (naive version first)

New module: `agentic_v2/motion/path_planner.py`.
- [ ] `q_start -> q_goal`, N linear interpolation samples, `check_collision` (Phase 3) at each sample.
- [ ] If clear: accept the straight-line joint-space path.
- [ ] If not: try one intermediate waypoint (simple heuristic, e.g. lift-then-move) before falling back to failure - **do not** reach for RRT/CHOMP/cuRobo in this pass; naive-with-waypoint-fallback is enough to validate the pipeline shape.
- [ ] Decide here (explicit checkpoint, not default-drift): stay on `ee=True` per-step IK (current v1 approach) or switch to `JointPositionActionMode(ee=False, absolute=True)` and plan once in joint space, then execute the trajectory open-loop per chunk. The user's proposal recommends the latter — flag this as a real behavior change worth confirming before committing, since it changes how `Execution Monitor` (Phase 10) needs to interrupt/replan mid-trajectory.

## Phase 6 — Grasp candidate generation

New module: `agentic_v2/motion/candidate_generator.py` (grasp side).
- [ ] `generate_grasp_candidates(object_name) -> list[GraspCandidate]` (grasp pose, pregrasp pose, approach axis, required aperture, score).
- [ ] For objects with a real geometric affordance we already measured this session (pot handles, via P13's real geom data) - keep that as *environment affordance data*, not a scripted trajectory, exactly the same "measured, not guessed" principle v1 landed on for `SYMBOLIC_TARGETS`.
- [ ] For generic objects (cube, blocks) - derive candidates from `aabb`/`bbox` geometry (already wired in v1's `state.py` this session) rather than guessing offsets.
- [ ] Rank candidates by IK feasibility (Phase 4) + collision-free-ness (Phase 3), don't just take the first one.

## Phase 7 — Bimanual constraint layer

New module: `agentic_v2/constraints/bimanual.py`.
- [ ] `check_dual_arm_feasibility(left_pose, right_pose)` - joint IK feasibility must be checked on the **joint configuration**, not independently per arm (a real gap in v1 - self-collision was only caught reactively, mid-execution, never predicted).
- [ ] Relative-pose-preservation constraint for tasks like `lift_pot`, once both arms hold the object: penalize/reject candidates where the EE-to-object relative transform would need to change a lot between consecutive steps (this directly targets the "grasped both handles, then lost the grip during lift" failure mode found in tonight's last real run).

## Phase 8 — Handover / rendezvous planner

New module: `agentic_v2/skills/handover.py`.
- [ ] Replace v1's `handover_midpoint = (left_ee + right_ee) / 2` heuristic with a real optimization: sample rendezvous region -> donor IK -> receiver IK -> dual-arm collision -> grasp compatibility -> pick the best point.
- [ ] Reuse Phase 3/4/7 primitives - this phase is mostly composition, not new low-level machinery.

## Phase 9 — Skill library + re-attach the LLM

New modules: `agentic_v2/skills/{base,grasp,bimanual_grasp,lift,handover,transport,place}.py`, `agentic_v2/llm_planner.py`, `agentic_v2/prompts.py`.
- [ ] Each skill exposes precondition/postcondition/failure-reason, and internally runs candidate-generate -> feasibility-check -> plan -> execute -> verify (Phases 3-8), never exposing raw offsets to the caller.
- [ ] LLM prompt now only offers `skill` + semantic params (object, roles, goal) - no `ee_offset`/`steps`/`yaw`. This is the actual architecture change the whole plan is for; do it only after Phases 3-8 are independently tested, so a failure here can be attributed to *planning*, not *motion*.
- [ ] `FeasibilityGate` rejection -> structured reason string back to the LLM (e.g. `"HANDOVER_REGION_EMPTY: no pose is reachable by both arms without self-collision"`) instead of v1's `"target not reached within tolerance"`.

## Phase 10 — Execution monitor, replanner, evaluator

New modules: `agentic_v2/{executor,monitor,replanner,evaluator}.py`.
- [ ] Chunked execution with per-chunk invariant checks (object displacement, holding state) - this generalizes and replaces v1's ad hoc P19 (self-collision breaker) / P22 (object-bump breaker), making them a designed feature instead of a bolted-on patch.
- [ ] `benchmark_success` vs `quality`/`behavior_quality` reported as separate top-level fields (v1's `quality_assessment` already does this split - keep it).
- [ ] Optional: carry forward v1's P23 cross-trial memory idea (`summarize_trial_for_memory`) into v2's `memory.py`, now summarizing structured failure-reason codes instead of free-text messages.

## Phase 11 — Per-task validation (in this order)

1. `cube_handover` - best first test: exercises grasp, handover/rendezvous, dual-arm reachability, collision, release, without lift_pot's dual-grasp-then-lift complexity.
2. `lift_pot` - dual grasp + bimanual relative-pose constraint (Phase 7) + synchronized transport.
3. `stack_two_blocks` - grasp, transport, placement, optionally handoff (the 0.7m single-arm-carry problem v1's P15 found - v2's handover planner should let the LLM choose a handoff strategy for real, not just get a better-worded hint about one).

## Phase 12 — Ablation (the actual "results" section for your professor)

Compare, same 3 tasks, same metrics (`success rate`, `subtask_progress`,
`env_collision`, `self_collision`, `path_length`, `slip`):

| Method | Feasibility Gate | Collision-aware Planner | Replan | LLM |
| --- | --- | --- | --- | --- |
| v1 (this session's endpoint, P23) | no | no | yes (text feedback) | yes |
| v2-A | IK only | no | yes | yes |
| v2-B | full | yes | yes | yes |
| v2-C (ablation) | full | yes | **no** | fixed script |
| v2-Full | full | yes | yes | yes |

This table is the actual deliverable - it directly shows what each
architectural piece contributes, which is a much stronger thing to hand your
professor than a single success-rate number.

## Reference repos - do you need to download anything

Yes, worth getting these two locally (small, directly relevant, I can read
their real solver code instead of reasoning from the paper abstract):

- **ReKep** (`github.com/huangwl18/ReKep`) - `ik_solver.py`, `subgoal_solver.py`, `path_solver.py`, `constraint_generation.py` map almost 1:1 onto Phases 4/5/7 above. Highest value.
- **VoxPoser** (`github.com/huangwl18/VoxPoser`) - not for its code (different problem - value maps from vision), but `LMP.py` / `interfaces.py` / `planners.py` / `controllers.py` is a clean example of exactly the module split Phase 9 needs (LLM program / robot API / planner / controller kept separate).

Lower priority, skip cloning for now (read about them, don't try to integrate):
- **VLM-TAMP / kitchen-worlds** - PyBullet + PDDLStream, large and heavy; useful to *read* for the VLM-subgoal <-> TAMP-feasibility <-> replan loop shape, but porting its actual code to MuJoCo isn't practical.
- **cuRobo** - GPU/CUDA, Isaac-adjacent dependency footprint; a good *future* swap-in for Phase 5's path planner once the naive version works, not a day-one dependency.
- **LABOR-Agent** - small, directly bimanual, but its main contribution (LLM coordination/sequencing) is closer to what v1 already does; lower incremental value than ReKep/VoxPoser for *this* redesign specifically.

Put whichever you clone in a location outside `roboeval/agentic_v2/` (e.g. a
top-level `reference/` folder, gitignored) so they're readable but never
mistaken for part of the actual submission.

## Effort reality check

This is not a few-hours patch like tonight's P22/P23 - each of Phases 3-9 is
its own small project with its own testing checkpoint, matching the user's
own "don't skip phases" instruction. Recommended: treat Phase 0-2 as the
next concrete session (scaffold + types + the correctness fixes, all low
risk, no new hard dependencies), and revisit scope/pace after that lands.
