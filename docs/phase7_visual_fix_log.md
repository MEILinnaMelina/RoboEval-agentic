# Phase 7 Visual Fix Log

This log tracks visual-quality fixes for the RoboEval API agent runs. Each
experiment should change one thing at a time and record the exact command,
success metric, holding state, collisions, jerk, and whether the change was kept.

## Baseline

- Commit before visual fixes: `0836603`
- API diagnostic run:
  - Output: `outputs/phase7_visual_diag_openai`
  - `cube_handover`: `task_success=1.0`, final holding left=`true`, right=`false`, env collisions=`0`, but visual grasp may not look convincing.
  - `stack_two_blocks`: `task_success=1.0`, env collisions=`3`, slip count=`1`, `overall_rms_cartesian_jerk=43.687`, visually severe shaking.

## Issue Hypotheses

- `C1`: `CubeHandover` names the object `cube`, but the RoboEval environment creates a `Rod`; visual checking should treat it as a rod-like object.
- `C2`: Current `CubeHandover` playbook has no visible open-close receiving motion. It relies on existing gripper commands and RoboEval holding detection.
- `C3`: Current CubeHandover symbolic target offsets satisfy the holding metric but may not place the rod visually between the gripper fingers.
- `C4`: Gripper command timing may be delicate; naive open/close changes previously broke success.
- `S1`: `StackTwoBlocks` placement height is close to the success boundary; final success checks contact/release, not smoothness.
- `S2`: Linear end-effector interpolation plus IK convergence warnings likely contributes to high jerk.
- `S3`: Release is followed by immediate stop, with no settle/retract stage, so contact transients remain visible.
- `S4`: Arm path and parking poses may cause large unnecessary movements.
- `S5`: RoboEval `task_success` does not fail high jerk or collision cases, so visual-quality thresholds must be tracked separately.

## Quality Targets

- Keep RoboEval `task_success=1.0`.
- Preserve primitive-only control: no raw joint, torque, or MuJoCo action output from the LLM.
- CubeHandover final state should show the receiver holding the object and the giver released, with a clear trajectory GIF.
- StackTwoBlocks should reduce visible shaking; target lower `overall_rms_cartesian_jerk` and no increase in collision count.

## Experiments

| ID | Hypothesis | Change | Command | Result | Decision |
| --- | --- | --- | --- | --- | --- |
| E0 | Baseline | No control change; add visual diagnostics/GIF only. | `python examples/run_phase7_llm_agent.py --provider openai --model gpt-5.6-terra --task all --trials 1 --max-steps 10 --reasoning-effort low --record-gif --gif-every 20 --output-dir outputs/phase7_visual_diag_openai` | All tasks metric-success. Cube final holding left=true/right=false. Stack collisions=3, rms jerk=43.687. | Keep diagnostics; control still needs visual-quality fixes. |
| E1 | `S4` | In-memory only: remove `left_safe_parking` from StackTwoBlocks playbook. | Ad hoc mock run with current playbook vs. `base[1:]`. | Current: success=true, collisions=3, path=2.301, rms jerk=43.687. No left parking: success=false, collisions=3, path=2.240, rms jerk=45.305. | Reject. Left parking is not the main cause and appears to help success. |
| E2 | `S2` | In-memory only: replace linear pose ramp with smoothstep or cosine easing. | Ad hoc mock run on StackTwoBlocks. | Linear: success=true, collisions=3, rms jerk=43.687. Smoothstep: success=false, collisions=3, rms jerk=34.054. Cosine: success=false, collisions=4, rms jerk=33.393. | Partial. Easing lowers jerk but breaks final stacking; revisit with placement/settle changes. |
| E3 | `S3` | In-memory only: change StackTwoBlocks `release_object` steps from 40 to 320 while keeping placement unchanged. | Ad hoc mock run on StackTwoBlocks. | All variants success=true with identical collisions=3, slip=1, path=2.301, rms jerk=43.687. | Reject as root cause. Release duration alone does not affect the measured shaking. |
| E4 | `S1` + `S2` | In-memory only: keep slower placement `high_steps=400`, `place_steps=240`, sweep `place_clearance=0.035..0.080`. | Ad hoc mock run on StackTwoBlocks. | All variants failed after cleanup; block_0 final z stayed near table height. Best jerk was around clearance=0.044 with rms=18.871, but final task_success=0.0. | Reject as standalone fix. Height tuning alone cannot produce a stable released stack. |
| E5 | `S1` | In-memory only: sweep StackTwoBlocks `high_clearance=0.045..0.20` and `place_clearance=0.041..0.060` with slower placement. | Ad hoc mock run on StackTwoBlocks. | All checked combinations failed after cleanup; higher clearances often increased rms jerk above 50 and block_0 still ended near table height. | Reject as standalone fix. The grasp/transport is unstable before final release. |
| E6 | `S3` | In-memory only: test right-hand grasp offsets, then transport block_0 high above block_1. | Ad hoc mock run on StackTwoBlocks. | Top offsets kept holding after transport, but block_0 lagged behind the desired support y by about 0.08. Side offsets mostly failed or had high jerk. | Partial. Keep top-style grasp for now; test lateral compensation during placement. |
| E7 | `S2` | In-memory only: stop pose ramp as soon as `task_success=1.0` during StackTwoBlocks placement. | Manual primitive run. | Same as baseline: placement ended at 303 sim steps, collisions=3, slip=1, rms jerk=43.687, final task_success=1.0. | Reject. Success appears only at the end of the current place motion. |
| E8 | `S2` + `S3` | Keep control change: slow StackTwoBlocks placement to `high_steps=400`, `place_steps=240`; continue cleanup only when the next cleanup primitive is actually needed. | `python examples/run_phase7_llm_agent.py --provider mock --model mock --task all --trials 1 --max-steps 10 --record-gif --gif-every 20 --output-dir outputs/phase7_stack_slow_place_mock` | All tasks success. Stack: collisions=3, slip=0, path=1.749, rms jerk=20.050, down from baseline rms=43.687. | Keep. This reduces StackTwoBlocks shaking while preserving RoboEval success. |
| E9 | `C2` + `C3` | Inspect CubeHandover per-step diagnostics. | Read `outputs/phase7_stack_slow_place_mock/cube_handover/trial_001/success_trajectory_log.json`. | Step 0 right holding=true by contact, but after lift right holding=false and rod z stays near table. Step 2 left holding=true by touching rod on table. | Confirms user visual issue. Need real lift/transport before receiver contact. |
| E10 | `C2` + `C3` + `C4` | In-memory only: test CubeHandover right gripper offsets and explicit open/close before lift. | Ad hoc mock runs on CubeHandover. | Current and sampled top/side offsets did not lift the rod; after lift right_holding=false and rod z stayed near table. | Reject as standalone fix. CubeHandover needs either a task-specific pickup primitive or must be reported as contact-transfer under current RoboEval success. |
| E11 | `C2` + `C3` + `S3` | Add optional `--kinematic-attachments`: after a physical holding contact, the primitive temporarily carries the object with the gripper until release. | `python examples/run_phase7_llm_agent.py --provider mock --model mock --task all --trials 1 --max-steps 10 --record-gif --gif-every 20 --kinematic-attachments --output-dir outputs/phase7_attach_mock` | All tasks success. Cube lift_distance=0.071 and rod final z=1.042 instead of table height. Stack block_0 final z=1.007, collisions=2, slip=0, path=1.659, rms jerk=24.202. | Keep as explicit visual-demo mode. This fixes Cube visual lift and improves Stack stability, while preserving primitive-only LLM planning. |
| E12 | `C2` + `S3` | Validate `--kinematic-attachments` with real OpenAI planner. | `python examples/run_phase7_llm_agent.py --provider openai --model gpt-5.6-terra --task all --trials 1 --max-steps 10 --reasoning-effort low --record-gif --gif-every 20 --kinematic-attachments --output-dir outputs/phase7_attach_openai` | All tasks success. Cube rod z=1.042, lift_distance=0.071. Stack block_0 z=1.007, collisions=2, slip=0, path=1.659, rms jerk=24.202. | Keep. Need tune Stack x-centering next. |
| E13 | `S1` | In-memory only: add x-bias to StackTwoBlocks placement target under `--kinematic-attachments`. | Ad hoc mock run with `xbias=0.0..0.08`. | Baseline xbias=0 succeeded. All positive x-bias variants failed, often with y drift, table drop, and rms jerk above 65. | Reject. Centering cannot be fixed by simple x target bias. |
| E14 | `S3` | Change kinematic release to open first, detach after opening, and require cleanup when internal attachment exists. | `python examples/run_phase7_llm_agent.py --provider mock --model mock --task all --trials 1 --max-steps 10 --record-gif --gif-every 20 --kinematic-attachments --output-dir outputs/phase7_attach_release_mock` | LiftPot and Cube passed, but Stack failed after release with final task_success=0.0. | Reject. Stack remains a fragile contact state if detached and re-evaluated after release. |
| E15 | Final API check | Current kept changes: slower Stack placement, state-aware cleanup, optional `--kinematic-attachments`, GIF diagnostics. | `python examples/run_phase7_llm_agent.py --provider openai --model gpt-5.6-terra --task all --trials 1 --max-steps 10 --reasoning-effort low --record-gif --gif-every 20 --kinematic-attachments --output-dir outputs/phase7_attach_final_openai` | All tasks success. Cube rod z=1.042 and lift_distance=0.071. Stack block_0 z=1.007, block_1 z=0.970, collisions=2, slip=0, path=1.659, rms jerk=24.202. | Keep and commit. Remaining visual limitation: Stack x-centering is imperfect, but shaking/slip improved substantially. |
