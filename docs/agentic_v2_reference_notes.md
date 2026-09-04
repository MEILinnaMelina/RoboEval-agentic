# Agentic v2 Reference Notes

## ReKep

Local checkout: `E:\djf\ReKep-main`.

- `ik_solver.py` is a thin Lula CCD IK wrapper. RoboEval v2 does not depend on
  Lula and independently wraps RoboEval's dm_control/MuJoCo IK.
- `path_solver.py` optimizes end-effector control points against SDF and
  keypoint constraints. RoboEval v2 initially uses checked joint interpolation
  and a bounded waypoint fallback.
- `subgoal_solver.py`, `constraint_generation.py`, and `main.py` inform the
  separation between semantic stages, constraints, planning, and execution.
- No license file was visible in this checkout on 2026-09-04. Only ideas and
  architecture are used; source is not copied.

## VoxPoser

Local checkout: `E:\djf\VoxPoser-main`.

- `src/LMP.py`, `interfaces.py`, `planners.py`, and `controllers.py` inform the
  separation of model calls, environment adapters, planners, and controllers.
- The repository has an MIT license. Any future direct adaptation must retain
  its notice and be documented here.
- VoxPoser is not a runtime dependency of RoboEval v2.

