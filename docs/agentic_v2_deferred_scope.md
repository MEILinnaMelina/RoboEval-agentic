# Agentic v2 Deferred Scope

Phase 12 is a scope gate, not an implementation phase. The v2 runner accepts
only lift_pot, cube_handover, and stack_two_blocks.

The following work remains intentionally disabled until the complete Phase 11
base-task table exists:

- position/orientation and longer-horizon RoboEval variants;
- vision-only or learned perception;
- RRT, CHOMP, ReKep trajectory optimization, cuRobo, and Isaac migration;
- PDDLStream or another symbolic task planner;
- learned grasp policies, OpenVLA, and direct low-level model control.

Adding any item above requires a new experiment-manifest version. It must not
silently change the base-task v2 result table.

