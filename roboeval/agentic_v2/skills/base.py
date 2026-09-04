"""Shared deterministic machinery behind semantic Agentic v2 skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from roboeval.agentic_v2.executor import MonitoredExecutor
from roboeval.agentic_v2.motion.candidate_generator import CandidateGenerator
from roboeval.agentic_v2.motion.collision_checker import CollisionChecker
from roboeval.agentic_v2.motion.ik import IKSearchResult, MultiStartIK
from roboeval.agentic_v2.motion.path_planner import JointPathPlanner, PathSearchResult
from roboeval.agentic_v2.state import collect_scene_state
from roboeval.agentic_v2.types import (
    ConstraintSet,
    ExecutionReport,
    FailureCode,
    HeldObjectAttachment,
    Pose,
    SkillRequest,
    SkillResult,
    to_jsonable,
)


@dataclass
class SkillContext:
    env: Any
    checker: CollisionChecker
    ik: MultiStartIK
    planner: JointPathPlanner
    executor: MonitoredExecutor
    candidates: CandidateGenerator
    attachments: dict[tuple[str, str], HeldObjectAttachment] = field(default_factory=dict)
    planning_trace: list[dict[str, Any]] = field(default_factory=list)
    # Support-surface height (world z of the object's bottom face) observed
    # while each object was last at rest, keyed by object name. Lets a
    # staged handover put an object back down at a known-good height even
    # in scenes with no other resting object to reference.
    resting_surfaces: dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        env: Any,
        *,
        render: bool = False,
        frame_callback: Any | None = None,
        feasibility_gate: str = "full",
    ) -> "SkillContext":
        if feasibility_gate not in {"full", "ik-only"}:
            raise ValueError(f"unknown feasibility gate {feasibility_gate!r}")
        checker = CollisionChecker(
            env,
            enforce_contacts=feasibility_gate == "full",
        )
        return cls(
            env=env,
            checker=checker,
            ik=MultiStartIK(env, checker),
            planner=JointPathPlanner(env, checker),
            executor=MonitoredExecutor(
                env,
                collision_checker=checker,
                render=render,
                frame_callback=frame_callback,
            ),
            candidates=CandidateGenerator(env),
        )


class BaseSkill:
    def __init__(self, context: SkillContext) -> None:
        self.context = context

    @property
    def env(self) -> Any:
        return self.context.env

    def move(
        self,
        *,
        name: str,
        targets: Mapping[str, Pose],
        constraints: ConstraintSet | None = None,
        protected_objects: Mapping[str, Pose] | None = None,
        require_holds: bool = True,
        candidate_count: int = 7,
        stop_condition: Callable[[Any], bool] | None = None,
        terminal_constraints: ConstraintSet | None = None,
    ) -> tuple[ExecutionReport | None, IKSearchResult, PathSearchResult | None]:
        started = perf_counter()
        constraints = constraints or ConstraintSet()
        state = collect_scene_state(self.env)
        ik_result = self.context.ik.solve_candidates(
            targets,
            count=candidate_count,
            constraints=constraints,
        )
        if not ik_result.report.feasible:
            self._record_motion_trace(
                name, targets, constraints, ik_result, None, None, started
            )
            return None, ik_result, None
        path_result = self.context.planner.plan_to_candidates(
            state.robot.joint_positions,
            ik_result.accepted,
            constraints=constraints,
            name=name,
        )
        if path_result.plan is None:
            self._record_motion_trace(
                name, targets, constraints, ik_result, path_result, None, started
            )
            return None, ik_result, path_result
        execution = self.context.executor.execute(
            path_result.plan,
            protected_objects=protected_objects,
            require_holds=require_holds,
            stop_condition=stop_condition,
            terminal_constraints=terminal_constraints,
        )
        self._record_motion_trace(
            name, targets, constraints, ik_result, path_result, execution, started
        )
        return execution, ik_result, path_result

    def _record_motion_trace(
        self,
        name: str,
        targets: Mapping[str, Pose],
        constraints: ConstraintSet,
        ik_result: IKSearchResult,
        path_result: PathSearchResult | None,
        execution: ExecutionReport | None,
        started: float,
    ) -> None:
        plan = path_result.plan if path_result is not None else None
        self.context.planning_trace.append(
            {
                "name": name,
                "targets": to_jsonable(targets),
                "constraints": to_jsonable(constraints),
                "ik": {
                    "report": to_jsonable(ik_result.report),
                    "accepted": to_jsonable(ik_result.accepted),
                    "rejected": to_jsonable(ik_result.rejected),
                },
                "path": {
                    "report": to_jsonable(path_result.report),
                    "attempted_paths": path_result.attempted_paths,
                    "plan": to_jsonable(plan),
                }
                if path_result is not None
                else None,
                "execution": to_jsonable(execution) if execution is not None else None,
                "elapsed_seconds": perf_counter() - started,
            }
        )

    def retreat(self, side: str, *, distance: float = 0.08) -> ExecutionReport | None:
        state = collect_scene_state(self.env)
        current = state.robot.arms[side].ee_pose
        target = Pose(
            (
                current.position[0],
                current.position[1],
                current.position[2] + distance,
            ),
            current.quaternion_wxyz,
        )
        execution, _, _ = self.move(
            name=f"checked_retreat_{side}",
            targets={side: target},
            require_holds=False,
        )
        return execution

    @staticmethod
    def failure(
        request: SkillRequest,
        state: Any,
        code: FailureCode,
        message: str,
        reports: list[ExecutionReport],
        diagnostics: dict[str, Any] | None = None,
    ) -> SkillResult:
        return SkillResult(
            request=request,
            success=False,
            message=message,
            state=state,
            failure_code=code,
            execution_reports=tuple(reports),
            diagnostics=diagnostics or {},
        )
