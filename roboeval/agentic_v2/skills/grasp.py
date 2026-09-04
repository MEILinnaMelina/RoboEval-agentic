"""Verified single-arm grasp skill."""

from __future__ import annotations

from typing import Any

from roboeval.agentic_v2.executor import CLOSE_COMMAND, OPEN_COMMAND
from roboeval.agentic_v2.state import capture_attachment, collect_scene_state
from roboeval.agentic_v2.types import (
    AllowedContactPolicy,
    AllowedContactRule,
    ConstraintSet,
    FailureCode,
    Pose,
    SkillName,
    SkillRequest,
    SkillResult,
)
from roboeval.agentic_v2.skills.base import BaseSkill


class GraspSkill(BaseSkill):
    def execute(self, request: SkillRequest) -> SkillResult:
        if request.skill is not SkillName.GRASP:
            raise ValueError("GraspSkill received a non-grasp request")
        state = collect_scene_state(self.env)
        object_name = request.object_name
        if object_name not in state.objects:
            return self.failure(
                request, state, FailureCode.PRECONDITION_FAILED,
                f"object {object_name!r} is unavailable", [],
            )
        side = self._select_side(request, state)
        if side in state.objects[object_name].held_by:
            return SkillResult(request, True, f"{side} already holds {object_name}", state)
        candidates = self.context.candidates.grasp_candidates(object_name, side, state)
        if not candidates:
            return self.failure(
                request, state, FailureCode.NO_VALID_GRASP,
                f"no aperture-compatible grasp for {object_name}", [],
            )

        reports = []
        attempts: list[dict[str, Any]] = []
        open_report = self.context.executor.command_gripper(
            side,
            OPEN_COMMAND,
            constraints=ConstraintSet(allowed_contacts=candidates[0].contact_policy),
        )
        reports.append(open_report)
        if not open_report.success:
            return self.failure(
                request, open_report.final_state,
                open_report.failure_code or FailureCode.EXECUTION_DIVERGED,
                "failed to open before approach", reports,
            )

        for candidate in candidates:
            result = self._attempt(request, candidate, reports)
            attempts.append(result[1])
            if result[0] is not None:
                return result[0]
        state = collect_scene_state(self.env)
        return self.failure(
            request,
            state,
            FailureCode.NO_VALID_GRASP,
            f"all {len(candidates)} grasp candidates failed",
            reports,
            {"attempts": attempts},
        )

    def _attempt(self, request, candidate, reports):
        object_name = candidate.object_name
        side = candidate.side
        initial = collect_scene_state(self.env)
        protected = {object_name: initial.objects[object_name].pose}
        pre_execution, pre_ik, pre_path = self.move(
            name=f"{candidate.name}_pregrasp",
            targets={side: candidate.pregrasp_pose},
            protected_objects=protected,
            require_holds=False,
        )
        evidence = self._evidence(candidate.name, "pregrasp", pre_execution, pre_ik, pre_path)
        if pre_execution is None or not pre_execution.success:
            if pre_execution is not None:
                reports.append(pre_execution)
                self._record_recovery(reports, side)
            return None, evidence
        reports.append(pre_execution)

        approach_constraints = ConstraintSet(allowed_contacts=candidate.contact_policy)
        # No stop_condition: is_gripper_holding_object() is a pure
        # pad-contact check with no requirement that the gripper be closed
        # or gripping firmly (roboeval/robots/gripper.py:184), so an early
        # stop on "first contact" can leave the wrist short of the
        # precisely-computed grasp pose - most visible for objects resting
        # close to a support surface, where the shortfall leaves fingers
        # nearer the table on close. Travel the full planned path instead.
        approach, approach_ik, approach_path = self.move(
            name=f"{candidate.name}_approach",
            targets={side: candidate.grasp_pose},
            constraints=approach_constraints,
            protected_objects=protected,
            require_holds=False,
        )
        evidence = self._evidence(candidate.name, "approach", approach, approach_ik, approach_path)
        if approach is None or not approach.success:
            if approach is not None:
                reports.append(approach)
            self._record_recovery(reports, side)
            return None, evidence
        reports.append(approach)

        close_report = self.context.executor.command_gripper(
            side,
            CLOSE_COMMAND,
            steps=16,
            constraints=approach_constraints,
        )
        reports.append(close_report)
        if not close_report.success:
            self._record_recovery(reports, side, open_first=True, constraints=approach_constraints)
            evidence.update({"stage": "close", "failure": close_report.failure_code.value})
            return None, evidence
        grasped = collect_scene_state(self.env)
        if side not in grasped.objects[object_name].held_by:
            self._record_recovery(reports, side, open_first=True, constraints=approach_constraints)
            evidence.update({"stage": "verify_contact", "failure": FailureCode.GRASP_FAILED.value})
            return None, evidence

        attachment = capture_attachment(grasped, object_name, side)
        verified = self._verify_lift(candidate, attachment, reports)
        if verified is None:
            self._record_recovery(reports, side, open_first=True, constraints=approach_constraints)
            evidence.update({"stage": "verify_lift", "failure": FailureCode.SLIP_DETECTED.value})
            return None, evidence
        self.context.attachments[(object_name, side)] = attachment
        final_state = collect_scene_state(self.env)
        return SkillResult(
            request,
            True,
            f"verified {side} grasp of {object_name}",
            final_state,
            execution_reports=tuple(reports),
            diagnostics={"candidate": candidate.name, "attachment": attachment},
        ), evidence

    def _verify_lift(self, candidate, attachment, reports):
        state = collect_scene_state(self.env)
        side = candidate.side
        current = state.robot.arms[side].ee_pose
        target = Pose(
            (current.position[0], current.position[1], current.position[2] + 0.025),
            current.quaternion_wxyz,
        )
        departure_policy = AllowedContactPolicy(
            rules=candidate.contact_policy.rules
            + (
                AllowedContactRule(f"object:{candidate.object_name}", "scene:*table*"),
                AllowedContactRule(f"object:{candidate.object_name}", "scene:cabinet_*"),
            ),
            penetration_tolerance=0.01,
        )
        constraints = ConstraintSet(
            allowed_contacts=departure_policy,
            held_objects=(attachment,),
            position_tolerance=0.035,
            orientation_tolerance=0.25,
        )
        execution, _, _ = self.move(
            name=f"{candidate.name}_verification_lift",
            targets={side: target},
            constraints=constraints,
            require_holds=True,
            candidate_count=5,
        )
        if execution is not None:
            reports.append(execution)
        if execution is None or not execution.success:
            return None
        final_state = collect_scene_state(self.env)
        if side not in final_state.objects[candidate.object_name].held_by:
            return None
        return execution

    def _record_recovery(
        self,
        reports,
        side: str,
        *,
        open_first: bool = False,
        constraints: ConstraintSet | None = None,
    ) -> None:
        if open_first:
            opened = self.context.executor.command_gripper(
                side,
                OPEN_COMMAND,
                constraints=constraints,
            )
            reports.append(opened)
            if not opened.success:
                return
        for key in tuple(self.context.attachments):
            if key[1] == side:
                self.context.attachments.pop(key, None)
        retreat = self.retreat(side)
        if retreat is not None:
            reports.append(retreat)

    @staticmethod
    def _select_side(request: SkillRequest, state) -> str:
        direct = [side for side in ("left", "right") if side in request.roles]
        if len(direct) == 1:
            return direct[0]
        for role in ("donor", "receiver"):
            value = request.roles.get(role)
            if value in ("left", "right"):
                return value
        if request.strategy in ("left", "right"):
            return request.strategy
        obj = state.objects[request.object_name]
        return min(
            ("left", "right"),
            key=lambda side: sum(
                (state.robot.arms[side].ee_pose.position[index] - obj.pose.position[index]) ** 2
                for index in range(3)
            ),
        )

    @staticmethod
    def _evidence(name, stage, execution, ik_result, path_result):
        if execution is not None:
            code = execution.failure_code
        elif path_result is not None:
            code = path_result.report.failure_code
        else:
            code = ik_result.report.failure_code
        return {
            "candidate": name,
            "stage": stage,
            "failure": code.value if code else None,
            "ik_accepted": len(ik_result.accepted),
            "ik_rejected": len(ik_result.rejected),
            "path_attempted": path_result.attempted_paths if path_result else 0,
        }
