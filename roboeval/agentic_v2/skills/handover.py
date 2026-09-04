"""Verified donor-to-receiver transfer with sampled rendezvous poses."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from roboeval.agentic_v2.constraints.bimanual import ee_targets_for_object_pose
from roboeval.agentic_v2.executor import CLOSE_COMMAND, OPEN_COMMAND
from roboeval.agentic_v2.motion.candidate_generator import MAXIMUM_APERTURE
from roboeval.agentic_v2.skills.base import BaseSkill
from roboeval.agentic_v2.skills.grasp import GraspSkill
from roboeval.agentic_v2.skills.transport import TransportSkill, held_constraints
from roboeval.agentic_v2.state import capture_attachment, collect_scene_state, infer_handover_roles
from roboeval.agentic_v2.types import (
    AllowedContactPolicy,
    AllowedContactRule,
    ConstraintSet,
    FailureCode,
    Pose,
    RendezvousCandidate,
    SkillName,
    SkillRequest,
    SkillResult,
)


_PAD_OFFSET = 0.1034
_RECEIVER_GAP_AXIS_CANDIDATES = (
    np.array([0.0, 0.0, 1.0]),
    np.array([1.0, 0.0, 0.0]),
)
# Below this, the object is too small for donor and receiver to grip
# simultaneously at opposite ends without their hands colliding with each
# other (two Panda grippers are each roughly this scale) - use a staged
# place-then-regrasp handoff instead of a joint dual hold.
_DUAL_HOLD_MIN_SIZE = 0.10


class HandoverSkill(BaseSkill):
    def execute(self, request: SkillRequest) -> SkillResult:
        if request.skill is not SkillName.HANDOVER:
            raise ValueError("HandoverSkill received the wrong request")
        state = collect_scene_state(self.env)
        object_name = request.object_name
        if object_name not in state.objects:
            return self.failure(request, state, FailureCode.PRECONDITION_FAILED, "object unavailable", [])
        try:
            inferred = infer_handover_roles(state, object_name)
        except ValueError as error:
            return self.failure(request, state, FailureCode.PRECONDITION_FAILED, str(error), [])
        donor = request.roles.get("donor", inferred["donor"])
        receiver = request.roles.get("receiver", inferred["receiver"])
        if donor != inferred["donor"] or receiver != inferred["receiver"]:
            return self.failure(
                request, state, FailureCode.PRECONDITION_FAILED,
                f"live roles are donor={inferred['donor']}, receiver={inferred['receiver']}", [],
            )

        reports = []
        attempts = []
        obj = state.objects[object_name]
        if max(obj.canonical_size) <= _DUAL_HOLD_MIN_SIZE:
            donor_attachment = (
                self.context.attachments.get((object_name, donor))
                or capture_attachment(state, object_name, donor)
            )
            result = self._staged_regrasp(request, donor, receiver, donor_attachment, reports, attempts)
            if result is not None:
                return result
            code = self._dominant_failure_code(attempts)
            return self.failure(
                request, collect_scene_state(self.env), code,
                f"staged regrasp failed (last reason: {code.value})", reports,
                {"attempts": attempts, "strategy": "staged_regrasp"},
            )

        transport_failures = []
        reached_rendezvous = False
        for index, pose in enumerate(self._rendezvous_poses(state, object_name)):
            transported = TransportSkill(self.context).to_pose(request, pose)
            reports.extend(transported.execution_reports)
            if not transported.success:
                transport_failures.append(
                    transported.failure_code or FailureCode.PATH_BLOCKED
                )
                attempts.append({
                    "rendezvous": index,
                    "failure": (
                        transported.failure_code or FailureCode.PATH_BLOCKED
                    ).value,
                })
                continue
            reached_rendezvous = True
            result = self._transfer(request, donor, receiver, reports, attempts)
            if result is not None:
                return result
        if not reached_rendezvous and transport_failures:
            code = transport_failures[-1]
            return self.failure(
                request,
                collect_scene_state(self.env),
                code,
                f"all handover transports failed: {code.value}",
                reports,
                {"attempts": attempts},
            )
        # HANDOVER_REGION_EMPTY as a blanket code hides what actually blocked
        # every attempt (SELF_COLLISION vs GRASP_FAILED vs PATH_BLOCKED are
        # very different problems); surface the most recent concrete reason
        # instead so aggregate failure histograms are attributable.
        code = self._dominant_failure_code(attempts)
        return self.failure(
            request, collect_scene_state(self.env), code,
            f"no verified handover candidate (last reason: {code.value})", reports,
            {"attempts": attempts},
        )

    @staticmethod
    def _dominant_failure_code(attempts: list[dict]) -> FailureCode:
        for entry in reversed(attempts):
            value = entry.get("failure")
            if value:
                try:
                    return FailureCode(value)
                except ValueError:
                    continue
        return FailureCode.HANDOVER_REGION_EMPTY

    def _transfer(self, request, donor, receiver, reports, attempts):
        object_name = request.object_name
        state = collect_scene_state(self.env)
        donor_attachment = (
            self.context.attachments.get((object_name, donor))
            or capture_attachment(state, object_name, donor)
        )
        candidates = self._receiver_candidates(state, object_name, donor, receiver)
        if not candidates:
            attempts.append({
                "stage": "receiver_compatibility",
                "failure": FailureCode.NO_VALID_GRASP.value,
            })
            return self.failure(
                request,
                state,
                FailureCode.NO_VALID_GRASP,
                "receiver gripper is incompatible with the object cross-section",
                reports,
                {"attempts": attempts},
            )
        opened = self.context.executor.command_gripper(
            receiver, OPEN_COMMAND,
            constraints=held_constraints(object_name, (donor_attachment,)),
        )
        reports.append(opened)
        if not opened.success:
            return self.failure(
                request, opened.final_state,
                opened.failure_code or FailureCode.EXECUTION_DIVERGED,
                "receiver gripper could not open", reports,
            )
        for candidate in candidates:
            constraints = held_constraints(
                object_name, (donor_attachment,), extra_rules=candidate.contact_policy.rules,
            )
            pre, pre_ik, pre_path = self.move(
                name=f"{candidate.name}_pregrasp",
                targets={receiver: candidate.receiver_pregrasp_pose},
                constraints=constraints,
                require_holds=True,
                candidate_count=9,
            )
            if pre is not None:
                reports.append(pre)
            if pre is None or not pre.success:
                attempts.append(self._evidence(candidate.name, "receiver_pregrasp", pre, pre_ik, pre_path))
                continue
            terminal = ConstraintSet(
                allowed_contacts=AllowedContactPolicy(candidate.contact_policy.rules, 0.02),
                held_objects=(donor_attachment,),
                position_tolerance=0.06,
                orientation_tolerance=0.35,
            )
            approach, approach_ik, approach_path = self.move(
                name=f"{candidate.name}_approach",
                targets={receiver: candidate.receiver_grasp_pose},
                constraints=constraints,
                require_holds=True,
                candidate_count=9,
                stop_condition=lambda observed: receiver in observed.objects[object_name].held_by,
                terminal_constraints=terminal,
            )
            if approach is not None:
                reports.append(approach)
            if approach is None or not approach.success:
                attempts.append(self._evidence(candidate.name, "receiver_approach", approach, approach_ik, approach_path))
                # A blocked/interrupted approach can leave the receiver arm
                # partway into a crowded pose; a checked retreat keeps the
                # next candidate from starting there.
                retreat = self.retreat(receiver)
                if retreat is not None:
                    reports.append(retreat)
                continue
            closed = self.context.executor.command_gripper(
                receiver, CLOSE_COMMAND, steps=16, constraints=terminal,
            )
            reports.append(closed)
            both = collect_scene_state(self.env)
            if not closed.success or set(both.objects[object_name].held_by) != {donor, receiver}:
                attempts.append({
                    "candidate": candidate.name,
                    "stage": "receiver_close",
                    "failure": FailureCode.GRASP_FAILED.value,
                    "holders": both.objects[object_name].held_by,
                })
                self._recover_receiver(receiver, object_name, donor_attachment, reports)
                continue
            receiver_attachment = capture_attachment(both, object_name, receiver)
            dual = held_constraints(object_name, (donor_attachment, receiver_attachment))
            verification = self.context.executor.execute(
                self.context.executor.hold_plan("verify_dual_handover_hold", steps=8, constraints=dual),
                require_holds=True,
            )
            reports.append(verification)
            if not verification.success:
                attempts.append({
                    "candidate": candidate.name,
                    "stage": "dual_verify",
                    "failure": (
                        verification.failure_code or FailureCode.GRASP_FAILED
                    ).value,
                })
                self._recover_receiver(receiver, object_name, donor_attachment, reports)
                continue
            return self._release_donor(request, donor, receiver, donor_attachment, receiver_attachment, reports)
        return None

    def _staged_regrasp(self, request, donor, receiver, donor_attachment, reports, attempts):
        """Place a too-small-to-share object within reach of both arms, have
        the donor fully let go and clear out, then let the receiver grasp it
        fresh with the ordinary (already verified) single-arm grasp skill -
        avoids ever needing two grippers near the same few centimeters."""

        object_name = request.object_name
        state = collect_scene_state(self.env)
        surface_z = self._resting_surface_z(state, object_name)
        if surface_z is None:
            attempts.append({
                "stage": "staged_surface",
                "failure": FailureCode.HANDOVER_REGION_EMPTY.value,
                "reason": "no resting reference object to place onto",
            })
            return None
        obj = state.objects[object_name]
        half_height = obj.canonical_size[2] / 2.0
        # Servoing all the way down to near-zero clearance fights the table
        # once real contact force appears (position control keeps commanding
        # further down after the object is physically blocked) and trips the
        # tracking-divergence monitor. Hover with real clearance instead and
        # let the release + settle steps below drop it the rest of the way
        # under gravity.
        place_pose = Pose(
            (obj.pose.position[0], 0.0, surface_z + half_height + 0.015),
            obj.pose.quaternion_wxyz,
        )
        lower_constraints = held_constraints(
            object_name, (donor_attachment,),
            extra_rules=(
                AllowedContactRule(f"object:{object_name}", "scene:*table*"),
                AllowedContactRule(f"object:{object_name}", "scene:cabinet_*"),
            ),
        )
        lower, lower_ik, lower_path = self.move(
            name=f"{object_name}_staged_lower",
            targets=ee_targets_for_object_pose(place_pose, (donor_attachment,)),
            constraints=lower_constraints,
            require_holds=True,
            candidate_count=9,
        )
        if lower is not None:
            reports.append(lower)
        if lower is None or not lower.success:
            attempts.append(self._evidence(f"{object_name}_staged", "staged_lower", lower, lower_ik, lower_path))
            return None
        released = self.context.executor.command_gripper(donor, OPEN_COMMAND, constraints=lower_constraints)
        reports.append(released)
        if not released.success:
            attempts.append({
                "stage": "staged_release",
                "failure": (released.failure_code or FailureCode.RELEASE_FAILED).value,
            })
            return None
        self.context.attachments.pop((object_name, donor), None)
        # A plain vertical retreat leaves the donor hovering directly above
        # the drop point (which is at the workspace center so the receiver
        # can reach it) - the receiver's top-down approach then collides
        # with the donor's own arm. Pull the donor back toward its own side
        # as well as up so the center is actually clear.
        donor_sign = 1.0 if donor == "left" else -1.0
        donor_state = collect_scene_state(self.env)
        donor_current = donor_state.robot.arms[donor].ee_pose
        clear_target = Pose(
            (
                donor_current.position[0],
                donor_current.position[1] + donor_sign * 0.28,
                donor_current.position[2] + 0.12,
            ),
            donor_current.quaternion_wxyz,
        )
        clear, _, _ = self.move(
            name=f"handover_staged_clear_{donor}", targets={donor: clear_target}, require_holds=False,
        )
        if clear is not None:
            reports.append(clear)
        else:
            retreat = self.retreat(donor)
            if retreat is not None:
                reports.append(retreat)
        settle = self.context.executor.execute(
            self.context.executor.hold_plan("staged_settle", steps=10),
            require_holds=False,
        )
        reports.append(settle)

        grasp_result = GraspSkill(self.context).execute(
            SkillRequest(
                SkillName.GRASP, object_name=object_name,
                roles={receiver: "regrasp"}, goal="stable_hold",
            )
        )
        reports.extend(grasp_result.execution_reports)
        if not grasp_result.success:
            attempts.append({
                "stage": "staged_regrasp",
                "failure": (
                    grasp_result.failure_code.value
                    if grasp_result.failure_code else FailureCode.GRASP_FAILED.value
                ),
                "message": grasp_result.message,
                "diagnostics": grasp_result.diagnostics,
            })
            return None
        final = collect_scene_state(self.env)
        return SkillResult(
            request, True, f"staged handover of {object_name} from {donor} to {receiver}", final,
            execution_reports=tuple(reports),
            diagnostics={"donor": donor, "receiver": receiver, "strategy": "staged_regrasp"},
        )

    @staticmethod
    def _resting_surface_z(state, object_name):
        """Bottom-face height of another object at rest, used as a proxy for
        the table surface (avoids hardcoding or re-deriving table geometry)."""

        for name, other in state.objects.items():
            if name == object_name or other.held_by:
                continue
            return other.pose.position[2] - other.canonical_size[2] / 2.0
        return None

    def _release_donor(self, request, donor, receiver, donor_attachment, receiver_attachment, reports):
        object_name = request.object_name
        dual = held_constraints(object_name, (donor_attachment, receiver_attachment))
        opened = self.context.executor.command_gripper(donor, OPEN_COMMAND, constraints=dual)
        reports.append(opened)
        if not opened.success:
            return self.failure(
                request,
                opened.final_state,
                FailureCode.RELEASE_FAILED,
                "donor gripper failed to open",
                reports,
            )
        self.context.attachments.pop((object_name, donor), None)
        self.context.attachments[(object_name, receiver)] = receiver_attachment
        state = collect_scene_state(self.env)
        if donor in state.objects[object_name].held_by:
            current = state.robot.arms[donor].ee_pose
            direction = 1.0 if donor == "left" else -1.0
            target = Pose(
                (current.position[0], current.position[1] + 0.10 * direction, current.position[2] + 0.03),
                current.quaternion_wxyz,
            )
            single = held_constraints(
                object_name, (receiver_attachment,),
                extra_rules=(AllowedContactRule(f"robot:{donor}:finger", f"object:{object_name}"),),
            )
            retreat, _, _ = self.move(
                name=f"handover_release_retreat_{donor}", targets={donor: target},
                constraints=single, require_holds=True,
            )
            if retreat is not None:
                reports.append(retreat)
        final = collect_scene_state(self.env)
        if receiver not in final.objects[object_name].held_by or donor in final.objects[object_name].held_by:
            return self.failure(
                request, final, FailureCode.RELEASE_FAILED,
                "receiver hold was not isolated after donor release", reports,
            )
        self.context.attachments[(object_name, receiver)] = capture_attachment(final, object_name, receiver)
        return SkillResult(
            request, True, f"transferred {object_name} from {donor} to {receiver}", final,
            execution_reports=tuple(reports),
            diagnostics={"donor": donor, "receiver": receiver},
        )

    def _recover_receiver(self, receiver, object_name, donor_attachment, reports):
        constraints = held_constraints(
            object_name, (donor_attachment,),
            extra_rules=(AllowedContactRule(f"robot:{receiver}:finger", f"object:{object_name}"),),
        )
        reports.append(self.context.executor.command_gripper(receiver, OPEN_COMMAND, constraints=constraints))
        self.context.attachments.pop((object_name, receiver), None)
        retreat = self.retreat(receiver)
        if retreat is not None:
            reports.append(retreat)

    @staticmethod
    def _rendezvous_poses(state, object_name):
        obj = state.objects[object_name]
        z = max(obj.pose.position[2], 1.08)
        return (
            Pose((obj.pose.position[0], 0.0, z), obj.pose.quaternion_wxyz),
            Pose((0.55, 0.0, z + 0.04), obj.pose.quaternion_wxyz),
            Pose((0.48, 0.0, z + 0.07), obj.pose.quaternion_wxyz),
        )

    @staticmethod
    def _receiver_candidates(state, object_name, donor, receiver):
        obj = state.objects[object_name]
        # The donor grasps at the object's center (no offset). Scale the
        # receiver's offset with the object's own length so the two hands
        # land near opposite ends instead of crowding the same few
        # centimeters - a flat constant collided for a short rod and left
        # too much slack for a long block.
        # Use the fixed physical size, not the live world AABB: the object
        # tilts slightly as soon as the donor lifts it, which inflates the
        # live AABB and made this offset/aperture logic flicker between
        # runs even though the object's true geometry never changed.
        length = float(obj.canonical_size[1])
        # 0.025 left too little of the fingertip past the object's tip for a
        # reliable close (closed but never registered as held); 0.035 is the
        # smallest margin that both clears the donor's hand and lets the
        # receiver's fingers close on real material (empirically swept).
        tip_margin = 0.035
        offset = max(0.0, length / 2.0 - tip_margin) if max(obj.canonical_size) > 0.10 else 0.0
        receiver_sign = 1.0 if receiver == "left" else -1.0
        # canonical_size/offset are in the object's own body frame; rotate
        # the offset into world space with the object's *current* pose
        # instead of assuming world Y tracks the object's length axis -
        # that only held before because these objects were near-unrotated.
        object_rotation = np.asarray(obj.pose.as_matrix())[:3, :3]
        contact = np.asarray(obj.pose.position) + object_rotation @ np.array(
            [0.0, receiver_sign * offset, 0.0]
        )
        inward = np.array([0.0, -receiver_sign, 0.0])
        policy = AllowedContactPolicy(
            (
                AllowedContactRule(f"robot:{donor}:finger", f"object:{object_name}"),
                AllowedContactRule(f"robot:{receiver}:finger", f"object:{object_name}"),
            ),
            penetration_tolerance=0.012,
        )
        result = []
        for gap_index, gap in enumerate(_RECEIVER_GAP_AXIS_CANDIDATES):
            # Aperture must be checked per candidate axis, not assumed to be
            # world Z: an elongated object (e.g. a rod) can be too wide to
            # grip along one horizontal axis but fit along the other. `gap`
            # is a world-frame direction but canonical_size is body-frame,
            # so rotate it into the object's frame before comparing.
            gap_local = object_rotation.T @ gap
            required_aperture = float(np.dot(np.abs(gap_local), obj.canonical_size)) + 0.012
            if required_aperture > MAXIMUM_APERTURE:
                continue
            lateral = np.cross(gap, inward)
            norm = np.linalg.norm(lateral)
            if norm < 1e-6:
                continue
            lateral = lateral / norm
            base = np.column_stack((lateral, gap, inward))
            for index, matrix in enumerate((base, np.column_stack((-lateral, -gap, inward)))):
                rotation = Rotation.from_matrix(matrix)
                wrist = contact - matrix[:, 2] * _PAD_OFFSET
                pregrasp = wrist - matrix[:, 2] * 0.07
                result.append(RendezvousCandidate(
                    f"{object_name}_{receiver}_rendezvous_{gap_index}_{index}", obj.pose,
                    Pose.from_matrix(HandoverSkill._matrix(pregrasp, rotation)),
                    Pose.from_matrix(HandoverSkill._matrix(wrist, rotation)),
                    receiver, policy, float(np.linalg.norm(pregrasp - np.asarray(state.robot.arms[receiver].ee_pose.position))),
                ))
        return tuple(sorted(result, key=lambda item: item.score))

    @staticmethod
    def _matrix(position, rotation):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = position
        return matrix

    @staticmethod
    def _evidence(name, stage, execution, ik_result, path_result):
        code = execution.failure_code if execution is not None else (
            path_result.report.failure_code if path_result is not None else ik_result.report.failure_code
        )
        return {"candidate": name, "stage": stage, "failure": code.value if code else None}
