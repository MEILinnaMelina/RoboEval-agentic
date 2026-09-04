from __future__ import annotations

import numpy as np

from roboeval.agentic_v2.monitor import ExecutionMonitor
from roboeval.agentic_v2.types import (
    ArmState,
    ConstraintSet,
    FailureCode,
    FeasibilityReport,
    HeldObjectAttachment,
    ObjectState,
    Pose,
    RobotState,
    SceneState,
)


class FakeChecker:
    def __init__(self, report=None):
        self.report = report or FeasibilityReport(True)

    def check_live_contacts(self, constraints):
        return self.report


def state(*, joints=0.0, velocity=0.0, held_by=(), object_x=0.0):
    pose = Pose.identity()
    arms = {
        side: ArmState(side, (joints,) * 7, (velocity,) * 7, pose, 0.0, 0.08)
        for side in ("left", "right")
    }
    return SceneState(
        "cube_handover", "CubeHandover", 0, 20, (16,),
        RobotState((joints,) * 14, (velocity,) * 14, arms),
        {"cube": ObjectState("cube", Pose((object_x, 0, 0), (1, 0, 0, 0)), (object_x, 0, 0), (.04, .2, .04), held_by=held_by)},
        {"task_success": 0.0},
    )


def test_tracking_divergence_requires_patience_then_interrupts() -> None:
    monitor = ExecutionMonitor(FakeChecker())
    event = None
    for step in range(monitor.config.tracking_patience):
        event = monitor.evaluate(
            step=step,
            target_joints=np.ones(14),
            state=state(joints=0.0),
            constraints=ConstraintSet(),
        )
    assert event is not None
    assert event.code is FailureCode.EXECUTION_DIVERGED


def test_forbidden_live_contact_interrupts_immediately() -> None:
    checker = FakeChecker(FeasibilityReport(False, FailureCode.SELF_COLLISION, "contact"))
    event = ExecutionMonitor(checker).evaluate(
        step=0,
        target_joints=np.zeros(14),
        state=state(),
        constraints=ConstraintSet(),
    )
    assert event is not None
    assert event.code is FailureCode.SELF_COLLISION


def test_lost_grasp_interrupts_before_transport_continues() -> None:
    attachment = HeldObjectAttachment("cube", "right", Pose.identity())
    monitor = ExecutionMonitor(FakeChecker())
    event = None
    for step in range(monitor.config.hold_patience):
        event = monitor.evaluate(
            step=step,
            target_joints=np.zeros(14),
            state=state(held_by=()),
            constraints=ConstraintSet(held_objects=(attachment,)),
        )
    assert event is not None
    assert event.code is FailureCode.SLIP_DETECTED


def test_single_step_hold_blip_does_not_interrupt() -> None:
    # A momentary "not holding" reading below patience must not fail the
    # grasp - only a loss that persists should (see test above).
    attachment = HeldObjectAttachment("cube", "right", Pose.identity())
    monitor = ExecutionMonitor(FakeChecker())
    event = monitor.evaluate(
        step=0,
        target_joints=np.zeros(14),
        state=state(held_by=()),
        constraints=ConstraintSet(held_objects=(attachment,)),
    )
    assert event is None
    # And recovering resets the counter rather than carrying it forward.
    event = monitor.evaluate(
        step=1,
        target_joints=np.zeros(14),
        state=state(held_by=("right",)),
        constraints=ConstraintSet(held_objects=(attachment,)),
    )
    assert event is None
    assert monitor._hold_failures == 0


def test_protected_object_displacement_interrupts() -> None:
    event = ExecutionMonitor(FakeChecker()).evaluate(
        step=0,
        target_joints=np.zeros(14),
        state=state(object_x=0.1),
        constraints=ConstraintSet(),
        protected_objects={"cube": Pose.identity()},
    )
    assert event is not None
    assert event.code is FailureCode.OBJECT_DISPLACED
