from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from roboeval.agentic_v2.motion.path_planner import JointPathPlanner
from roboeval.agentic_v2.types import FailureCode, FeasibilityReport, IKCandidate


class FakeChecker:
    def __init__(self, blocked=None):
        self.qpos_addresses = tuple(range(14))
        self.lower = np.full(14, -3.0)
        self.upper = np.full(14, 3.0)
        self.blocked = blocked or (lambda point: False)
        self.checked = []

    def check(self, point, constraints):
        point = np.asarray(point)
        self.checked.append(point.copy())
        if self.blocked(point):
            return FeasibilityReport(False, FailureCode.SELF_COLLISION, "blocked")
        return FeasibilityReport(True)


def fake_env():
    robot = SimpleNamespace(
        grippers={"left": object(), "right": object()},
        get_initial_qpos=lambda: np.zeros(16),
    )
    return SimpleNamespace(control_frequency=20, robot=robot)


def candidate(value: float, score: float = 0.0) -> IKCandidate:
    return IKCandidate("test", (value,) * 14, True, 0.0, 0.0, score=score)


def test_path_samples_are_bounded_and_all_checked() -> None:
    checker = FakeChecker()
    planner = JointPathPlanner(
        fake_env(), checker, maximum_joint_step=0.04,
        maximum_velocity=0.8, maximum_acceleration=2.0,
    )
    result = planner.plan_to_candidates(np.zeros(14), [candidate(0.4)])
    assert result.plan is not None
    positions = np.asarray([point.joint_positions for point in result.plan.points])
    assert np.max(np.abs(np.diff(positions, axis=0))) <= 0.04 + 1e-12
    assert len(checker.checked) == len(result.plan.points)
    dynamics = planner.dynamics(result.plan.points)
    assert dynamics["maximum_velocity"] <= 0.8 + 1e-9
    assert dynamics["maximum_acceleration"] <= 2.0 + 1e-9


def test_alternate_ik_goal_is_tried_before_waypoint() -> None:
    checker = FakeChecker(blocked=lambda point: bool(np.max(point) > 0.15))
    planner = JointPathPlanner(fake_env(), checker)
    result = planner.plan_to_candidates(
        np.zeros(14),
        [candidate(0.3), candidate(0.1)],
    )
    assert result.plan is not None
    assert result.plan.diagnostics["goal_index"] == 1
    assert result.plan.diagnostics["route"] == "straight"
    assert result.attempted_paths == 2


def test_blocked_path_returns_blocking_sample_evidence() -> None:
    checker = FakeChecker(blocked=lambda point: bool(np.linalg.norm(point) > 0.02))
    planner = JointPathPlanner(fake_env(), checker)
    result = planner.plan_to_candidates(np.zeros(14), [candidate(0.2)])
    assert result.plan is None
    assert result.report.failure_code is FailureCode.PATH_BLOCKED
    assert result.report.diagnostics["attempts"]
    assert result.report.diagnostics["attempts"][0]["sample"] is not None
