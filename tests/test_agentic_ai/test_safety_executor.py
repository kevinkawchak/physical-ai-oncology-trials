"""Unit tests for agentic-ai/examples-agentic-ai/05_safety_constrained_agent_executor.py.

Tests cover ConstraintType, ConstraintSeverity, ViolationResponse,
ActionCategory enums, SafetyConstraint, ConstraintResult, AgentAction,
SafetyVerdict dataclasses, and constraint checking logic.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "agentic-ai/examples-agentic-ai/05_safety_constrained_agent_executor.py"


@pytest.fixture()
def mod():
    return load_module("safety_executor", MOD_PATH)


class TestEnums:
    def test_constraint_type(self, mod):
        names = [e.name for e in mod.ConstraintType]
        for t in ("PRE_CONDITION", "POST_CONDITION", "INVARIANT", "SAFETY_GATE"):
            assert t in names

    def test_constraint_severity(self, mod):
        names = [e.name for e in mod.ConstraintSeverity]
        for s in ("ADVISORY", "BLOCKING", "CRITICAL", "EMERGENCY"):
            assert s in names

    def test_violation_response(self, mod):
        names = [e.name for e in mod.ViolationResponse]
        for r in ("LOG_ONLY", "REJECT_ACTION", "ROLLBACK", "EMERGENCY_STOP"):
            assert r in names

    def test_action_category(self, mod):
        names = [e.name for e in mod.ActionCategory]
        for c in ("OBSERVATION", "NAVIGATION", "MANIPULATION", "ENERGY", "CRITICAL"):
            assert c in names


class TestSafetyConstraint:
    def test_creation(self, mod):
        constraint = mod.SafetyConstraint(
            constraint_id="SC-001",
            name="Force limit",
            description="End-effector force must not exceed 10 N",
            constraint_type=mod.ConstraintType.INVARIANT,
            severity=mod.ConstraintSeverity.CRITICAL,
            violation_response=mod.ViolationResponse.EMERGENCY_STOP,
            predicate=lambda state: state.get("force", 0) <= 10.0,
            applicable_actions=[mod.ActionCategory.MANIPULATION],
            regulatory_reference="IEC 80601-2-77",
        )
        assert constraint.constraint_id == "SC-001"
        assert constraint.severity == mod.ConstraintSeverity.CRITICAL

    def test_check_satisfied(self, mod):
        constraint = mod.SafetyConstraint(
            constraint_id="SC-002",
            name="Workspace check",
            description="Robot must be within workspace",
            constraint_type=mod.ConstraintType.PRE_CONDITION,
            severity=mod.ConstraintSeverity.BLOCKING,
            violation_response=mod.ViolationResponse.REJECT_ACTION,
            predicate=lambda state: state.get("in_workspace", False),
            applicable_actions=[mod.ActionCategory.NAVIGATION],
        )
        result = constraint.check({"in_workspace": True})
        assert isinstance(result, mod.ConstraintResult)
        assert result.satisfied is True

    def test_check_violated(self, mod):
        constraint = mod.SafetyConstraint(
            constraint_id="SC-003",
            name="Velocity limit",
            description="Velocity must not exceed 0.5 m/s",
            constraint_type=mod.ConstraintType.INVARIANT,
            severity=mod.ConstraintSeverity.CRITICAL,
            violation_response=mod.ViolationResponse.EMERGENCY_STOP,
            predicate=lambda state: state.get("velocity", 0) <= 0.5,
            applicable_actions=[mod.ActionCategory.NAVIGATION],
        )
        result = constraint.check({"velocity": 1.0})
        assert result.satisfied is False
        assert result.violation_response == mod.ViolationResponse.EMERGENCY_STOP


class TestConstraintResult:
    def test_creation(self, mod):
        cr = mod.ConstraintResult(
            constraint_id="SC-001",
            constraint_name="Force limit",
            satisfied=True,
            severity=mod.ConstraintSeverity.CRITICAL,
            violation_response=mod.ViolationResponse.EMERGENCY_STOP,
            message="Force within limits",
        )
        assert cr.satisfied is True
        assert cr.constraint_id == "SC-001"


class TestAgentAction:
    def test_creation(self, mod):
        action = mod.AgentAction(
            action_id="ACT-001",
            category=mod.ActionCategory.MANIPULATION,
            description="Grasp specimen",
            parameters={"gripper_force": 5.0},
            requires_confirmation=True,
        )
        assert action.category == mod.ActionCategory.MANIPULATION
        assert action.requires_confirmation is True


class TestSafetyVerdict:
    def test_approved(self, mod):
        verdict = mod.SafetyVerdict(
            action_id="ACT-001",
            approved=True,
            constraint_results=[],
            blocking_violations=[],
        )
        assert verdict.approved is True

    def test_rejected_with_violations(self, mod):
        violation = mod.ConstraintResult(
            constraint_id="SC-001",
            constraint_name="Force limit",
            satisfied=False,
            severity=mod.ConstraintSeverity.CRITICAL,
            violation_response=mod.ViolationResponse.EMERGENCY_STOP,
            message="Force exceeded",
        )
        verdict = mod.SafetyVerdict(
            action_id="ACT-002",
            approved=False,
            constraint_results=[violation],
            blocking_violations=[violation],
        )
        assert verdict.approved is False
        assert len(verdict.blocking_violations) == 1
