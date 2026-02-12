"""Tests for agentic-ai/examples-agentic-ai/05_safety_constrained_agent_executor.py

Covers safety constraints, constraint checking, and the executor.
"""

from __future__ import annotations


# ── Enum values ──────────────────────────────────────────────────────────


class TestEnums:
    def test_constraint_types(self, safety_exec_mod):
        ct = safety_exec_mod.ConstraintType
        assert ct.PRE_CONDITION.value == "pre_condition"
        assert ct.INVARIANT.value == "invariant"
        assert ct.SAFETY_GATE.value == "safety_gate"

    def test_constraint_severity(self, safety_exec_mod):
        cs = safety_exec_mod.ConstraintSeverity
        assert cs.ADVISORY.value == "advisory"
        assert cs.BLOCKING.value == "blocking"
        assert cs.EMERGENCY.value == "emergency"

    def test_violation_responses(self, safety_exec_mod):
        vr = safety_exec_mod.ViolationResponse
        assert vr.LOG_ONLY.value == "log_only"
        assert vr.EMERGENCY_STOP.value == "emergency_stop"

    def test_action_categories(self, safety_exec_mod):
        ac = safety_exec_mod.ActionCategory
        assert ac.OBSERVATION.value == "observation"
        assert ac.MANIPULATION.value == "manipulation"
        assert ac.CRITICAL.value == "critical_action"


# ── SafetyConstraint dataclass ───────────────────────────────────────────


class TestSafetyConstraint:
    def test_creation(self, safety_exec_mod):
        constraint = safety_exec_mod.SafetyConstraint(
            constraint_id="SC-001",
            name="Force limit",
            description="Max contact force 5N",
            constraint_type=safety_exec_mod.ConstraintType.INVARIANT,
            severity=safety_exec_mod.ConstraintSeverity.BLOCKING,
            violation_response=safety_exec_mod.ViolationResponse.REJECT_ACTION,
        )
        assert constraint.constraint_id == "SC-001"

    def test_check_satisfied(self, safety_exec_mod):
        constraint = safety_exec_mod.SafetyConstraint(
            constraint_id="SC-002",
            name="Always pass",
            description="Test",
            constraint_type=safety_exec_mod.ConstraintType.INVARIANT,
            severity=safety_exec_mod.ConstraintSeverity.BLOCKING,
            violation_response=safety_exec_mod.ViolationResponse.REJECT_ACTION,
            predicate=lambda state: True,
        )
        result = constraint.check({"robot_position": [0, 0, 0]})
        assert result.satisfied is True

    def test_check_violated(self, safety_exec_mod):
        constraint = safety_exec_mod.SafetyConstraint(
            constraint_id="SC-003",
            name="Always fail",
            description="Test",
            constraint_type=safety_exec_mod.ConstraintType.PRE_CONDITION,
            severity=safety_exec_mod.ConstraintSeverity.CRITICAL,
            violation_response=safety_exec_mod.ViolationResponse.EMERGENCY_STOP,
            predicate=lambda state: False,
        )
        result = constraint.check({"robot_position": [0, 0, 0]})
        assert result.satisfied is False


# ── AgentAction dataclass ────────────────────────────────────────────────


class TestAgentAction:
    def test_creation(self, safety_exec_mod):
        action = safety_exec_mod.AgentAction(
            action_id="ACT-001",
            category=safety_exec_mod.ActionCategory.MANIPULATION,
            description="Move to target pose",
        )
        assert action.action_id == "ACT-001"
        assert action.requires_confirmation is False


# ── OncologyRoboticsConstraintLibrary ────────────────────────────────────


class TestConstraintLibrary:
    def test_get_all_constraints(self, safety_exec_mod):
        lib = safety_exec_mod.OncologyRoboticsConstraintLibrary
        constraints = lib.get_all_constraints()
        assert isinstance(constraints, list)
        assert len(constraints) > 0

    def test_workspace_constraints(self, safety_exec_mod):
        lib = safety_exec_mod.OncologyRoboticsConstraintLibrary
        ws = lib.workspace_constraints()
        assert isinstance(ws, list)
        assert len(ws) > 0


# ── SafetyConstrainedExecutor ────────────────────────────────────────────


class TestSafetyConstrainedExecutor:
    def test_creation_with_defaults(self, safety_exec_mod):
        executor = safety_exec_mod.SafetyConstrainedExecutor()
        assert executor is not None

    def test_evaluate_safe_action(self, safety_exec_mod):
        executor = safety_exec_mod.SafetyConstrainedExecutor()
        action = safety_exec_mod.AgentAction(
            action_id="EVAL-001",
            category=safety_exec_mod.ActionCategory.OBSERVATION,
            description="Read telemetry",
        )
        state = {
            "robot_position": [0.1, 0.1, 0.1],
            "robot_velocity": 0.05,
            "contact_force": 1.0,
            "workspace_limits": {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [-0.5, 0.5]},
            "keepout_zones": [],
            "procedure_phase": "operate",
            "emergency_stop_available": True,
        }
        verdict = executor.evaluate_action(action, state)
        assert isinstance(verdict, safety_exec_mod.SafetyVerdict)
        assert 0.0 <= verdict.safety_score <= 1.0

    def test_get_statistics(self, safety_exec_mod):
        executor = safety_exec_mod.SafetyConstrainedExecutor()
        stats = executor.get_statistics()
        assert isinstance(stats, dict)
        assert "total_actions_evaluated" in stats

    def test_get_audit_log(self, safety_exec_mod):
        executor = safety_exec_mod.SafetyConstrainedExecutor()
        log = executor.get_audit_log()
        assert isinstance(log, list)
