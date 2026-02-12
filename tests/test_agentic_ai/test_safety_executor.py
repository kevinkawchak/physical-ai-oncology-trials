"""Tests for agentic-ai/examples-agentic-ai/05_safety_constrained_agent_executor.py"""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module(
        "safety_constrained_agent_executor",
        "agentic-ai/examples-agentic-ai/05_safety_constrained_agent_executor.py",
    )


@pytest.fixture()
def nominal_state():
    """State that passes all standard constraints."""
    return {
        "patient_identity_verified": True,
        "timeout_completed": True,
        "heart_rate_bpm": 75,
        "spo2_percent": 98,
        "systolic_bp": 118,
        "current_force_n": 2.0,
        "force_rate_n_s": 0.5,
        "max_force_n": 10.0,
        "max_force_rate_n_s": 5.0,
        "current_velocity_m_s": 0.05,
        "max_velocity_m_s": 0.1,
        "approach_velocity_m_s": 0.02,
        "distance_to_target_mm": 50.0,
        "commanded_velocity_m_s": 0.05,
        "target_position": [0.05, -0.02, 0.15],
        "workspace_limits": {
            "x_range_m": [-0.3, 0.3],
            "y_range_m": [-0.3, 0.3],
            "z_range_m": [0.0, 0.4],
        },
        "keepout_zones": [],
        "instrument_path_clear": True,
        "energy_activation_confirmed": False,
    }


# ============================================================
# Enum tests
# ============================================================


class TestConstraintType:
    def test_values(self, mod):
        assert mod.ConstraintType.PRE_CONDITION.value == "pre_condition"
        assert mod.ConstraintType.POST_CONDITION.value == "post_condition"
        assert mod.ConstraintType.INVARIANT.value == "invariant"
        assert mod.ConstraintType.SAFETY_GATE.value == "safety_gate"

    def test_member_count(self, mod):
        assert len(mod.ConstraintType) == 4


class TestConstraintSeverity:
    def test_values(self, mod):
        assert mod.ConstraintSeverity.ADVISORY.value == "advisory"
        assert mod.ConstraintSeverity.BLOCKING.value == "blocking"
        assert mod.ConstraintSeverity.CRITICAL.value == "critical"
        assert mod.ConstraintSeverity.EMERGENCY.value == "emergency"

    def test_ordering_by_list_index(self, mod):
        members = list(mod.ConstraintSeverity)
        assert members.index(mod.ConstraintSeverity.ADVISORY) < members.index(mod.ConstraintSeverity.EMERGENCY)

    def test_member_count(self, mod):
        assert len(mod.ConstraintSeverity) == 4


class TestViolationResponse:
    def test_values(self, mod):
        assert mod.ViolationResponse.LOG_ONLY.value == "log_only"
        assert mod.ViolationResponse.REJECT_ACTION.value == "reject_action"
        assert mod.ViolationResponse.ROLLBACK.value == "rollback"
        assert mod.ViolationResponse.EMERGENCY_STOP.value == "emergency_stop"
        assert mod.ViolationResponse.HUMAN_REVIEW.value == "human_review"

    def test_member_count(self, mod):
        assert len(mod.ViolationResponse) == 5


class TestActionCategory:
    def test_values(self, mod):
        assert mod.ActionCategory.OBSERVATION.value == "observation"
        assert mod.ActionCategory.NAVIGATION.value == "navigation"
        assert mod.ActionCategory.MANIPULATION.value == "manipulation"
        assert mod.ActionCategory.ENERGY.value == "energy_delivery"
        assert mod.ActionCategory.CRITICAL.value == "critical_action"

    def test_member_count(self, mod):
        assert len(mod.ActionCategory) == 5


# ============================================================
# Dataclass tests
# ============================================================


class TestSafetyConstraint:
    def test_creation(self, mod):
        sc = mod.SafetyConstraint(
            constraint_id="TEST-001",
            name="Test Constraint",
            description="A test constraint",
            constraint_type=mod.ConstraintType.PRE_CONDITION,
            severity=mod.ConstraintSeverity.BLOCKING,
            violation_response=mod.ViolationResponse.REJECT_ACTION,
        )
        assert sc.constraint_id == "TEST-001"
        assert sc.name == "Test Constraint"

    def test_check_satisfied(self, mod):
        sc = mod.SafetyConstraint(
            constraint_id="T-001",
            name="Always True",
            description="Always passes",
            constraint_type=mod.ConstraintType.PRE_CONDITION,
            severity=mod.ConstraintSeverity.BLOCKING,
            violation_response=mod.ViolationResponse.REJECT_ACTION,
            predicate=lambda state: True,
        )
        result = sc.check({})
        assert result.satisfied is True
        assert "PASSED" in result.message

    def test_check_violated(self, mod):
        sc = mod.SafetyConstraint(
            constraint_id="T-002",
            name="Always False",
            description="Always fails",
            constraint_type=mod.ConstraintType.PRE_CONDITION,
            severity=mod.ConstraintSeverity.CRITICAL,
            violation_response=mod.ViolationResponse.EMERGENCY_STOP,
            predicate=lambda state: False,
        )
        result = sc.check({})
        assert result.satisfied is False
        assert "VIOLATED" in result.message
        assert result.violation_response == mod.ViolationResponse.EMERGENCY_STOP

    def test_check_exception_handled(self, mod):
        sc = mod.SafetyConstraint(
            constraint_id="T-003",
            name="Error Constraint",
            description="Throws error",
            constraint_type=mod.ConstraintType.INVARIANT,
            severity=mod.ConstraintSeverity.BLOCKING,
            violation_response=mod.ViolationResponse.ROLLBACK,
            predicate=lambda state: state["missing_key"],
        )
        result = sc.check({})
        assert result.satisfied is False
        assert "EVALUATION ERROR" in result.message


class TestConstraintResult:
    def test_satisfied_state(self, mod):
        cr = mod.ConstraintResult(
            constraint_id="CR-001",
            constraint_name="Test",
            satisfied=True,
            severity=mod.ConstraintSeverity.BLOCKING,
            violation_response=mod.ViolationResponse.LOG_ONLY,
            message="Passed",
        )
        assert cr.satisfied is True

    def test_violated_state(self, mod):
        cr = mod.ConstraintResult(
            constraint_id="CR-002",
            constraint_name="Test",
            satisfied=False,
            severity=mod.ConstraintSeverity.CRITICAL,
            violation_response=mod.ViolationResponse.REJECT_ACTION,
            message="Failed",
        )
        assert cr.satisfied is False
        assert cr.severity == mod.ConstraintSeverity.CRITICAL


class TestAgentAction:
    def test_creation(self, mod):
        action = mod.AgentAction(
            action_id="ACT-001",
            category=mod.ActionCategory.NAVIGATION,
            description="Move to position",
        )
        assert action.action_id == "ACT-001"
        assert action.category == mod.ActionCategory.NAVIGATION
        assert action.requires_confirmation is False

    def test_creation_with_rollback(self, mod):
        action = mod.AgentAction(
            action_id="ACT-002",
            category=mod.ActionCategory.MANIPULATION,
            description="Grasp tissue",
            rollback_action="release_tissue",
        )
        assert action.rollback_action == "release_tissue"


class TestSafetyVerdict:
    def test_creation(self, mod):
        v = mod.SafetyVerdict(action_id="ACT-001", approved=True)
        assert v.approved is True
        assert v.safety_score == 1.0

    def test_to_dict(self, mod):
        v = mod.SafetyVerdict(
            action_id="ACT-001",
            approved=False,
            safety_score=0.5,
            requires_human_approval=True,
        )
        d = v.to_dict()
        assert d["approved"] is False
        assert d["safety_score"] == 0.5
        assert d["requires_human_approval"] is True


# ============================================================
# OncologyRoboticsConstraintLibrary tests
# ============================================================


class TestOncologyRoboticsConstraintLibrary:
    def test_get_all_constraints_non_empty(self, mod):
        constraints = mod.OncologyRoboticsConstraintLibrary.get_all_constraints()
        assert len(constraints) > 0

    def test_constraint_count(self, mod):
        constraints = mod.OncologyRoboticsConstraintLibrary.get_all_constraints()
        # 2 workspace + 2 force + 2 velocity + 3 clinical + 2 energy = 11
        assert len(constraints) == 11

    def test_force_limit_constraint_exists(self, mod):
        constraints = mod.OncologyRoboticsConstraintLibrary.force_constraints()
        assert len(constraints) == 2
        ids = [c.constraint_id for c in constraints]
        assert "FC-001" in ids

    def test_workspace_constraint_exists(self, mod):
        constraints = mod.OncologyRoboticsConstraintLibrary.workspace_constraints()
        assert len(constraints) == 2
        ids = [c.constraint_id for c in constraints]
        assert "WS-001" in ids

    def test_velocity_constraint_exists(self, mod):
        constraints = mod.OncologyRoboticsConstraintLibrary.velocity_constraints()
        assert len(constraints) == 2
        ids = [c.constraint_id for c in constraints]
        assert "VC-001" in ids

    def test_clinical_constraints(self, mod):
        constraints = mod.OncologyRoboticsConstraintLibrary.clinical_constraints()
        assert len(constraints) == 3

    def test_energy_constraints(self, mod):
        constraints = mod.OncologyRoboticsConstraintLibrary.energy_constraints()
        assert len(constraints) == 2


# ============================================================
# SafetyConstrainedExecutor tests
# ============================================================


class TestSafetyConstrainedExecutor:
    def test_init_default_constraints(self, mod):
        executor = mod.SafetyConstrainedExecutor()
        stats = executor.get_statistics()
        assert stats["total_constraints"] == 11

    def test_evaluate_action_nominal_passes(self, mod, nominal_state):
        executor = mod.SafetyConstrainedExecutor()
        action = mod.AgentAction(
            action_id="SAFE-001",
            category=mod.ActionCategory.NAVIGATION,
            description="Move to safe position",
        )
        verdict = executor.evaluate_action(action, nominal_state)
        assert verdict.approved is True
        assert verdict.safety_score > 0.0

    def test_evaluate_action_workspace_violation(self, mod, nominal_state):
        executor = mod.SafetyConstrainedExecutor()
        action = mod.AgentAction(
            action_id="OOB-001",
            category=mod.ActionCategory.NAVIGATION,
            description="Move outside workspace",
        )
        state = {**nominal_state, "target_position": [0.5, 0.0, 0.2]}
        verdict = executor.evaluate_action(action, state)
        assert verdict.approved is False
        assert len(verdict.blocking_violations) > 0

    def test_evaluate_action_force_violation(self, mod, nominal_state):
        executor = mod.SafetyConstrainedExecutor()
        action = mod.AgentAction(
            action_id="FORCE-001",
            category=mod.ActionCategory.MANIPULATION,
            description="Continue retraction",
        )
        state = {**nominal_state, "current_force_n": 15.0}
        verdict = executor.evaluate_action(action, state)
        assert verdict.approved is False

    def test_evaluate_action_missing_patient_verification(self, mod, nominal_state):
        executor = mod.SafetyConstrainedExecutor()
        action = mod.AgentAction(
            action_id="NOVERIFY-001",
            category=mod.ActionCategory.MANIPULATION,
            description="Begin tissue dissection",
        )
        state = {**nominal_state, "patient_identity_verified": False}
        verdict = executor.evaluate_action(action, state)
        assert verdict.approved is False

    def test_energy_action_requires_human_approval(self, mod, nominal_state):
        executor = mod.SafetyConstrainedExecutor()
        action = mod.AgentAction(
            action_id="ENERGY-001",
            category=mod.ActionCategory.ENERGY,
            description="Activate cautery",
        )
        state = {**nominal_state, "energy_activation_confirmed": False}
        verdict = executor.evaluate_action(action, state)
        assert verdict.requires_human_approval is True
        assert verdict.approved is False

    def test_observation_action_unrestricted(self, mod, nominal_state):
        executor = mod.SafetyConstrainedExecutor()
        action = mod.AgentAction(
            action_id="OBS-001",
            category=mod.ActionCategory.OBSERVATION,
            description="Query robot state",
        )
        verdict = executor.evaluate_action(action, nominal_state)
        assert verdict.approved is True

    def test_execute_with_safety_approved(self, mod, nominal_state):
        executor = mod.SafetyConstrainedExecutor()
        action = mod.AgentAction(
            action_id="EXEC-001",
            category=mod.ActionCategory.NAVIGATION,
            description="Move to position",
        )

        def mock_execute(a):
            return {"status": "executed", "action_id": a.action_id, "new_state": {}}

        result = executor.execute_with_safety(action, nominal_state, mock_execute)
        assert result["status"] == "executed"
        assert "safety_score" in result

    def test_execute_with_safety_rejected(self, mod, nominal_state):
        executor = mod.SafetyConstrainedExecutor()
        action = mod.AgentAction(
            action_id="EXEC-002",
            category=mod.ActionCategory.NAVIGATION,
            description="Move outside workspace",
        )
        state = {**nominal_state, "target_position": [0.5, 0.0, 0.2]}

        def mock_execute(a):
            return {"status": "executed", "action_id": a.action_id, "new_state": {}}

        result = executor.execute_with_safety(action, state, mock_execute)
        assert result["status"] == "rejected"

    def test_statistics_increments(self, mod, nominal_state):
        executor = mod.SafetyConstrainedExecutor()
        action = mod.AgentAction(
            action_id="STAT-001",
            category=mod.ActionCategory.NAVIGATION,
            description="Move",
        )
        executor.evaluate_action(action, nominal_state)
        stats = executor.get_statistics()
        assert stats["total_actions_evaluated"] == 1

    def test_audit_log_populated(self, mod, nominal_state):
        executor = mod.SafetyConstrainedExecutor()
        action = mod.AgentAction(
            action_id="AUDIT-001",
            category=mod.ActionCategory.NAVIGATION,
            description="Move",
        )
        executor.evaluate_action(action, nominal_state)
        log = executor.get_audit_log()
        assert len(log) == 1
        assert log[0]["action_id"] == "AUDIT-001"

    def test_format_verdict(self, mod, nominal_state):
        executor = mod.SafetyConstrainedExecutor()
        action = mod.AgentAction(
            action_id="FMT-001",
            category=mod.ActionCategory.NAVIGATION,
            description="Move",
        )
        verdict = executor.evaluate_action(action, nominal_state)
        formatted = executor.format_verdict(verdict)
        assert isinstance(formatted, str)
        assert "Safety Verdict" in formatted

    def test_keepout_zone_violation(self, mod, nominal_state):
        executor = mod.SafetyConstrainedExecutor()
        action = mod.AgentAction(
            action_id="KO-001",
            category=mod.ActionCategory.MANIPULATION,
            description="Dissect near vessel",
        )
        state = {
            **nominal_state,
            "target_position": [0.1, 0.1, 0.2],
            "keepout_zones": [
                {"center": [0.1, 0.1, 0.2], "radius_m": 0.02, "label": "pulmonary_artery"},
            ],
        }
        verdict = executor.evaluate_action(action, state)
        assert verdict.approved is False
