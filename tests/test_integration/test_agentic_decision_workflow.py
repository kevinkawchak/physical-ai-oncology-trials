"""Integration test: Agent Decision -> Action -> Audit Trail.

Validates the agentic decision pipeline: safety constraint checking,
action approval, and audit trail generation.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from tests.conftest import load_module


def _run(coro):
    """Run an async coroutine synchronously for testing."""
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture()
def safety_mod():
    return load_module("safety_executor", "agentic-ai/examples-agentic-ai/05_safety_constrained_agent_executor.py")


@pytest.fixture()
def mcp_mod():
    return load_module("mcp_clinical_robotics", "agentic-ai/examples-agentic-ai/01_mcp_clinical_robotics_server.py")


class TestAgenticDecisionWorkflow:
    def test_constraint_check_then_action(self, safety_mod):
        """Define a constraint, create an action, verify approval logic."""
        constraint = safety_mod.SafetyConstraint(
            constraint_id="INTEG-SC-001",
            name="Force limit",
            description="Force must be under 10 N",
            constraint_type=safety_mod.ConstraintType.PRE_CONDITION,
            severity=safety_mod.ConstraintSeverity.BLOCKING,
            violation_response=safety_mod.ViolationResponse.REJECT_ACTION,
            predicate=lambda state: state.get("force", 0) <= 10.0,
            applicable_actions=[safety_mod.ActionCategory.MANIPULATION],
        )
        # Safe state
        result = constraint.check({"force": 5.0})
        assert result.satisfied is True

        # Unsafe state
        result = constraint.check({"force": 15.0})
        assert result.satisfied is False
        assert result.violation_response == safety_mod.ViolationResponse.REJECT_ACTION

    def test_mcp_server_action_audit_trail(self, mcp_mod):
        """MCP server actions should generate audit trail entries."""
        server = mcp_mod.ClinicalRoboticsMCPServer(trial_id="INTEG-TRIAL-001")
        initial_len = len(server.audit_trail)

        _run(server.handle_tool_call("get_robot_telemetry", {}))
        _run(server.handle_tool_call("get_patient_vitals", {}))
        _run(
            server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to_pose",
                    "target_pose": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    "max_velocity": 0.03,
                },
            )
        )

        assert len(server.audit_trail) > initial_len
        export = server.export_audit_trail()
        data = json.loads(export)
        assert "trial_id" in data
        assert isinstance(data["entries"], list)

    def test_full_decision_chain(self, safety_mod, mcp_mod):
        """Constraint check -> MCP command -> audit record."""
        # Step 1: Check safety
        constraint = safety_mod.SafetyConstraint(
            constraint_id="INTEG-CHAIN-001",
            name="Workspace check",
            description="Target must be in workspace",
            constraint_type=safety_mod.ConstraintType.PRE_CONDITION,
            severity=safety_mod.ConstraintSeverity.BLOCKING,
            violation_response=safety_mod.ViolationResponse.REJECT_ACTION,
            predicate=lambda state: state.get("in_workspace", False),
            applicable_actions=[safety_mod.ActionCategory.NAVIGATION],
        )
        check_result = constraint.check({"in_workspace": True})
        assert check_result.satisfied is True

        # Step 2: Execute command via MCP
        server = mcp_mod.ClinicalRoboticsMCPServer(trial_id="INTEG-CHAIN")
        result = _run(
            server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to_pose",
                    "target_pose": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    "max_velocity": 0.03,
                },
            )
        )
        assert result is not None

        # Step 3: Verify audit trail
        assert len(server.audit_trail) > 0
