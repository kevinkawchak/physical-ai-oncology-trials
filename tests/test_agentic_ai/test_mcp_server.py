"""Unit tests for agentic-ai/examples-agentic-ai/01_mcp_clinical_robotics_server.py.

Tests cover RobotMode, ProcedurePhase enums, RobotTelemetry dataclass,
ClinicalRoboticsMCPServer tool/resource definitions, commands, and audit trail.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.conftest import load_module

MOD_PATH = "agentic-ai/examples-agentic-ai/01_mcp_clinical_robotics_server.py"


@pytest.fixture()
def mod():
    return load_module("mcp_clinical_robotics", MOD_PATH)


@pytest.fixture()
def server(mod):
    return mod.ClinicalRoboticsMCPServer(trial_id="TEST-MCP-001")


def _run(coro):
    """Run an async coroutine synchronously for testing."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestEnums:
    def test_robot_mode(self, mod):
        names = [e.name for e in mod.RobotMode]
        for mode in ("IDLE", "HOMING", "MANUAL", "AUTONOMOUS", "EMERGENCY_STOP"):
            assert mode in names

    def test_procedure_phase(self, mod):
        names = [e.name for e in mod.ProcedurePhase]
        assert "PRE_OPERATIVE" in names
        assert "POST_OPERATIVE" in names


class TestRobotTelemetry:
    def test_creation(self, mod):
        telem = mod.RobotTelemetry(
            joint_positions=[0.0] * 7,
            joint_velocities=[0.0] * 7,
            joint_torques=[0.0] * 7,
            end_effector_pose=[0.0, 0.0, -0.15, 0.0, 0.0, 0.0],
            force_torque=[0.0] * 6,
            timestamp=1_000_000,
            mode=mod.RobotMode.IDLE,
        )
        assert telem.mode == mod.RobotMode.IDLE
        assert len(telem.joint_positions) == 7


class TestToolDefinitions:
    def test_tools_count(self, mod):
        tools = mod.ClinicalRoboticsMCPTools.get_tool_definitions()
        assert len(tools) >= 5

    def test_tool_has_required_fields(self, mod):
        for tool in mod.ClinicalRoboticsMCPTools.get_tool_definitions():
            assert "name" in tool
            assert "description" in tool


class TestResourceDefinitions:
    def test_resources_exist(self, mod):
        resources = mod.ClinicalRoboticsMCPResources.get_resource_definitions()
        assert len(resources) >= 2

    def test_resource_uris(self, mod):
        for res in mod.ClinicalRoboticsMCPResources.get_resource_definitions():
            assert "uri" in res


class TestServerTelemetry:
    def test_get_telemetry(self, server, mod):
        result = _run(server.handle_tool_call("get_robot_telemetry", {}))
        assert result is not None

    def test_get_workspace_boundaries(self, server):
        result = _run(server.handle_tool_call("get_workspace_boundaries", {}))
        assert result is not None


class TestServerCommands:
    def test_valid_command(self, server):
        result = _run(
            server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to_pose",
                    "target_pose": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    "max_velocity": 0.05,
                },
            )
        )
        assert result is not None

    def test_out_of_bounds_rejected(self, server):
        result = _run(
            server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to_pose",
                    "target_pose": [10.0, 10.0, 10.0, 0.0, 0.0, 0.0],
                    "max_velocity": 0.05,
                },
            )
        )
        assert result["status"] == "rejected"

    def test_velocity_clamping(self, server):
        result = _run(
            server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to_pose",
                    "target_pose": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    "max_velocity": 5.0,
                },
            )
        )
        assert result is not None
        # Velocity should be clamped to the workspace max of 0.1 m/s
        assert result["max_velocity_m_s"] <= 0.1


class TestAuditTrail:
    def test_audit_grows(self, server):
        initial_count = len(server.audit_trail)
        _run(server.handle_tool_call("get_robot_telemetry", {}))
        _run(server.handle_tool_call("get_robot_telemetry", {}))
        assert len(server.audit_trail) > initial_count

    def test_export_audit_json(self, server):
        _run(server.handle_tool_call("get_robot_telemetry", {}))
        export = server.export_audit_trail()
        data = json.loads(export)
        assert "trial_id" in data
        assert isinstance(data["entries"], list)
