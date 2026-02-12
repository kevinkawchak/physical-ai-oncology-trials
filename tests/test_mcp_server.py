"""Tests for agentic-ai/examples-agentic-ai/01_mcp_clinical_robotics_server.py

Covers ClinicalRoboticsMCPServer tool handlers, resource handlers,
audit trail, and data models.
"""

from __future__ import annotations

import asyncio
import json


def run_async(coro):
    """Run an async coroutine synchronously for testing."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Data models ───────────────────────────────────────────────────────────


class TestDataModels:
    def test_robot_telemetry_to_dict(self, mcp_mod):
        t = mcp_mod.RobotTelemetry()
        d = t.to_dict()
        assert "joint_positions_rad" in d
        assert "mode" in d
        assert d["mode"] == "idle"

    def test_patient_vitals_to_dict(self, mcp_mod):
        v = mcp_mod.PatientVitals()
        d = v.to_dict()
        assert d["heart_rate_bpm"] == 72.0
        assert "blood_pressure" in d

    def test_robot_mode_enum(self, mcp_mod):
        assert mcp_mod.RobotMode.IDLE.value == "idle"
        assert mcp_mod.RobotMode.EMERGENCY_STOP.value == "emergency_stop"

    def test_procedure_phase_enum(self, mcp_mod):
        assert mcp_mod.ProcedurePhase.PRE_OPERATIVE.value == "pre_operative"


# ── MCP Tool Definitions ─────────────────────────────────────────────────


class TestToolDefinitions:
    def test_tool_count(self, mcp_mod):
        tools = mcp_mod.ClinicalRoboticsMCPTools.get_tool_definitions()
        assert len(tools) >= 7

    def test_tools_have_required_fields(self, mcp_mod):
        tools = mcp_mod.ClinicalRoboticsMCPTools.get_tool_definitions()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_safety_levels_assigned(self, mcp_mod):
        tools = mcp_mod.ClinicalRoboticsMCPTools.get_tool_definitions()
        safety_levels = {t["name"]: t.get("safety_level") for t in tools}
        assert safety_levels["get_robot_telemetry"] == "read_only"
        assert safety_levels["send_robot_command"] == "critical"


# ── MCP Resource Definitions ─────────────────────────────────────────────


class TestResourceDefinitions:
    def test_resource_count(self, mcp_mod):
        resources = mcp_mod.ClinicalRoboticsMCPResources.get_resource_definitions()
        assert len(resources) == 5

    def test_resources_have_uris(self, mcp_mod):
        resources = mcp_mod.ClinicalRoboticsMCPResources.get_resource_definitions()
        for r in resources:
            assert r["uri"].startswith("oncology://")


# ── Server – Telemetry ────────────────────────────────────────────────────


class TestServerTelemetry:
    def test_get_telemetry(self, mcp_server):
        result = run_async(mcp_server.handle_tool_call("get_robot_telemetry", {}))
        assert "mode" in result
        assert result["safety_status"] == "nominal"

    def test_get_telemetry_with_derivatives(self, mcp_server):
        result = run_async(mcp_server.handle_tool_call("get_robot_telemetry", {"include_derivatives": True}))
        assert "mode" in result


# ── Server – Workspace ────────────────────────────────────────────────────


class TestServerWorkspace:
    def test_get_workspace(self, mcp_server):
        result = run_async(mcp_server.handle_tool_call("get_workspace_boundaries", {}))
        assert "workspace_limits" in result
        assert "keepout_zones" in result


# ── Server – Robot Commands ───────────────────────────────────────────────


class TestServerCommands:
    def test_valid_command_accepted(self, mcp_server):
        result = run_async(
            mcp_server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to_pose",
                    "target_pose": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                    "max_velocity": 0.05,
                },
            )
        )
        assert result["status"] in ("pending_confirmation", "accepted")

    def test_out_of_bounds_rejected(self, mcp_server):
        result = run_async(
            mcp_server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to_pose",
                    "target_pose": [5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
                },
            )
        )
        assert result["status"] == "rejected"

    def test_keepout_zone_rejection(self, mcp_server):
        mcp_server._keepout_zones.append({"label": "test_zone", "center": [0.1, 0.1, 0.1], "radius_m": 0.05})
        result = run_async(
            mcp_server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to_pose",
                    "target_pose": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                },
            )
        )
        assert result["status"] == "rejected"
        assert "keep-out" in result["reason"]

    def test_velocity_clamped(self, mcp_server):
        result = run_async(
            mcp_server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to_pose",
                    "target_pose": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                    "max_velocity": 999.0,
                },
            )
        )
        assert result.get("max_velocity_m_s", 999) <= mcp_server._workspace_limits["max_velocity_m_s"]


# ── Server – DICOM ────────────────────────────────────────────────────────


class TestServerDICOM:
    def test_query_dicom(self, mcp_server):
        result = run_async(
            mcp_server.handle_tool_call(
                "query_dicom_study",
                {"study_uid": "1.2.3.4", "include_segmentation": True},
            )
        )
        assert result["patient_id"] == "[DE-IDENTIFIED]"
        assert "segmentation" in result

    def test_tumor_coordinates(self, mcp_server):
        result = run_async(mcp_server.handle_tool_call("get_tumor_coordinates", {"include_margins": True}))
        assert "tumor_centroid" in result
        assert "margin_surface" in result


# ── Server – Vitals ───────────────────────────────────────────────────────


class TestServerVitals:
    def test_get_vitals(self, mcp_server):
        result = run_async(mcp_server.handle_tool_call("get_patient_vitals", {}))
        assert result["heart_rate_bpm"] == 72.0
        assert "trends" in result
        assert "alerts" in result


# ── Server – Procedure Events ─────────────────────────────────────────────


class TestServerProcedureEvents:
    def test_log_event(self, mcp_server):
        result = run_async(
            mcp_server.handle_tool_call(
                "log_procedure_event",
                {"event_type": "clinical_observation", "description": "test event"},
            )
        )
        assert result["status"] == "logged"
        assert "event_id" in result

    def test_get_checklist_all_phases(self, mcp_server):
        result = run_async(mcp_server.handle_tool_call("get_procedure_checklist", {}))
        assert "sign_in" in result
        assert "time_out" in result
        assert "sign_out" in result

    def test_get_checklist_single_phase(self, mcp_server):
        result = run_async(mcp_server.handle_tool_call("get_procedure_checklist", {"phase": "sign_in"}))
        assert "items" in result
        assert result["total"] >= 5


# ── Server – Resources ────────────────────────────────────────────────────


class TestServerResources:
    def test_read_protocol(self, mcp_server):
        result = run_async(mcp_server.handle_resource_read("oncology://trial/protocol"))
        assert result["trial_id"] == "TEST-TRIAL-001"

    def test_read_safety_constraints(self, mcp_server):
        result = run_async(mcp_server.handle_resource_read("oncology://safety/constraints"))
        assert "workspace_limits" in result
        assert "force_limits" in result

    def test_unknown_resource(self, mcp_server):
        result = run_async(mcp_server.handle_resource_read("oncology://unknown"))
        assert "error" in result

    def test_unknown_tool(self, mcp_server):
        result = run_async(mcp_server.handle_tool_call("nonexistent_tool", {}))
        assert "error" in result


# ── Audit Trail ───────────────────────────────────────────────────────────


class TestAuditTrail:
    def test_audit_trail_grows(self, mcp_server):
        initial = len(mcp_server.audit_trail)
        run_async(mcp_server.handle_tool_call("get_robot_telemetry", {}))
        assert len(mcp_server.audit_trail) > initial

    def test_export_audit_trail_json(self, mcp_server):
        run_async(mcp_server.handle_tool_call("get_patient_vitals", {}))
        exported = mcp_server.export_audit_trail()
        data = json.loads(exported)
        assert data["trial_id"] == "TEST-TRIAL-001"
        assert len(data["entries"]) > 0
