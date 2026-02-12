"""Integration tests: Robot command -> Safety monitoring -> Emergency stop.

Tests the cross-module flow from MCP clinical robotics server through the
realtime safety monitoring module, verifying that unsafe commands are
blocked and safety violations are captured in event logs.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def safety_mod():
    return load_module(
        "safety_monitoring",
        "examples-new/01_realtime_safety_monitoring.py",
    )


@pytest.fixture()
def mcp_mod():
    return load_module(
        "mcp_clinical_robotics",
        "agentic-ai/examples-agentic-ai/01_mcp_clinical_robotics_server.py",
    )


@pytest.fixture()
def mcp_server(mcp_mod):
    return mcp_mod.ClinicalRoboticsMCPServer(trial_id="SAFETY-TEST-001")


@pytest.fixture()
def monitor(safety_mod):
    limits = safety_mod.SafetyLimits()
    return safety_mod.SafetyMonitor(limits)


def _make_robot_state(safety_mod, ee_pos, ee_vel=None, force_xyz=None, timestamp_ns=1_000_000_000):
    """Create a RobotState with given end-effector values."""
    ft = safety_mod.ForceTorqueReading(
        force_xyz_n=np.array(force_xyz if force_xyz is not None else [0.0, 0.0, 0.0]),
        torque_xyz_nm=np.zeros(3),
        timestamp_ns=timestamp_ns,
    )
    return safety_mod.RobotState(
        joint_positions_rad=np.zeros(7),
        joint_velocities_rad_s=np.zeros(7),
        ee_position_m=np.array(ee_pos),
        ee_velocity_m_s=np.array(ee_vel if ee_vel is not None else [0.0, 0.0, 0.0]),
        ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        force_torque=ft,
        timestamp_ns=timestamp_ns,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRobotToSafety:
    """Integration: Robot commands -> Safety monitoring -> Emergency stop."""

    def test_safety_monitor_blocks_unsafe_command(self, safety_mod, monitor):
        """SafetyMonitor rejects command with end-effector outside workspace."""
        # Far outside typical workspace (workspace_center is [0,0,-0.15], radius 0.15)
        state = _make_robot_state(safety_mod, ee_pos=[10.0, 10.0, 10.0])
        cmd_pos = np.array([10.0, 10.0, 10.0])
        cmd_vel = np.array([0.0, 0.0, 0.0])
        _, _, level = monitor.check_and_filter(cmd_pos, cmd_vel, state)
        # Should be at least WARNING or CRITICAL for out-of-workspace
        assert level.value >= safety_mod.SafetyLevel.WARNING.value

    def test_mcp_keepout_zone_enforced(self, mcp_mod, mcp_server):
        """MCP server rejects commands targeting keepout zones."""
        mcp_server._keepout_zones = [{"center": [0.1, 0.1, 0.1], "radius_m": 0.05, "label": "critical_structure"}]
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            mcp_server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to",
                    "target_pose": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                    "requires_confirmation": False,
                },
            )
        )
        assert result["status"] == "rejected"
        assert "keep-out" in result.get("reason", "").lower() or "keep" in result.get("reason", "").lower()

    def test_force_limit_stops_robot(self, safety_mod, monitor):
        """Force exceeding limit triggers CRITICAL severity."""
        max_force = monitor.limits.force_max_n
        state = _make_robot_state(safety_mod, ee_pos=[0.0, 0.0, -0.15], force_xyz=[max_force + 5.0, 0.0, 0.0])
        cmd_pos = np.array([0.0, 0.0, -0.15])
        cmd_vel = np.array([0.0, 0.0, 0.0])
        _, _, level = monitor.check_and_filter(cmd_pos, cmd_vel, state)
        assert level.value >= safety_mod.SafetyLevel.CRITICAL.value

    def test_velocity_limit_enforced(self, safety_mod, monitor):
        """Velocity exceeding limit triggers a violation."""
        max_vel = monitor.limits.velocity_max_m_s
        high_vel = np.array([max_vel + 1.0, 0.0, 0.0])
        state = _make_robot_state(safety_mod, ee_pos=[0.0, 0.0, -0.15], ee_vel=[max_vel + 1.0, 0.0, 0.0])
        cmd_pos = np.array([0.0, 0.0, -0.15])
        _, _, level = monitor.check_and_filter(cmd_pos, high_vel, state)
        assert level.value >= safety_mod.SafetyLevel.WARNING.value

    def test_safety_reset_requires_auth(self, safety_mod, monitor):
        """reset_stop requires an authorization code; empty string is rejected."""
        # Trigger a critical event first
        state = _make_robot_state(safety_mod, ee_pos=[0.0, 0.0, -0.15], force_xyz=[100.0, 0.0, 0.0])
        cmd_pos = np.array([0.0, 0.0, -0.15])
        cmd_vel = np.zeros(3)
        monitor.check_and_filter(cmd_pos, cmd_vel, state)
        # Try to reset without valid code (empty string -> rejected)
        monitor.reset_stop(authorization_code="")
        assert monitor._is_stopped  # Should still be stopped

    def test_event_log_captures_violations(self, safety_mod, monitor):
        """All safety events logged in the monitor's event log."""
        state = _make_robot_state(safety_mod, ee_pos=[0.0, 0.0, -0.15], force_xyz=[100.0, 0.0, 0.0])
        cmd_pos = np.array([0.0, 0.0, -0.15])
        cmd_vel = np.zeros(3)
        monitor.check_and_filter(cmd_pos, cmd_vel, state)
        events = monitor.get_event_log()
        assert len(events) >= 1

    def test_mcp_workspace_outside_rejected(self, mcp_mod, mcp_server):
        """MCP server rejects target poses outside workspace boundaries."""
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            mcp_server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to",
                    "target_pose": [10.0, 10.0, 10.0, 0.0, 0.0, 0.0],
                    "requires_confirmation": False,
                },
            )
        )
        assert result["status"] == "rejected"

    def test_nominal_state_passes_safety(self, safety_mod, monitor):
        """A nominal state within all limits passes safety checks."""
        state = _make_robot_state(
            safety_mod,
            ee_pos=[0.0, 0.0, -0.15],
            ee_vel=[0.01, 0.0, 0.0],
            force_xyz=[0.5, 0.0, 0.0],
        )
        cmd_pos = np.array([0.0, 0.0, -0.15])
        cmd_vel = np.array([0.01, 0.0, 0.0])
        _, _, level = monitor.check_and_filter(cmd_pos, cmd_vel, state)
        assert level == safety_mod.SafetyLevel.NOMINAL

    def test_mcp_valid_command_accepted(self, mcp_mod, mcp_server):
        """A valid command within workspace is accepted or pending confirmation."""
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            mcp_server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to",
                    "target_pose": [0.1, 0.1, 0.2, 0.0, 0.0, 0.0],
                    "requires_confirmation": True,
                },
            )
        )
        assert result["status"] in ("accepted", "pending_confirmation")

    def test_safety_level_enum_ordering(self, safety_mod):
        """SafetyLevel enum has expected ordering (NOMINAL < WARNING < CRITICAL < EMERGENCY)."""
        assert safety_mod.SafetyLevel.NOMINAL.value < safety_mod.SafetyLevel.WARNING.value
        assert safety_mod.SafetyLevel.WARNING.value < safety_mod.SafetyLevel.CRITICAL.value
        assert safety_mod.SafetyLevel.CRITICAL.value < safety_mod.SafetyLevel.EMERGENCY.value

    def test_mcp_audit_logs_rejected_command(self, mcp_mod, mcp_server):
        """Rejected commands are still captured in the audit trail."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            mcp_server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to",
                    "target_pose": [10.0, 10.0, 10.0, 0.0, 0.0, 0.0],
                    "requires_confirmation": False,
                },
            )
        )
        assert len(mcp_server.audit_trail) >= 1
