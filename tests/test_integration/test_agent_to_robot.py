"""Integration test: Agentic AI -> Robot safety workflow.

Validates the flow from MCP server commands through safety checking
and robot control.
"""

from __future__ import annotations

import asyncio

import numpy as np

from tests.conftest import load_module


def run_async(coro):
    """Run an async coroutine synchronously for testing."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestAgentToRobotWorkflow:
    def test_mcp_command_through_safety_check(self):
        mcp_mod = load_module(
            "mcp_clinical_robotics",
            "agentic-ai/examples-agentic-ai/01_mcp_clinical_robotics_server.py",
        )
        safety_mod = load_module(
            "safety_monitoring",
            "examples-new/01_realtime_safety_monitoring.py",
        )

        # Create MCP server and send command
        server = mcp_mod.ClinicalRoboticsMCPServer(trial_id="INT-SAFETY-001")
        result = run_async(
            server.handle_tool_call(
                "send_robot_command",
                {
                    "command_type": "move_to_pose",
                    "target_pose": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                    "max_velocity": 0.05,
                },
            )
        )
        assert result["status"] in ("pending_confirmation", "accepted")

        # Validate through safety monitor
        limits = safety_mod.SafetyLimits()
        monitor = safety_mod.SafetyMonitor(limits)
        ft = safety_mod.ForceTorqueReading(
            force_xyz_n=np.zeros(3),
            torque_xyz_nm=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        state = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.array([0.01, 0.0, 0.0]),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=ft,
            timestamp_ns=1_000_000_000,
        )
        safe_pos, safe_vel, level = monitor.check_and_filter(
            np.array([0.0, 0.0, -0.15]),
            np.array([0.01, 0.0, 0.0]),
            state,
        )
        assert level == safety_mod.SafetyLevel.NOMINAL

    def test_mcp_audit_trail_populated(self):
        mcp_mod = load_module(
            "mcp_clinical_robotics",
            "agentic-ai/examples-agentic-ai/01_mcp_clinical_robotics_server.py",
        )
        server = mcp_mod.ClinicalRoboticsMCPServer(trial_id="INT-AUDIT-001")
        run_async(server.handle_tool_call("get_robot_telemetry", {}))
        run_async(server.handle_tool_call("get_patient_vitals", {}))
        assert len(server.audit_trail) >= 2
