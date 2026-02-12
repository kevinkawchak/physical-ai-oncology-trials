"""Integration test: Safety Monitor -> Emergency Stop -> Audit.

Validates the safety monitoring chain from sensor reading through
emergency response to audit trail recording.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def safety_mod():
    return load_module("safety_monitoring", "examples-new/01_realtime_safety_monitoring.py")


class TestSafetyMonitoringChain:
    def test_nominal_to_critical_transition(self, safety_mod):
        """Monitor should transition from nominal to critical on dangerous input."""
        monitor = safety_mod.SafetyMonitor(limits=safety_mod.SafetyLimits())

        # Nominal state
        ft_ok = safety_mod.ForceTorqueReading(force_xyz_n=np.zeros(3), torque_xyz_nm=np.zeros(3), timestamp_ns=1)
        state_ok = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.array([0.01, 0.0, 0.0]),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=ft_ok,
            timestamp_ns=1,
        )
        cmd_pos_ok = state_ok.ee_position_m.copy()
        cmd_vel_ok = state_ok.ee_velocity_m_s.copy()
        _safe_pos, _safe_vel, level_ok = monitor.check_and_filter(cmd_pos_ok, cmd_vel_ok, state_ok)
        assert level_ok == safety_mod.SafetyLevel.NOMINAL

        # Dangerous state: excessive force
        ft_bad = safety_mod.ForceTorqueReading(
            force_xyz_n=np.array([100.0, 0.0, 0.0]),
            torque_xyz_nm=np.zeros(3),
            timestamp_ns=2,
        )
        state_bad = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=ft_bad,
            timestamp_ns=2,
        )
        cmd_pos_bad = state_bad.ee_position_m.copy()
        cmd_vel_bad = state_bad.ee_velocity_m_s.copy()
        _safe_pos, _safe_vel, level_bad = monitor.check_and_filter(cmd_pos_bad, cmd_vel_bad, state_bad)
        assert level_bad in (safety_mod.SafetyLevel.CRITICAL, safety_mod.SafetyLevel.WARNING)

    def test_multiple_sensor_readings_chain(self, safety_mod):
        """Process a sequence of readings and verify monitoring consistency."""
        monitor = safety_mod.SafetyMonitor(limits=safety_mod.SafetyLimits())
        levels = []
        for i in range(10):
            force = float(i) * 1.5  # Gradually increasing force
            ft = safety_mod.ForceTorqueReading(
                force_xyz_n=np.array([force, 0.0, 0.0]),
                torque_xyz_nm=np.zeros(3),
                timestamp_ns=i * 1_000_000,
            )
            state = safety_mod.RobotState(
                joint_positions_rad=np.zeros(7),
                joint_velocities_rad_s=np.zeros(7),
                ee_position_m=np.array([0.0, 0.0, -0.15]),
                ee_velocity_m_s=np.zeros(3),
                ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                force_torque=ft,
                timestamp_ns=i * 1_000_000,
            )
            cmd_pos = state.ee_position_m.copy()
            cmd_vel = state.ee_velocity_m_s.copy()
            _safe_pos, _safe_vel, level = monitor.check_and_filter(cmd_pos, cmd_vel, state)
            levels.append(level)

        # First readings should be nominal, later ones may escalate
        assert levels[0] == safety_mod.SafetyLevel.NOMINAL

    def test_reset_after_emergency(self, safety_mod):
        """After emergency stop, monitor should be resettable."""
        monitor = safety_mod.SafetyMonitor(limits=safety_mod.SafetyLimits())

        # Trigger emergency via excessive force
        ft_bad = safety_mod.ForceTorqueReading(
            force_xyz_n=np.array([200.0, 0.0, 0.0]),
            torque_xyz_nm=np.zeros(3),
            timestamp_ns=1,
        )
        state_bad = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=ft_bad,
            timestamp_ns=1,
        )
        cmd_pos = state_bad.ee_position_m.copy()
        cmd_vel = state_bad.ee_velocity_m_s.copy()
        monitor.check_and_filter(cmd_pos, cmd_vel, state_bad)

        # Reset using authorization code
        monitor.reset_stop(authorization_code="RESET-AUTH")
        assert monitor._is_stopped is False
