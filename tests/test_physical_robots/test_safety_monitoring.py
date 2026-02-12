"""Unit tests for examples-new/01_realtime_safety_monitoring.py.

Tests cover SafetyLevel, StopCategory enums, SafetyLimits, ForceTorqueReading,
RobotState, SafetyMonitor, ForceTorqueSensorProcessor, WorkspaceBoundaryGenerator.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "examples-new/01_realtime_safety_monitoring.py"


@pytest.fixture()
def mod():
    return load_module("safety_monitoring", MOD_PATH)


@pytest.fixture()
def monitor(mod):
    limits = mod.SafetyLimits()
    return mod.SafetyMonitor(limits=limits)


@pytest.fixture()
def ft_zero(mod):
    return mod.ForceTorqueReading(
        force_xyz_n=np.zeros(3),
        torque_xyz_nm=np.zeros(3),
        timestamp_ns=1_000_000_000,
    )


@pytest.fixture()
def robot_ok(mod, ft_zero):
    return mod.RobotState(
        joint_positions_rad=np.array([0.0, 0.1, -0.2, 0.3, -0.1, 0.05, 0.0]),
        joint_velocities_rad_s=np.zeros(7),
        ee_position_m=np.array([0.0, 0.0, -0.15]),
        ee_velocity_m_s=np.array([0.01, 0.0, 0.0]),
        ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        force_torque=ft_zero,
        timestamp_ns=1_000_000_000,
    )


class TestEnums:
    def test_safety_level_ordering(self, mod):
        assert mod.SafetyLevel.NOMINAL.value < mod.SafetyLevel.EMERGENCY.value

    def test_stop_category_values(self, mod):
        names = [e.name for e in mod.StopCategory]
        for cat in ("CATEGORY_0", "CATEGORY_1", "CATEGORY_2"):
            assert cat in names


class TestSafetyLimits:
    def test_defaults(self, mod):
        limits = mod.SafetyLimits()
        assert limits.force_max_n > 0
        assert limits.velocity_max_m_s > 0

    def test_workspace_bounds(self, mod):
        limits = mod.SafetyLimits()
        assert hasattr(limits, "workspace_center_m")
        assert hasattr(limits, "workspace_radius_m")


class TestSafetyMonitor:
    def test_nominal_state_passes(self, mod, monitor, robot_ok):
        cmd_pos = robot_ok.ee_position_m.copy()
        cmd_vel = robot_ok.ee_velocity_m_s.copy()
        _safe_pos, _safe_vel, level = monitor.check_and_filter(cmd_pos, cmd_vel, robot_ok)
        assert level == mod.SafetyLevel.NOMINAL

    def test_excessive_force_triggers_critical(self, mod, monitor):
        ft_high = mod.ForceTorqueReading(
            force_xyz_n=np.array([100.0, 0.0, 0.0]),
            torque_xyz_nm=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        state = mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=ft_high,
            timestamp_ns=1_000_000_000,
        )
        cmd_pos = state.ee_position_m.copy()
        cmd_vel = state.ee_velocity_m_s.copy()
        _safe_pos, _safe_vel, level = monitor.check_and_filter(cmd_pos, cmd_vel, state)
        assert level in (mod.SafetyLevel.CRITICAL, mod.SafetyLevel.WARNING)

    def test_outside_workspace(self, mod, monitor):
        ft = mod.ForceTorqueReading(force_xyz_n=np.zeros(3), torque_xyz_nm=np.zeros(3), timestamp_ns=1_000_000_000)
        state = mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([10.0, 10.0, 10.0]),  # Far outside
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=ft,
            timestamp_ns=1_000_000_000,
        )
        # Command to far-outside position
        cmd_pos = np.array([10.0, 10.0, 10.0])
        cmd_vel = np.zeros(3)
        _safe_pos, _safe_vel, level = monitor.check_and_filter(cmd_pos, cmd_vel, state)
        assert level == mod.SafetyLevel.CRITICAL

    def test_excessive_velocity(self, mod, monitor):
        ft = mod.ForceTorqueReading(force_xyz_n=np.zeros(3), torque_xyz_nm=np.zeros(3), timestamp_ns=1_000_000_000)
        state = mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.array([5.0, 0.0, 0.0]),  # Very fast
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=ft,
            timestamp_ns=1_000_000_000,
        )
        cmd_pos = state.ee_position_m.copy()
        cmd_vel = np.array([5.0, 0.0, 0.0])
        _safe_pos, _safe_vel, level = monitor.check_and_filter(cmd_pos, cmd_vel, state)
        assert level in (mod.SafetyLevel.CRITICAL, mod.SafetyLevel.WARNING)

    def test_reset_clears_stop(self, mod, monitor):
        monitor._is_stopped = True
        monitor.reset_stop(authorization_code="RESET-AUTH")
        assert monitor._is_stopped is False

    def test_event_log(self, mod, monitor, robot_ok):
        cmd_pos = robot_ok.ee_position_m.copy()
        cmd_vel = robot_ok.ee_velocity_m_s.copy()
        monitor.check_and_filter(cmd_pos, cmd_vel, robot_ok)
        # Event log should track checks
        assert hasattr(monitor, "_event_log")
        assert isinstance(monitor.get_event_log(), list)


class TestForceTorqueSensorProcessor:
    def test_bias_calibration(self, mod):
        proc = mod.ForceTorqueSensorProcessor()
        readings = [
            mod.ForceTorqueReading(
                force_xyz_n=np.array([0.1, -0.05, 0.02]),
                torque_xyz_nm=np.array([0.01, -0.01, 0.005]),
                timestamp_ns=int(i * 1e7),
            )
            for i in range(10)
        ]
        proc.calibrate_bias(readings)
        assert np.linalg.norm(proc._bias_force) > 0

    def test_process_removes_bias(self, mod):
        proc = mod.ForceTorqueSensorProcessor()
        proc._bias_force = np.array([0.1, 0.0, 0.0])
        proc._bias_torque = np.zeros(3)
        # Set filter alpha to 1.0 (pass-through) so we test pure bias removal
        proc._filter_alpha = 1.0
        raw = mod.ForceTorqueReading(
            force_xyz_n=np.array([1.1, 2.0, 3.0]),
            torque_xyz_nm=np.array([0.1, 0.2, 0.3]),
            timestamp_ns=1_000_000_000,
        )
        processed = proc.process(raw)
        assert processed.force_xyz_n[0] == pytest.approx(1.0, abs=0.01)


class TestWorkspaceBoundaryGenerator:
    def test_identity_transform(self, mod):
        gen = mod.WorkspaceBoundaryGenerator(patient_to_robot_transform=np.eye(4))
        center_robot, radius = gen.compute_workspace_from_tumor(
            tumor_center_patient_m=np.array([0.0, 0.0, 0.0]),
            tumor_radius_m=0.03,
        )
        assert center_robot is not None
        assert radius > 0
        np.testing.assert_allclose(center_robot, np.array([0.0, 0.0, 0.0]), atol=1e-6)
