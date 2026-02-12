"""Tests for examples-new/01_realtime_safety_monitoring.py

Covers SafetyMonitor, ForceTorqueSensorProcessor, and WorkspaceBoundaryGenerator
with nominal, warning, and critical scenarios.
"""

from __future__ import annotations

import numpy as np


# ── SafetyLevel / StopCategory enums ──────────────────────────────────────


class TestEnums:
    def test_safety_levels_ordering(self, safety_mod):
        levels = safety_mod.SafetyLevel
        assert levels.NOMINAL.value < levels.WARNING.value
        assert levels.WARNING.value < levels.CRITICAL.value
        assert levels.CRITICAL.value < levels.EMERGENCY.value

    def test_stop_categories_exist(self, safety_mod):
        cats = safety_mod.StopCategory
        assert cats.CATEGORY_0.value == "immediate_power_removal"
        assert cats.CATEGORY_1.value == "controlled_deceleration_then_stop"
        assert cats.CATEGORY_2.value == "controlled_stop_power_maintained"


# ── SafetyLimits defaults ─────────────────────────────────────────────────


class TestSafetyLimits:
    def test_default_force_max(self, default_safety_limits):
        assert default_safety_limits.force_max_n == 5.0

    def test_default_workspace(self, default_safety_limits):
        np.testing.assert_array_equal(
            default_safety_limits.workspace_center_m,
            np.array([0.0, 0.0, -0.15]),
        )
        assert default_safety_limits.workspace_radius_m == 0.15

    def test_default_joint_limits_count(self, default_safety_limits):
        assert len(default_safety_limits.joint_limits_deg) == 7


# ── SafetyMonitor – nominal operation ─────────────────────────────────────


class TestSafetyMonitorNominal:
    def test_nominal_pass_through(self, safety_mod, default_safety_limits, nominal_robot_state):
        monitor = safety_mod.SafetyMonitor(default_safety_limits)
        cmd_pos = np.array([0.0, 0.0, -0.15])
        cmd_vel = np.array([0.01, 0.0, 0.0])

        safe_pos, safe_vel, level = monitor.check_and_filter(cmd_pos, cmd_vel, nominal_robot_state)

        assert level == safety_mod.SafetyLevel.NOMINAL
        np.testing.assert_array_equal(safe_pos, cmd_pos)
        np.testing.assert_array_equal(safe_vel, cmd_vel)

    def test_event_log_empty_after_nominal(self, safety_mod, default_safety_limits, nominal_robot_state):
        monitor = safety_mod.SafetyMonitor(default_safety_limits)
        monitor.check_and_filter(
            np.array([0.0, 0.0, -0.15]),
            np.array([0.01, 0.0, 0.0]),
            nominal_robot_state,
        )
        assert len(monitor.get_event_log()) == 0

    def test_status_summary_initial(self, safety_mod, default_safety_limits):
        monitor = safety_mod.SafetyMonitor(default_safety_limits)
        status = monitor.get_status_summary()
        assert status["is_stopped"] is False
        assert status["total_cycles"] == 0
        assert status["total_events"] == 0


# ── SafetyMonitor – force limit violations ────────────────────────────────


class TestSafetyMonitorForce:
    def test_force_exceeds_limit_triggers_critical(self, safety_mod, default_safety_limits):
        monitor = safety_mod.SafetyMonitor(default_safety_limits)

        # Force above 5 N limit
        high_force_ft = safety_mod.ForceTorqueReading(
            force_xyz_n=np.array([6.0, 0.0, 0.0]),
            torque_xyz_nm=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        state = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=high_force_ft,
            timestamp_ns=1_000_000_000,
        )

        _, _, level = monitor.check_and_filter(
            np.array([0.0, 0.0, -0.15]),
            np.zeros(3),
            state,
        )
        assert level == safety_mod.SafetyLevel.CRITICAL
        assert monitor.get_status_summary()["is_stopped"] is True

    def test_force_warning_threshold(self, safety_mod, default_safety_limits):
        monitor = safety_mod.SafetyMonitor(default_safety_limits)
        # Force at 82% of 5 N = 4.1 N  (warning threshold is 80%)
        warn_ft = safety_mod.ForceTorqueReading(
            force_xyz_n=np.array([4.1, 0.0, 0.0]),
            torque_xyz_nm=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        state = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=warn_ft,
            timestamp_ns=1_000_000_000,
        )

        _, _, level = monitor.check_and_filter(
            np.array([0.0, 0.0, -0.15]),
            np.zeros(3),
            state,
        )
        assert level == safety_mod.SafetyLevel.WARNING

    def test_invalid_ft_sensor_triggers_critical(self, safety_mod, default_safety_limits):
        monitor = safety_mod.SafetyMonitor(default_safety_limits)
        invalid_ft = safety_mod.ForceTorqueReading(
            force_xyz_n=np.zeros(3),
            torque_xyz_nm=np.zeros(3),
            timestamp_ns=1_000_000_000,
            is_valid=False,
        )
        state = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=invalid_ft,
            timestamp_ns=1_000_000_000,
        )
        _, _, level = monitor.check_and_filter(
            np.array([0.0, 0.0, -0.15]),
            np.zeros(3),
            state,
        )
        assert level == safety_mod.SafetyLevel.CRITICAL


# ── SafetyMonitor – workspace violations ──────────────────────────────────


class TestSafetyMonitorWorkspace:
    def test_outside_workspace_triggers_critical(self, safety_mod, default_safety_limits, zero_ft_reading):
        monitor = safety_mod.SafetyMonitor(default_safety_limits)
        state = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=zero_ft_reading,
            timestamp_ns=1_000_000_000,
        )
        # Command position far outside workspace
        far_pos = np.array([1.0, 1.0, 1.0])
        _, _, level = monitor.check_and_filter(far_pos, np.zeros(3), state)
        assert level == safety_mod.SafetyLevel.CRITICAL


# ── SafetyMonitor – velocity violations ───────────────────────────────────


class TestSafetyMonitorVelocity:
    def test_excessive_velocity_triggers_critical(self, safety_mod, default_safety_limits, zero_ft_reading):
        monitor = safety_mod.SafetyMonitor(default_safety_limits)
        state = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=zero_ft_reading,
            timestamp_ns=1_000_000_000,
        )
        fast_vel = np.array([0.5, 0.0, 0.0])  # 0.5 m/s > 0.25 limit
        _, _, level = monitor.check_and_filter(
            np.array([0.0, 0.0, -0.15]),
            fast_vel,
            state,
        )
        assert level == safety_mod.SafetyLevel.CRITICAL


# ── SafetyMonitor – callbacks ─────────────────────────────────────────────


class TestSafetyMonitorCallbacks:
    def test_critical_callback_fires(self, safety_mod, default_safety_limits, zero_ft_reading):
        monitor = safety_mod.SafetyMonitor(default_safety_limits)
        callback_events: list = []
        monitor.register_callback(
            safety_mod.SafetyLevel.CRITICAL,
            lambda e: callback_events.append(e),
        )

        # Trigger critical via workspace violation
        state = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=zero_ft_reading,
            timestamp_ns=1_000_000_000,
        )
        monitor.check_and_filter(np.array([1.0, 1.0, 1.0]), np.zeros(3), state)
        assert len(callback_events) >= 1
        assert callback_events[0].level == safety_mod.SafetyLevel.CRITICAL


# ── SafetyMonitor – reset ─────────────────────────────────────────────────


class TestSafetyMonitorReset:
    def test_reset_clears_stop(self, safety_mod, default_safety_limits, zero_ft_reading):
        monitor = safety_mod.SafetyMonitor(default_safety_limits)
        state = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=zero_ft_reading,
            timestamp_ns=1_000_000_000,
        )
        # Trigger stop
        monitor.check_and_filter(np.array([1.0, 1.0, 1.0]), np.zeros(3), state)
        assert monitor.get_status_summary()["is_stopped"] is True

        monitor.reset_stop("AUTH-CODE")
        assert monitor.get_status_summary()["is_stopped"] is False


# ── ForceTorqueSensorProcessor ────────────────────────────────────────────


class TestForceTorqueSensorProcessor:
    def test_bias_calibration(self, safety_mod):
        proc = safety_mod.ForceTorqueSensorProcessor(filter_cutoff_hz=50.0, sample_rate_hz=1000.0)
        readings = [
            safety_mod.ForceTorqueReading(
                force_xyz_n=np.array([0.1, -0.05, 0.02]),
                torque_xyz_nm=np.zeros(3),
                timestamp_ns=int(i * 1e6),
            )
            for i in range(100)
        ]
        proc.calibrate_bias(readings)
        np.testing.assert_allclose(proc._bias_force, [0.1, -0.05, 0.02], atol=1e-6)

    def test_process_removes_bias(self, safety_mod):
        proc = safety_mod.ForceTorqueSensorProcessor(filter_cutoff_hz=50.0, sample_rate_hz=1000.0)
        bias_reading = safety_mod.ForceTorqueReading(
            force_xyz_n=np.array([1.0, 0.0, 0.0]),
            torque_xyz_nm=np.zeros(3),
            timestamp_ns=0,
        )
        proc.calibrate_bias([bias_reading] * 100)

        raw = safety_mod.ForceTorqueReading(
            force_xyz_n=np.array([1.5, 0.0, 0.0]),
            torque_xyz_nm=np.zeros(3),
            timestamp_ns=int(1e6),
        )
        processed = proc.process(raw)
        # After bias removal the effective force should be ~0.5 N (filtered)
        assert processed.force_xyz_n[0] > 0.0
        assert processed.force_xyz_n[0] < 1.5

    def test_filter_alpha_computation(self, safety_mod):
        alpha = safety_mod.ForceTorqueSensorProcessor._compute_filter_alpha(50.0, 1000.0)
        assert 0 < alpha < 1


# ── WorkspaceBoundaryGenerator ────────────────────────────────────────────


class TestWorkspaceBoundaryGenerator:
    def test_identity_transform(self, safety_mod):
        gen = safety_mod.WorkspaceBoundaryGenerator()
        tumor_center = np.array([0.05, 0.0, 0.1])
        center_robot, radius = gen.compute_workspace_from_tumor(
            tumor_center_patient_m=tumor_center,
            tumor_radius_m=0.02,
        )
        np.testing.assert_allclose(center_robot, tumor_center, atol=1e-10)
        assert radius > 0.02  # radius must exceed tumor size

    def test_max_radius_clamp(self, safety_mod):
        gen = safety_mod.WorkspaceBoundaryGenerator()
        _, radius = gen.compute_workspace_from_tumor(
            tumor_center_patient_m=np.array([0.0, 0.0, 0.0]),
            tumor_radius_m=0.5,  # very large
            max_radius_m=0.20,
        )
        assert radius <= 0.20
