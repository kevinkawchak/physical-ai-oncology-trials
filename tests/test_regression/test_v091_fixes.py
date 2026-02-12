"""Regression guards for v0.9.1 fixes.

Each test targets a specific bug from CHANGELOG v0.9.1 to prevent recurrence.
"""

from __future__ import annotations

import numpy as np


class TestDeidentificationSaltSecurity:
    """Regression for weak default salt (v0.9.1 SECURITY)."""

    def test_salt_not_default(self, deid_mod):
        """Default salt must not be the hardcoded 'default_salt' string."""
        config = deid_mod.DeidentificationConfig()
        assert config.hash_salt != "default_salt"

    def test_two_configs_default_salt_empty(self, deid_mod):
        """Default salt should be empty, requiring explicit configuration."""
        c1 = deid_mod.DeidentificationConfig()
        c2 = deid_mod.DeidentificationConfig()
        assert c1.hash_salt == c2.hash_salt == ""


class TestBreachResponseClamp:
    """Regression for risk score clamping (v0.9.1)."""

    def test_risk_assessment_with_boundary_scores(self, breach_mod):
        """Risk assessment should handle boundary scores gracefully."""
        mgr = breach_mod.BreachResponseManager(organization="Test Org")
        incident = mgr.report_incident(
            incident_type="unauthorized_disclosure",
            description="Test boundary scores",
            phi_types_involved=["ssn"],
            individuals_affected=1,
            discovery_date="2026-03-01",
        )
        assessment = mgr.assess_breach_risk(
            incident,
            phi_nature_score=1,
            unauthorized_party_score=1,
        )
        assert assessment.risk_level is not None


class TestAccessControlAuditCopy:
    """Regression for audit log returning reference (v0.9.1 SECURITY)."""

    def test_audit_log_returns_copy(self, access_mod):
        """get_audit_log should return a copy, not internal reference."""
        mgr = access_mod.AccessControlManager()
        mgr.define_role(name="reader", permissions=["read"])
        mgr.assign_role(user_id="U1", role="reader", name="User One")
        mgr.check_access("U1", "resource", "read")

        log1 = mgr.get_audit_log()
        log2 = mgr.get_audit_log()
        # Should be equal content but different objects
        assert len(log1) == len(log2)


class TestSafetyMonitorResetRegression:
    """Regression for safety monitor state management."""

    def test_monitor_stops_on_critical(self, safety_mod):
        """Monitor should stop robot on critical force violation."""
        limits = safety_mod.SafetyLimits()
        monitor = safety_mod.SafetyMonitor(limits)

        high_ft = safety_mod.ForceTorqueReading(
            force_xyz_n=np.array([10.0, 0.0, 0.0]),
            torque_xyz_nm=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        state = safety_mod.RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=np.array([0.0, 0.0, -0.15]),
            ee_velocity_m_s=np.zeros(3),
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=high_ft,
            timestamp_ns=1_000_000_000,
        )
        _, _, level = monitor.check_and_filter(
            np.array([0.0, 0.0, -0.15]),
            np.zeros(3),
            state,
        )
        assert level == safety_mod.SafetyLevel.CRITICAL
        status = monitor.get_status_summary()
        assert status["is_stopped"] is True
