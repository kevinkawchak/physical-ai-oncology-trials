"""Unit tests for privacy/access-control/access_control_manager.py.

Tests cover Permission, UserType enums, Role, User, AccessDecision,
AuditEntry dataclasses, and AccessControlManager RBAC engine.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "privacy/access-control/access_control_manager.py"


@pytest.fixture()
def mod():
    return load_module("access_control_manager", MOD_PATH)


@pytest.fixture()
def manager(mod):
    m = mod.AccessControlManager(mfa_required=False)
    m.define_role(
        name="investigator",
        description="Principal investigator",
        permissions=["read_phi", "write_clinical_data"],
        mfa_required=False,
    )
    m.assign_role(
        user_id="USR-001",
        role="investigator",
        name="Dr. Smith",
        email="smith@hospital.org",
        site_id="SITE-001",
        user_type="human",
    )
    return m


class TestEnums:
    def test_permission_count(self, mod):
        assert len(list(mod.Permission)) >= 15

    def test_user_type(self, mod):
        names = [e.name for e in mod.UserType]
        for t in ("HUMAN", "AI_SYSTEM", "SERVICE_ACCOUNT"):
            assert t in names


class TestRole:
    def test_has_permission(self, mod):
        role = mod.Role(
            name="test",
            description="Test role",
            permissions=["read_phi"],
        )
        assert role.has_permission("read_phi") is True
        assert role.has_permission("write_phi") is False


class TestUser:
    def test_creation(self, mod):
        user = mod.User(
            user_id="USR-TEST",
            name="Test User",
            email="test@example.com",
            role_name="investigator",
            user_type=mod.UserType.HUMAN,
            site_id="SITE-001",
            active=True,
        )
        assert user.user_id == "USR-TEST"
        assert user.active is True


class TestAccessDecision:
    def test_granted(self, mod):
        decision = mod.AccessDecision(
            granted=True,
            user_id="USR-001",
            resource="patient/001",
            action="read_phi",
            reason="User has read_phi permission",
            timestamp="2026-01-01T00:00:00",
            audit_id="AUD-000001",
        )
        assert decision.granted is True


class TestAccessControlManager:
    def test_check_access_granted(self, mod, manager):
        decision = manager.check_access(
            user_id="USR-001",
            resource="patient/PT-001",
            action="read_phi",
        )
        assert decision.granted is True

    def test_check_access_denied(self, mod, manager):
        decision = manager.check_access(
            user_id="USR-001",
            resource="audit/logs",
            action="export_audit_logs",
        )
        assert decision.granted is False

    def test_audit_log_grows(self, mod, manager):
        initial_len = len(manager.get_audit_log())
        manager.check_access(
            user_id="USR-001",
            resource="patient/PT-002",
            action="read_phi",
        )
        assert len(manager.get_audit_log()) > initial_len

    def test_audit_log_returns_copy(self, mod, manager):
        log = manager.get_audit_log()
        original_len = len(log)
        log.append("INJECTED")
        assert len(manager.get_audit_log()) == original_len

    def test_unknown_user_denied(self, mod, manager):
        decision = manager.check_access(
            user_id="NONEXISTENT",
            resource="patient/001",
            action="read_phi",
        )
        assert decision.granted is False
