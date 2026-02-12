"""Tests for Access Control Manager.

Validates RBAC role/permission enforcement, user lifecycle, audit trails,
MFA gating, site-level restrictions, and 21 CFR Part 11 audit integrity.

LICENSE: MIT
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------

MOD_PATH = "privacy/access-control/access_control_manager.py"


@pytest.fixture()
def acm_mod():
    return load_module("access_control_manager", MOD_PATH)


@pytest.fixture()
def manager(acm_mod):
    return acm_mod.AccessControlManager(
        compliance_framework="21_cfr_part_11",
        audit_enabled=True,
        mfa_required=True,
    )


@pytest.fixture()
def manager_no_mfa(acm_mod):
    return acm_mod.AccessControlManager(
        compliance_framework="hipaa",
        audit_enabled=True,
        mfa_required=False,
    )


def _add_pi_user(acm_mod, mgr, user_id="PI-001", mfa=True):
    """Helper to add a PI user with MFA enrolled."""
    mgr.assign_role(user_id=user_id, role="principal_investigator", name="Dr. Test", email="pi@test.org", site_id="S1")
    mgr._users[user_id].mfa_enrolled = mfa
    return user_id


# ===========================================================================
# TestPermissionEnum
# ===========================================================================


class TestPermissionEnum:
    def test_permission_count(self, acm_mod):
        perms = list(acm_mod.Permission)
        assert len(perms) == 20

    def test_permission_values_unique(self, acm_mod):
        values = [p.value for p in acm_mod.Permission]
        assert len(values) == len(set(values))


# ===========================================================================
# TestUserTypeEnum
# ===========================================================================


class TestUserTypeEnum:
    def test_user_types(self, acm_mod):
        names = {u.name for u in acm_mod.UserType}
        assert names == {"HUMAN", "AI_SYSTEM", "SERVICE_ACCOUNT"}


# ===========================================================================
# TestRole
# ===========================================================================


class TestRole:
    def test_has_permission_true(self, acm_mod):
        role = acm_mod.Role(name="test", description="d", permissions=["read_phi", "write_phi"])
        assert role.has_permission("read_phi") is True

    def test_has_permission_false(self, acm_mod):
        role = acm_mod.Role(name="test", description="d", permissions=["read_phi"])
        assert role.has_permission("export_phi") is False


# ===========================================================================
# TestPredefinedRoles
# ===========================================================================


class TestPredefinedRoles:
    def test_predefined_roles_loaded(self, manager):
        assert "principal_investigator" in manager._roles
        assert "ai_ml_engineer" in manager._roles
        assert "biostatistician" in manager._roles

    def test_ai_system_no_mfa_required(self, manager):
        assert manager._roles["ai_system"].mfa_required is False

    def test_pi_has_read_phi(self, manager):
        role = manager._roles["principal_investigator"]
        assert role.has_permission("read_phi")

    def test_biostatistician_lacks_read_phi(self, manager):
        role = manager._roles["biostatistician"]
        assert not role.has_permission("read_phi")


# ===========================================================================
# TestAssignRole
# ===========================================================================


class TestAssignRole:
    def test_assign_role_creates_user(self, acm_mod, manager):
        manager.assign_role(user_id="U1", role="biostatistician", name="Stat")
        assert "U1" in manager._users
        assert manager._users["U1"].role_name == "biostatistician"

    def test_assign_invalid_role_raises(self, manager):
        with pytest.raises(ValueError, match="not defined"):
            manager.assign_role(user_id="U2", role="nonexistent_role")

    def test_user_type_assignment(self, acm_mod, manager):
        manager.assign_role(user_id="AI-1", role="ai_system", user_type="ai_system")
        assert manager._users["AI-1"].user_type == acm_mod.UserType.AI_SYSTEM


# ===========================================================================
# TestCheckAccess
# ===========================================================================


class TestCheckAccess:
    def test_grant_with_valid_permission(self, acm_mod, manager):
        _add_pi_user(acm_mod, manager)
        decision = manager.check_access("PI-001", "records", "read_phi")
        assert decision.granted is True

    def test_deny_unknown_user(self, manager):
        decision = manager.check_access("NOBODY", "records", "read_phi")
        assert decision.granted is False
        assert "not found" in decision.reason.lower()

    def test_deny_inactive_user(self, acm_mod, manager):
        _add_pi_user(acm_mod, manager)
        manager.revoke_access("PI-001", reason="study ended")
        decision = manager.check_access("PI-001", "records", "read_phi")
        assert decision.granted is False

    def test_deny_missing_permission(self, acm_mod, manager):
        _add_pi_user(acm_mod, manager, user_id="STAT-1", mfa=True)
        manager.assign_role(user_id="STAT-1", role="biostatistician")
        manager._users["STAT-1"].mfa_enrolled = True
        decision = manager.check_access("STAT-1", "records", "read_phi")
        assert decision.granted is False

    def test_deny_mfa_not_enrolled(self, acm_mod, manager):
        _add_pi_user(acm_mod, manager, user_id="NO-MFA", mfa=False)
        decision = manager.check_access("NO-MFA", "records", "read_phi")
        assert decision.granted is False
        assert "mfa" in decision.reason.lower()

    def test_grant_without_mfa_when_manager_mfa_disabled(self, acm_mod, manager_no_mfa):
        manager_no_mfa.assign_role(user_id="U1", role="principal_investigator", name="Dr. X")
        # MFA not enrolled, but manager doesn't require it
        decision = manager_no_mfa.check_access("U1", "records", "read_phi")
        assert decision.granted is True

    def test_site_restriction_blocks_cross_site_phi(self, acm_mod, manager):
        _add_pi_user(acm_mod, manager, user_id="PI-S1")
        decision = manager.check_access("PI-S1", "records", "read_phi", context={"site_id": "S2"})
        assert decision.granted is False
        assert "site" in decision.reason.lower()


# ===========================================================================
# TestAccessExpiration
# ===========================================================================


class TestAccessExpiration:
    def test_expired_user_denied(self, acm_mod, manager):
        past = (datetime.now() - timedelta(days=1)).isoformat()
        manager.assign_role(user_id="EXP-1", role="principal_investigator", access_expiration=past)
        manager._users["EXP-1"].mfa_enrolled = True
        decision = manager.check_access("EXP-1", "records", "read_phi")
        assert decision.granted is False
        assert "expired" in decision.reason.lower()

    def test_valid_expiration_granted(self, acm_mod, manager):
        future = (datetime.now() + timedelta(days=30)).isoformat()
        manager.assign_role(user_id="VAL-1", role="principal_investigator", access_expiration=future)
        manager._users["VAL-1"].mfa_enrolled = True
        decision = manager.check_access("VAL-1", "records", "read_phi")
        assert decision.granted is True


# ===========================================================================
# TestAuditLog
# ===========================================================================


class TestAuditLog:
    def test_audit_log_returns_copy(self, acm_mod, manager):
        """v0.9.2 fix: get_audit_log returns a copy of the internal list."""
        _add_pi_user(acm_mod, manager)
        manager.check_access("PI-001", "records", "read_phi")
        log1 = manager.get_audit_log()
        log2 = manager.get_audit_log()
        assert log1 is not log2  # different list objects
        assert len(log1) == len(log2)

    def test_audit_entries_have_integrity_hash(self, acm_mod, manager):
        _add_pi_user(acm_mod, manager)
        manager.check_access("PI-001", "records", "read_phi")
        entries = manager.get_audit_log()
        for entry in entries:
            assert entry.integrity_hash != ""

    def test_audit_filter_by_user(self, acm_mod, manager):
        _add_pi_user(acm_mod, manager)
        manager.check_access("PI-001", "records", "read_phi")
        entries = manager.get_audit_log(user_id="PI-001")
        assert all(e.user_id == "PI-001" for e in entries)

    def test_audit_filter_by_action(self, acm_mod, manager):
        _add_pi_user(acm_mod, manager)
        manager.check_access("PI-001", "records", "read_phi")
        entries = manager.get_audit_log(action_filter="read_phi")
        assert len(entries) >= 1

    def test_audit_entry_to_dict(self, acm_mod):
        entry = acm_mod.AuditEntry(
            audit_id="AUD-1",
            timestamp="2026-01-01",
            user_id="U1",
            user_type="human",
            action="read_phi",
            resource="records",
            decision="granted",
            reason="OK",
        )
        d = entry.to_dict()
        assert d["audit_id"] == "AUD-1"
        assert "integrity_hash" in d


# ===========================================================================
# TestDefineRole
# ===========================================================================


class TestDefineRole:
    def test_custom_role_registered(self, manager):
        manager.define_role("custom_role", permissions=["read_deidentified"], description="Test role")
        assert "custom_role" in manager._roles

    def test_custom_role_has_correct_permissions(self, manager):
        manager.define_role("viewer", permissions=["read_deidentified", "run_queries"])
        role = manager._roles["viewer"]
        assert role.has_permission("read_deidentified")
        assert role.has_permission("run_queries")
        assert not role.has_permission("read_phi")


# ===========================================================================
# TestRevokeAccess
# ===========================================================================


class TestRevokeAccess:
    def test_revoke_deactivates_user(self, acm_mod, manager):
        _add_pi_user(acm_mod, manager)
        manager.revoke_access("PI-001", reason="terminated")
        assert manager._users["PI-001"].active is False

    def test_revoke_nonexistent_user_no_error(self, manager):
        # Should not raise
        manager.revoke_access("GHOST-USER", reason="n/a")
