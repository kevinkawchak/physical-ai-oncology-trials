"""Tests for privacy/access-control/access_control_manager.py

Covers Permission enum, Role/User dataclasses, and AccessControlManager.
"""

from __future__ import annotations


class TestPermissionEnum:
    def test_phi_permissions(self, access_mod):
        p = access_mod.Permission
        assert p.READ_PHI.value == "read_phi"
        assert p.WRITE_PHI.value == "write_phi"
        assert p.EXPORT_PHI.value == "export_phi"

    def test_model_permissions(self, access_mod):
        p = access_mod.Permission
        assert p.TRAIN_MODEL.value == "train_model"
        assert p.DEPLOY_MODEL.value == "deploy_model"


class TestUserTypeEnum:
    def test_members(self, access_mod):
        ut = access_mod.UserType
        assert ut.HUMAN.value == "human"
        assert ut.AI_SYSTEM.value == "ai_system"
        assert ut.SERVICE_ACCOUNT.value == "service_account"


class TestRole:
    def test_creation(self, access_mod):
        role = access_mod.Role(
            name="data_manager",
            description="Manages clinical data",
            permissions=["read_phi", "write_clinical_data"],
        )
        assert role.name == "data_manager"

    def test_has_permission(self, access_mod):
        role = access_mod.Role(
            name="reviewer",
            description="Data reviewer",
            permissions=["read_phi", "read_deidentified"],
        )
        assert role.has_permission("read_phi") is True
        assert role.has_permission("write_phi") is False


class TestAccessControlManager:
    def test_creation(self, access_mod):
        mgr = access_mod.AccessControlManager()
        assert mgr.compliance_framework == "21_cfr_part_11"

    def test_define_role(self, access_mod):
        mgr = access_mod.AccessControlManager()
        mgr.define_role(
            name="test_role",
            permissions=["read_phi"],
            description="Test role",
        )
        assert "test_role" in mgr._roles

    def test_assign_role(self, access_mod):
        mgr = access_mod.AccessControlManager()
        mgr.define_role(name="analyst", permissions=["read_deidentified"])
        mgr.assign_role(
            user_id="USR-001",
            role="analyst",
            name="Test User",
            email="test@example.com",
        )
        assert "USR-001" in mgr._users

    def test_check_access_granted(self, access_mod):
        mgr = access_mod.AccessControlManager()
        mgr.define_role(name="reader", permissions=["read_deidentified"], mfa_required=False)
        mgr.assign_role(user_id="USR-002", role="reader", name="Reader")
        decision = mgr.check_access("USR-002", "dataset_001", "read_deidentified")
        assert isinstance(decision, access_mod.AccessDecision)
        assert decision.granted is True

    def test_check_access_denied(self, access_mod):
        mgr = access_mod.AccessControlManager()
        mgr.define_role(name="limited", permissions=["read_deidentified"])
        mgr.assign_role(user_id="USR-003", role="limited", name="Limited")
        decision = mgr.check_access("USR-003", "dataset_001", "write_phi")
        assert decision.granted is False

    def test_revoke_access(self, access_mod):
        mgr = access_mod.AccessControlManager()
        mgr.define_role(name="temp", permissions=["read_deidentified"])
        mgr.assign_role(user_id="USR-004", role="temp", name="Temp")
        mgr.revoke_access("USR-004", reason="expired")
        decision = mgr.check_access("USR-004", "data", "read_deidentified")
        assert decision.granted is False

    def test_audit_log(self, access_mod):
        mgr = access_mod.AccessControlManager()
        mgr.define_role(name="audited", permissions=["read_phi"])
        mgr.assign_role(user_id="USR-005", role="audited", name="Audited")
        mgr.check_access("USR-005", "dataset", "read_phi")
        log = mgr.get_audit_log()
        assert isinstance(log, list)
        assert len(log) >= 1
