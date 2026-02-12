"""Integration test: Action -> Compliance Check -> Audit Record.

Validates the regulatory compliance pipeline: FDA tracking,
GCP compliance checking, and audit trail consistency.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def fda_mod():
    return load_module("fda_submission_tracker", "regulatory/fda-compliance/fda_submission_tracker.py")


@pytest.fixture()
def gcp_mod():
    return load_module("gcp_compliance_checker", "regulatory/ich-gcp/gcp_compliance_checker.py")


@pytest.fixture()
def access_mod():
    return load_module("access_control_manager", "privacy/access-control/access_control_manager.py")


class TestRegulatoryAuditTrail:
    def test_fda_submission_lifecycle(self, fda_mod):
        """Create a submission and track through status transitions."""
        submission = fda_mod.Submission(
            submission_id="SUB-INTEG-001",
            submission_type=fda_mod.SubmissionType.DE_NOVO,
            device_name="AI-Guided Surgical Robot",
            intended_use="Surgical assistance in oncology procedures",
            device_class=fda_mod.DeviceClass.CLASS_II,
            sponsor="Test Medical Inc.",
            status=fda_mod.SubmissionStatus.PLANNING,
            ai_ml_components=[
                fda_mod.AIMLComponent(
                    name="Tissue Segmentation Model",
                    model_type="unspecified",
                    architecture="U-Net",
                    training_data_size=10000,
                    training_data_sources=["institutional"],
                    validation_approach="k-fold cross-validation",
                    performance_metrics={"dice": 0.92},
                    locked_or_adaptive="locked",
                )
            ],
        )
        assert submission.status == fda_mod.SubmissionStatus.PLANNING
        d = submission.to_dict()
        assert d["submission_id"] == "SUB-INTEG-001"
        # to_dict() returns count of ai_ml_components, not the list
        assert d["ai_ml_components"] == 1

    def test_gcp_compliance_assessment(self, gcp_mod):
        """Run a GCP compliance assessment and verify scoring."""
        checker = gcp_mod.GCPComplianceChecker(
            guideline_version="E6_R3",
            jurisdiction="us_fda",
        )
        report = checker.verify_compliance(ai_components_present=True)
        assert isinstance(report, gcp_mod.ComplianceReport)
        assert 0 <= report.overall_score <= 100
        assert report.total_findings >= 0

    def test_access_control_audit(self, access_mod):
        """Access decisions should generate audit entries."""
        manager = access_mod.AccessControlManager(
            compliance_framework="21_cfr_part_11",
            audit_enabled=True,
            mfa_required=False,
        )

        # Define a custom role with string permissions
        manager.define_role(
            name="test_investigator",
            permissions=["read_phi", "write_clinical_data"],
            description="Test investigator role",
        )

        # Assign a user to that role
        manager.assign_role(
            user_id="USR-INTEG-001",
            role="test_investigator",
            name="Test Investigator",
            email="test@example.com",
            site_id="SITE-001",
            user_type="human",
        )

        # Check access with string action
        decision = manager.check_access(
            user_id="USR-INTEG-001",
            resource="patient/PT-001",
            action="read_phi",
        )
        assert isinstance(decision, access_mod.AccessDecision)

        # Verify audit log grew
        audit_log = manager.get_audit_log()
        assert len(audit_log) > 0
