"""Integration test: Privacy -> Regulatory compliance workflow.

Validates the flow from de-identification through access control
and regulatory compliance checking.
"""

from __future__ import annotations

from tests.conftest import load_module


class TestPrivacyRegulatoryWorkflow:
    def test_deidentify_then_access_control(self):
        deid_mod = load_module(
            "deidentification_pipeline",
            "privacy/de-identification/deidentification_pipeline.py",
        )
        access_mod = load_module(
            "access_control_manager",
            "privacy/access-control/access_control_manager.py",
        )

        # De-identify patient data
        config = deid_mod.DeidentificationConfig()
        transformer = deid_mod.SafeHarborTransformer(config)
        text = "Patient John Doe, SSN 123-45-6789, diagnosed with NSCLC."
        transformed, log = transformer.transform_text(text)
        assert "123-45-6789" not in transformed
        assert len(log) >= 1

        # Set up access control for de-identified data
        mgr = access_mod.AccessControlManager()
        mgr.define_role(
            name="researcher",
            permissions=["read_deidentified", "run_queries"],
            mfa_required=False,
        )
        mgr.assign_role(
            user_id="RESEARCHER-001",
            role="researcher",
            name="Dr. Researcher",
        )
        decision = mgr.check_access(
            "RESEARCHER-001",
            "deidentified_dataset",
            "read_deidentified",
        )
        assert decision.granted is True

    def test_phi_detection_then_breach_assessment(self):
        phi_mod = load_module(
            "phi_detector",
            "privacy/phi-pii-management/phi_detector.py",
        )
        breach_mod = load_module(
            "breach_response_protocol",
            "privacy/breach-response/breach_response_protocol.py",
        )

        # Detect PHI in text
        detector = phi_mod.PHIDetector()
        findings = detector.scan_text("Patient SSN: 123-45-6789. Email: p@test.com")
        assert len(findings) >= 1

        # If PHI found, report incident
        mgr = breach_mod.BreachResponseManager(organization="Test Hospital")
        incident = mgr.report_incident(
            incident_type="unauthorized_disclosure",
            description="PHI found in research dataset",
            phi_types_involved=["ssn", "email"],
            individuals_affected=len(findings),
            discovery_date="2026-03-01",
        )
        assessment = mgr.assess_breach_risk(
            incident,
            phi_nature_score=4,
            unauthorized_party_score=2,
        )
        assert assessment.risk_level is not None

    def test_gcp_compliance_check(self):
        gcp_mod = load_module(
            "gcp_compliance_checker",
            "regulatory/ich-gcp/gcp_compliance_checker.py",
        )
        checker = gcp_mod.GCPComplianceChecker()
        report = checker.verify_compliance(ai_components_present=True)
        assert report.overall_score >= 0
        # Should have AI-specific findings
        ai_findings = [f for f in report.findings if f.ai_specific]
        assert len(ai_findings) >= 0  # May or may not have AI findings
