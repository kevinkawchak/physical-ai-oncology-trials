"""Tests for regulatory/ich-gcp/gcp_compliance_checker.py

Covers GCP compliance checker, findings, and report generation.
"""

from __future__ import annotations


class TestEnums:
    def test_guideline_versions(self, gcp_mod):
        gv = gcp_mod.GuidelineVersion
        assert gv.E6_R2.value == "E6_R2"
        assert gv.E6_R3.value == "E6_R3"

    def test_compliance_status(self, gcp_mod):
        cs = gcp_mod.ComplianceStatus
        assert cs.COMPLIANT.value == "compliant"
        assert cs.NON_COMPLIANT.value == "non_compliant"
        assert cs.NOT_ASSESSED.value == "not_assessed"

    def test_finding_severity(self, gcp_mod):
        fs = gcp_mod.FindingSeverity
        assert fs.CRITICAL.value == "critical"
        assert fs.MINOR.value == "minor"


class TestComplianceFinding:
    def test_creation(self, gcp_mod):
        finding = gcp_mod.ComplianceFinding(
            finding_id="F-001",
            category="informed_consent",
            requirement="Documented consent required",
            status=gcp_mod.ComplianceStatus.COMPLIANT,
            severity=gcp_mod.FindingSeverity.MINOR,
            description="Consent forms properly documented",
        )
        assert finding.finding_id == "F-001"
        assert finding.ai_specific is False


class TestGCPComplianceChecker:
    def test_creation(self, gcp_mod):
        checker = gcp_mod.GCPComplianceChecker()
        assert checker is not None

    def test_verify_compliance(self, gcp_mod):
        checker = gcp_mod.GCPComplianceChecker()
        report = checker.verify_compliance()
        assert isinstance(report, gcp_mod.ComplianceReport)
        assert report.overall_score >= 0
        assert len(report.findings) > 0

    def test_compliance_report_to_dict(self, gcp_mod):
        checker = gcp_mod.GCPComplianceChecker()
        report = checker.verify_compliance()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "overall_score" in d
        assert "category_scores" in d

    def test_generate_report_text(self, gcp_mod):
        checker = gcp_mod.GCPComplianceChecker()
        report = checker.verify_compliance()
        text = checker.generate_report_text(report)
        assert isinstance(text, str)
        assert len(text) > 0
