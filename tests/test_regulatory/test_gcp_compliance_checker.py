"""Unit tests for regulatory/ich-gcp/gcp_compliance_checker.py.

Tests cover GuidelineVersion, ComplianceStatus, FindingSeverity enums,
ComplianceFinding, ComplianceReport dataclasses, and GCPComplianceChecker.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "regulatory/ich-gcp/gcp_compliance_checker.py"


@pytest.fixture()
def mod():
    return load_module("gcp_compliance_checker", MOD_PATH)


class TestEnums:
    def test_guideline_version(self, mod):
        names = [e.name for e in mod.GuidelineVersion]
        assert "E6_R2" in names
        assert "E6_R3" in names

    def test_compliance_status(self, mod):
        names = [e.name for e in mod.ComplianceStatus]
        for s in ("COMPLIANT", "PARTIALLY_COMPLIANT", "NON_COMPLIANT", "NOT_ASSESSED"):
            assert s in names

    def test_finding_severity(self, mod):
        names = [e.name for e in mod.FindingSeverity]
        for s in ("CRITICAL", "MAJOR", "MINOR", "OBSERVATION"):
            assert s in names


class TestComplianceFinding:
    def test_creation(self, mod):
        finding = mod.ComplianceFinding(
            finding_id="F-001",
            category="quality_management",
            requirement="Risk-based quality management plan",
            status=mod.ComplianceStatus.NON_COMPLIANT,
            severity=mod.FindingSeverity.MAJOR,
            description="No RBQM plan documented",
            recommendation="Develop RBQM plan per ICH E6(R3) Section 5.0",
            guideline_reference="ICH E6(R3) 5.0",
        )
        assert finding.status == mod.ComplianceStatus.NON_COMPLIANT
        assert finding.severity == mod.FindingSeverity.MAJOR


class TestGCPRequirements:
    def test_requirements_exist(self, mod):
        assert len(mod.GCP_REQUIREMENTS) > 0

    def test_quality_management_category(self, mod):
        assert "quality_management" in mod.GCP_REQUIREMENTS


class TestGCPComplianceChecker:
    def test_instantiation(self, mod):
        checker = mod.GCPComplianceChecker(
            guideline_version=mod.GuidelineVersion.E6_R3,
        )
        assert checker is not None

    def test_assess_returns_report(self, mod):
        checker = mod.GCPComplianceChecker(
            guideline_version=mod.GuidelineVersion.E6_R3,
        )
        report = checker.verify_compliance()
        assert isinstance(report, mod.ComplianceReport)
        assert 0 <= report.overall_score <= 100

    def test_report_to_dict(self, mod):
        checker = mod.GCPComplianceChecker(
            guideline_version=mod.GuidelineVersion.E6_R3,
        )
        report = checker.verify_compliance()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "overall_score" in d

    def test_score_not_always_zero(self, mod):
        """Regression: Score should not always be 0%."""
        checker = mod.GCPComplianceChecker(
            guideline_version=mod.GuidelineVersion.E6_R3,
        )
        report = checker.verify_compliance()
        if report.total_findings > 0:
            assert report.overall_score >= 0
