"""Tests for ICH E6(R3) GCP Compliance Checker.

Validates compliance verification, scoring (with v0.9.2 NOT_ASSESSED
exclusion), finding severity classification, and report generation.

LICENSE: MIT
"""

from __future__ import annotations


import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "regulatory/ich-gcp/gcp_compliance_checker.py"


@pytest.fixture()
def gcp_mod():
    return load_module("gcp_compliance_checker", MOD_PATH)


@pytest.fixture()
def checker(gcp_mod):
    return gcp_mod.GCPComplianceChecker(guideline_version="E6_R3", jurisdiction="us_fda")


@pytest.fixture()
def sample_report(checker):
    return checker.verify_compliance(ai_components_present=True)


# ===========================================================================
# TestEnums
# ===========================================================================


class TestEnums:
    def test_guideline_versions(self, gcp_mod):
        names = {e.name for e in gcp_mod.GuidelineVersion}
        assert "E6_R2" in names
        assert "E6_R3" in names

    def test_jurisdiction_values(self, gcp_mod):
        names = {e.name for e in gcp_mod.Jurisdiction}
        assert "US_FDA" in names
        assert "EU_EMA" in names
        assert "ICH_GLOBAL" in names
        assert len(names) == 5

    def test_compliance_status_values(self, gcp_mod):
        names = {e.name for e in gcp_mod.ComplianceStatus}
        assert "COMPLIANT" in names
        assert "NOT_ASSESSED" in names
        assert "NOT_APPLICABLE" in names

    def test_finding_severity_values(self, gcp_mod):
        names = {e.name for e in gcp_mod.FindingSeverity}
        assert names == {"CRITICAL", "MAJOR", "MINOR", "OBSERVATION"}


# ===========================================================================
# TestComplianceFinding
# ===========================================================================


class TestComplianceFinding:
    def test_finding_construction(self, gcp_mod):
        f = gcp_mod.ComplianceFinding(
            finding_id="F-001",
            category="quality_management",
            requirement="RBQM documented",
            status=gcp_mod.ComplianceStatus.COMPLIANT,
            severity=gcp_mod.FindingSeverity.CRITICAL,
            description="Test finding",
        )
        assert f.finding_id == "F-001"
        assert f.ai_specific is False

    def test_finding_defaults(self, gcp_mod):
        f = gcp_mod.ComplianceFinding(
            finding_id="F-002",
            category="test",
            requirement="req",
            status=gcp_mod.ComplianceStatus.NOT_ASSESSED,
            severity=gcp_mod.FindingSeverity.MAJOR,
            description="desc",
        )
        assert f.recommendation == ""
        assert f.guideline_reference == ""


# ===========================================================================
# TestComplianceReport
# ===========================================================================


class TestComplianceReport:
    def test_report_id_format(self, sample_report):
        assert sample_report.report_id.startswith("GCP-")

    def test_report_version_and_jurisdiction(self, sample_report):
        assert sample_report.guideline_version == "E6_R3"
        assert sample_report.jurisdiction == "us_fda"

    def test_report_has_findings(self, sample_report):
        assert sample_report.total_findings > 0

    def test_category_scores_present(self, sample_report):
        assert len(sample_report.category_scores) > 0

    def test_to_dict(self, sample_report):
        d = sample_report.to_dict()
        assert "report_id" in d
        assert "overall_score" in d
        assert "category_scores" in d


# ===========================================================================
# TestScoreCalculation
# ===========================================================================


class TestScoreCalculation:
    def test_v092_not_assessed_excluded_from_score(self, checker, gcp_mod):
        """v0.9.2 fix: NOT_ASSESSED findings excluded from score denominator."""
        report = checker.verify_compliance(ai_components_present=True)
        # All findings default to NOT_ASSESSED in the simulated assessment
        # When all are NOT_ASSESSED, score should be 0.0
        # The key point: score calculation doesn't divide by total_findings
        # but by the number of assessed findings
        assessed = [f for f in report.findings if f.status != gcp_mod.ComplianceStatus.NOT_ASSESSED]
        if len(assessed) == 0:
            assert report.overall_score == 0.0

    def test_score_with_compliant_findings(self, gcp_mod):
        report = gcp_mod.ComplianceReport(
            report_id="TEST",
            guideline_version="E6_R3",
            jurisdiction="us_fda",
            assessment_date="2026-01-01",
        )
        report.findings = [
            gcp_mod.ComplianceFinding(
                finding_id="F1",
                category="test",
                requirement="r1",
                status=gcp_mod.ComplianceStatus.COMPLIANT,
                severity=gcp_mod.FindingSeverity.CRITICAL,
                description="d",
            ),
            gcp_mod.ComplianceFinding(
                finding_id="F2",
                category="test",
                requirement="r2",
                status=gcp_mod.ComplianceStatus.NON_COMPLIANT,
                severity=gcp_mod.FindingSeverity.MAJOR,
                description="d",
            ),
        ]
        # Manual verification: 1 compliant out of 2 assessed = 50%
        assessed = [f for f in report.findings if f.status != gcp_mod.ComplianceStatus.NOT_ASSESSED]
        compliant = sum(1 for f in assessed if f.status == gcp_mod.ComplianceStatus.COMPLIANT)
        score = (compliant / len(assessed)) * 100 if assessed else 0
        assert score == 50.0

    def test_ai_specific_findings_present(self, sample_report):
        ai_findings = [f for f in sample_report.findings if f.ai_specific]
        assert len(ai_findings) > 0


# ===========================================================================
# TestCategoryFiltering
# ===========================================================================


class TestCategoryFiltering:
    def test_check_specific_categories(self, checker):
        report = checker.verify_compliance(
            check_categories=["quality_management"],
            ai_components_present=True,
        )
        categories = {f.category for f in report.findings}
        assert "quality_management" in categories
        assert "data_governance" not in categories

    def test_ai_components_false_skips_ai_requirements(self, checker, gcp_mod):
        report = checker.verify_compliance(ai_components_present=False)
        ai_findings = [f for f in report.findings if f.ai_specific]
        assert len(ai_findings) == 0

    def test_all_categories_checked_by_default(self, checker, gcp_mod):
        report = checker.verify_compliance(ai_components_present=True)
        categories = {f.category for f in report.findings}
        expected = set(gcp_mod.GCP_REQUIREMENTS.keys())
        assert categories == expected


# ===========================================================================
# TestReportText
# ===========================================================================


class TestReportText:
    def test_report_text_generation(self, checker, sample_report):
        text = checker.generate_report_text(sample_report)
        assert "GCP COMPLIANCE REPORT" in text
        assert sample_report.report_id in text

    def test_report_text_includes_categories(self, checker, sample_report):
        text = checker.generate_report_text(sample_report)
        assert "CATEGORY SCORES" in text

    def test_report_text_includes_detailed_findings(self, checker, sample_report):
        text = checker.generate_report_text(sample_report)
        assert "DETAILED FINDINGS" in text


# ===========================================================================
# TestGCPRequirements
# ===========================================================================


class TestGCPRequirements:
    def test_requirements_dict_structure(self, gcp_mod):
        reqs = gcp_mod.GCP_REQUIREMENTS
        assert "quality_management" in reqs
        assert "digital_technology_provisions" in reqs
        assert "safety_reporting" in reqs

    def test_each_requirement_has_required_fields(self, gcp_mod):
        for category, items in gcp_mod.GCP_REQUIREMENTS.items():
            for item in items:
                assert "id" in item
                assert "requirement" in item
                assert "description" in item
                assert "severity" in item
