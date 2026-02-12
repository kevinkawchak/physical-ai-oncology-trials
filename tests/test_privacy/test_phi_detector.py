"""Tests for PHI/PII Detection Pipeline.

Validates regex-based pattern detection, risk categorisation,
scan-result filtering, and DICOM tag definitions.

LICENSE: MIT
"""

from __future__ import annotations


import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------

PHI_MOD_PATH = "privacy/phi-pii-management/phi_detector.py"


@pytest.fixture()
def phi_mod():
    return load_module("phi_detector", PHI_MOD_PATH)


@pytest.fixture()
def detector(phi_mod):
    return phi_mod.PHIDetector(detection_mode="comprehensive", confidence_threshold=0.7)


@pytest.fixture()
def strict_detector(phi_mod):
    """Detector with a high confidence threshold."""
    return phi_mod.PHIDetector(detection_mode="quick", confidence_threshold=0.95)


# ===========================================================================
# TestPHICategoryEnum
# ===========================================================================


class TestPHICategoryEnum:
    """Verify the 18 HIPAA Safe Harbor identifiers are defined."""

    def test_all_18_categories_present(self, phi_mod):
        names = [m.name for m in phi_mod.PHICategory]
        assert len(names) == 18
        assert "SSN" in names
        assert "NAME" in names
        assert "MRN" in names

    def test_category_values_are_strings(self, phi_mod):
        for cat in phi_mod.PHICategory:
            assert isinstance(cat.value, str)

    def test_unique_values(self, phi_mod):
        values = [c.value for c in phi_mod.PHICategory]
        assert len(values) == len(set(values))


# ===========================================================================
# TestRiskLevel
# ===========================================================================


class TestRiskLevel:
    def test_risk_levels_defined(self, phi_mod):
        names = {r.name for r in phi_mod.RiskLevel}
        assert names == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


# ===========================================================================
# TestPHIFinding
# ===========================================================================


class TestPHIFinding:
    def test_to_dict_keys(self, phi_mod):
        f = phi_mod.PHIFinding(
            phi_type=phi_mod.PHICategory.SSN,
            location="char 0-11",
            value_preview="12***9",
            confidence=0.95,
            risk_level=phi_mod.RiskLevel.CRITICAL,
        )
        d = f.to_dict()
        assert "phi_type" in d
        assert "confidence" in d
        assert d["phi_type"] == "social_security_numbers"

    def test_finding_default_fields(self, phi_mod):
        f = phi_mod.PHIFinding(
            phi_type=phi_mod.PHICategory.EMAIL,
            location="char 0-20",
            value_preview="te***m",
            confidence=0.90,
            risk_level=phi_mod.RiskLevel.MEDIUM,
        )
        assert f.source_file == ""
        assert f.line_number == 0
        assert f.context == ""


# ===========================================================================
# TestScanResult
# ===========================================================================


class TestScanResult:
    def test_empty_scan_result(self, phi_mod):
        sr = phi_mod.ScanResult(scan_id="S-1", timestamp="2026-01-01", dataset_path="/tmp")
        assert sr.total_findings == 0
        assert sr.findings == []
        assert sr.risk_assessment == "not_assessed"

    def test_get_findings_by_category(self, phi_mod):
        f1 = phi_mod.PHIFinding(
            phi_type=phi_mod.PHICategory.SSN,
            location="a",
            value_preview="x",
            confidence=0.9,
            risk_level=phi_mod.RiskLevel.CRITICAL,
        )
        f2 = phi_mod.PHIFinding(
            phi_type=phi_mod.PHICategory.EMAIL,
            location="b",
            value_preview="y",
            confidence=0.9,
            risk_level=phi_mod.RiskLevel.MEDIUM,
        )
        sr = phi_mod.ScanResult(scan_id="S-2", timestamp="t", dataset_path="/tmp", findings=[f1, f2])
        ssn_results = sr.get_findings_by_category(phi_mod.PHICategory.SSN)
        assert len(ssn_results) == 1
        assert ssn_results[0].phi_type == phi_mod.PHICategory.SSN

    def test_get_findings_by_risk(self, phi_mod):
        f1 = phi_mod.PHIFinding(
            phi_type=phi_mod.PHICategory.SSN,
            location="a",
            value_preview="x",
            confidence=0.9,
            risk_level=phi_mod.RiskLevel.CRITICAL,
        )
        f2 = phi_mod.PHIFinding(
            phi_type=phi_mod.PHICategory.EMAIL,
            location="b",
            value_preview="y",
            confidence=0.9,
            risk_level=phi_mod.RiskLevel.MEDIUM,
        )
        sr = phi_mod.ScanResult(scan_id="S-3", timestamp="t", dataset_path="/tmp", findings=[f1, f2])
        crit = sr.get_findings_by_risk(phi_mod.RiskLevel.CRITICAL)
        assert len(crit) == 1

    def test_to_dict_structure(self, phi_mod):
        sr = phi_mod.ScanResult(scan_id="S-4", timestamp="t", dataset_path="/data")
        d = sr.to_dict()
        assert "scan_id" in d
        assert "findings_by_category" in d
        assert d["errors"] == []


# ===========================================================================
# TestPHIDetectorScanText
# ===========================================================================


class TestPHIDetectorScanText:
    def test_detects_ssn(self, detector):
        findings = detector.scan_text("SSN: 123-45-6789")
        types_found = {f.phi_type.name for f in findings}
        assert "SSN" in types_found

    def test_detects_email(self, detector):
        findings = detector.scan_text("Email: alice@example.com")
        types_found = {f.phi_type.name for f in findings}
        assert "EMAIL" in types_found

    def test_detects_phone(self, detector):
        findings = detector.scan_text("Phone: (555) 123-4567")
        types_found = {f.phi_type.name for f in findings}
        assert "PHONE" in types_found

    def test_detects_mrn(self, detector):
        findings = detector.scan_text("MRN: 12345678")
        types_found = {f.phi_type.name for f in findings}
        assert "MRN" in types_found

    def test_detects_ip_address(self, detector):
        findings = detector.scan_text("Server IP: 192.168.1.100")
        types_found = {f.phi_type.name for f in findings}
        assert "IP_ADDRESS" in types_found

    def test_no_findings_for_clean_text(self, detector):
        findings = detector.scan_text("This is a clean medical note about tumor staging.")
        # Filter out very low confidence or non-PHI matches
        high_conf = [f for f in findings if f.confidence >= 0.85]
        # Clean text should have no high-confidence PHI findings
        assert len(high_conf) == 0

    def test_confidence_threshold_filtering(self, phi_mod):
        """A very high threshold should suppress low-confidence findings."""
        d = phi_mod.PHIDetector(confidence_threshold=0.99)
        findings = d.scan_text("ZIP: 62701")
        # Geographic ZIP has confidence 0.60, should be filtered out
        geo = [f for f in findings if f.phi_type == phi_mod.PHICategory.GEOGRAPHIC]
        assert len(geo) == 0

    def test_custom_pattern_detection(self, phi_mod):
        d = phi_mod.PHIDetector(custom_patterns={"study_id": r"STUDY-\d{4}"})
        findings = d.scan_text("Enrolled in STUDY-1234.")
        custom = [f for f in findings if "Custom pattern" in f.context]
        assert len(custom) >= 1

    def test_multiple_findings_in_one_text(self, detector):
        text = "SSN: 123-45-6789, Email: test@example.com, Phone: (555) 111-2222"
        findings = detector.scan_text(text)
        types_found = {f.phi_type.name for f in findings}
        assert "SSN" in types_found
        assert "EMAIL" in types_found


# ===========================================================================
# TestRiskCategorisation
# ===========================================================================


class TestRiskCategorisation:
    def test_ssn_high_confidence_is_critical(self, detector, phi_mod):
        risk = detector._categorize_risk(phi_mod.PHICategory.SSN, 0.95)
        assert risk == phi_mod.RiskLevel.CRITICAL

    def test_ssn_low_confidence_is_high(self, detector, phi_mod):
        risk = detector._categorize_risk(phi_mod.PHICategory.SSN, 0.70)
        assert risk == phi_mod.RiskLevel.HIGH

    def test_email_high_confidence_is_medium(self, detector, phi_mod):
        risk = detector._categorize_risk(phi_mod.PHICategory.EMAIL, 0.95)
        assert risk == phi_mod.RiskLevel.MEDIUM

    def test_url_is_low(self, detector, phi_mod):
        risk = detector._categorize_risk(phi_mod.PHICategory.URL, 0.70)
        assert risk == phi_mod.RiskLevel.LOW


# ===========================================================================
# TestOverallRiskAssessment
# ===========================================================================


class TestOverallRiskAssessment:
    def test_clean_assessment(self, detector, phi_mod):
        sr = phi_mod.ScanResult(scan_id="X", timestamp="t", dataset_path="/x")
        assert detector._assess_overall_risk(sr) == "clean"

    def test_critical_assessment(self, detector, phi_mod):
        f = phi_mod.PHIFinding(
            phi_type=phi_mod.PHICategory.SSN,
            location="a",
            value_preview="x",
            confidence=0.99,
            risk_level=phi_mod.RiskLevel.CRITICAL,
        )
        sr = phi_mod.ScanResult(scan_id="X", timestamp="t", dataset_path="/x", findings=[f])
        assert detector._assess_overall_risk(sr) == "critical"


# ===========================================================================
# TestPHIPatternsDict
# ===========================================================================


class TestPHIPatternsDict:
    def test_patterns_dict_has_expected_categories(self, phi_mod):
        patterns = phi_mod.PHI_PATTERNS
        assert phi_mod.PHICategory.SSN in patterns
        assert phi_mod.PHICategory.EMAIL in patterns
        assert phi_mod.PHICategory.PHONE in patterns

    def test_each_pattern_has_required_keys(self, phi_mod):
        for cat, entries in phi_mod.PHI_PATTERNS.items():
            for entry in entries:
                assert "pattern" in entry
                assert "confidence" in entry
                assert "description" in entry


# ===========================================================================
# TestDICOMTagsDict
# ===========================================================================


class TestDICOMTags:
    def test_dicom_tags_defined(self, phi_mod):
        tags = phi_mod.PHI_DICOM_TAGS
        assert len(tags) > 0
        assert "(0010,0010)" in tags  # PatientName

    def test_dicom_tag_has_required_keys(self, phi_mod):
        for tag, info in phi_mod.PHI_DICOM_TAGS.items():
            assert "name" in info
            assert "category" in info
            assert "action" in info
