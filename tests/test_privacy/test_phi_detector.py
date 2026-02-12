"""Unit tests for privacy/phi-pii-management/phi_detector.py.

Tests cover PHICategory, RiskLevel enums, PHIFinding, ScanResult
dataclasses, PHI_PATTERNS constant, and PHIDetector scanning.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "privacy/phi-pii-management/phi_detector.py"


@pytest.fixture()
def mod():
    return load_module("phi_detector", MOD_PATH)


class TestEnums:
    def test_phi_category_18_identifiers(self, mod):
        # HIPAA defines 18 identifiers
        assert len(list(mod.PHICategory)) == 18

    def test_phi_category_members(self, mod):
        names = [e.name for e in mod.PHICategory]
        for cat in ("NAME", "SSN", "EMAIL", "PHONE", "MRN", "IP_ADDRESS"):
            assert cat in names

    def test_risk_level(self, mod):
        names = [e.name for e in mod.RiskLevel]
        for r in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            assert r in names


class TestPHIPatterns:
    def test_patterns_coverage(self, mod):
        assert len(mod.PHI_PATTERNS) >= 10

    def test_ssn_pattern_exists(self, mod):
        assert mod.PHICategory.SSN in mod.PHI_PATTERNS

    def test_email_pattern_exists(self, mod):
        assert mod.PHICategory.EMAIL in mod.PHI_PATTERNS


class TestPHIFinding:
    def test_creation(self, mod):
        finding = mod.PHIFinding(
            phi_type=mod.PHICategory.SSN,
            location="line 5",
            value_preview="123-**-****",
            confidence=0.99,
            risk_level=mod.RiskLevel.CRITICAL,
            source_file="test.csv",
            line_number=5,
        )
        assert finding.phi_type == mod.PHICategory.SSN
        assert finding.risk_level == mod.RiskLevel.CRITICAL

    def test_to_dict(self, mod):
        finding = mod.PHIFinding(
            phi_type=mod.PHICategory.EMAIL,
            location="line 10",
            value_preview="j***@example.com",
            confidence=0.95,
            risk_level=mod.RiskLevel.HIGH,
            source_file="data.csv",
            line_number=10,
        )
        d = finding.to_dict()
        assert isinstance(d, dict)


class TestScanResult:
    def test_creation(self, mod):
        result = mod.ScanResult(
            scan_id="SCAN-001",
            timestamp="2026-01-01T00:00:00",
            dataset_path="/data/test",
            total_findings=5,
            findings=[],
            files_scanned=3,
        )
        assert result.scan_id == "SCAN-001"
        assert result.total_findings == 5

    def test_get_findings_by_category(self, mod):
        findings = [
            mod.PHIFinding(
                phi_type=mod.PHICategory.SSN,
                location="line 1",
                value_preview="***",
                confidence=0.99,
                risk_level=mod.RiskLevel.CRITICAL,
                source_file="f.csv",
                line_number=1,
            ),
            mod.PHIFinding(
                phi_type=mod.PHICategory.EMAIL,
                location="line 2",
                value_preview="***",
                confidence=0.95,
                risk_level=mod.RiskLevel.HIGH,
                source_file="f.csv",
                line_number=2,
            ),
        ]
        result = mod.ScanResult(
            scan_id="SCAN-002",
            timestamp="2026-01-01T00:00:00",
            dataset_path="/data",
            total_findings=2,
            findings=findings,
            files_scanned=1,
        )
        ssn_findings = result.get_findings_by_category(mod.PHICategory.SSN)
        assert len(ssn_findings) == 1


class TestPHIDetector:
    def test_instantiation(self, mod):
        detector = mod.PHIDetector()
        assert detector is not None

    def test_scan_text_with_ssn(self, mod):
        detector = mod.PHIDetector()
        findings = detector.scan_text("Patient SSN: 123-45-6789")
        assert len(findings) >= 1

    def test_scan_text_with_email(self, mod):
        detector = mod.PHIDetector()
        findings = detector.scan_text("Contact: patient@hospital.org")
        assert len(findings) >= 1

    def test_scan_clean_text(self, mod):
        detector = mod.PHIDetector()
        findings = detector.scan_text("The tumor volume was 15.3 cm3")
        # Should not detect PHI in clinical data
        assert len(findings) == 0
