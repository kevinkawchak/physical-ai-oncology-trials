"""Tests for privacy/phi-pii-management/phi_detector.py

Covers PHICategory, PHIFinding, ScanResult, and PHIDetector.
"""

from __future__ import annotations


class TestEnums:
    def test_phi_categories(self, phi_mod):
        pc = phi_mod.PHICategory
        assert pc.NAME.value == "names"
        assert pc.SSN.value == "social_security_numbers"
        assert pc.EMAIL.value == "email_addresses"
        assert pc.IP_ADDRESS.value == "ip_addresses"

    def test_risk_levels(self, phi_mod):
        rl = phi_mod.RiskLevel
        assert rl.LOW.value == "low"
        assert rl.CRITICAL.value == "critical"


class TestPHIFinding:
    def test_creation(self, phi_mod):
        finding = phi_mod.PHIFinding(
            phi_type=phi_mod.PHICategory.SSN,
            location="field:ssn",
            value_preview="123-**-****",
            confidence=0.99,
            risk_level=phi_mod.RiskLevel.CRITICAL,
        )
        assert finding.phi_type == phi_mod.PHICategory.SSN
        assert finding.confidence == 0.99

    def test_to_dict(self, phi_mod):
        finding = phi_mod.PHIFinding(
            phi_type=phi_mod.PHICategory.EMAIL,
            location="text:line10",
            value_preview="j***@example.com",
            confidence=0.95,
            risk_level=phi_mod.RiskLevel.HIGH,
        )
        d = finding.to_dict()
        assert isinstance(d, dict)
        assert "phi_type" in d


class TestScanResult:
    def test_creation(self, phi_mod):
        result = phi_mod.ScanResult(
            scan_id="SCAN-001",
            timestamp="2026-03-01T09:00:00",
            dataset_path="/data/test",
        )
        assert result.total_findings == 0
        assert result.risk_assessment == "not_assessed"

    def test_to_dict(self, phi_mod):
        result = phi_mod.ScanResult(
            scan_id="SCAN-002",
            timestamp="2026-03-01T09:00:00",
            dataset_path="/data/test",
            files_scanned=10,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["files_scanned"] == 10


class TestPHIDetector:
    def test_creation(self, phi_mod):
        detector = phi_mod.PHIDetector()
        assert detector.detection_mode == "comprehensive"

    def test_scan_text_ssn(self, phi_mod):
        detector = phi_mod.PHIDetector()
        findings = detector.scan_text("Patient SSN is 123-45-6789.")
        ssn_findings = [f for f in findings if f.phi_type == phi_mod.PHICategory.SSN]
        assert len(ssn_findings) >= 1

    def test_scan_text_email(self, phi_mod):
        detector = phi_mod.PHIDetector()
        findings = detector.scan_text("Email: patient@hospital.org")
        email_findings = [f for f in findings if f.phi_type == phi_mod.PHICategory.EMAIL]
        assert len(email_findings) >= 1

    def test_scan_text_no_phi(self, phi_mod):
        detector = phi_mod.PHIDetector()
        findings = detector.scan_text("KPS 80. ECOG 1. Stage IIIA.")
        # Clinical terms should not trigger PHI detection (or at most low confidence)
        high_confidence = [f for f in findings if f.confidence >= 0.9]
        assert len(high_confidence) == 0
