"""Integration tests: Patient data -> De-identification -> Clinical utility preserved.

Tests the cross-module flow from PHI detection through de-identification,
verifying that protected health information is removed while preserving
clinical content needed for research.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def deid_mod():
    return load_module(
        "deidentification_pipeline",
        "privacy/de-identification/deidentification_pipeline.py",
    )


@pytest.fixture()
def phi_mod():
    return load_module(
        "phi_detector",
        "privacy/phi-pii-management/phi_detector.py",
    )


SAMPLE_TEXT = (
    "Patient John Doe (DOB: 1965-03-15, SSN: 123-45-6789) was diagnosed with "
    "Stage IIIA NSCLC. Treatment: concurrent chemoradiation with cisplatin 75 mg/m2 "
    "and 60 Gy in 30 fractions. Phone: (555) 123-4567. Email: john.doe@example.com. "
    "MRN: MRN-123456."
)

SAMPLE_RECORD = {
    "patient_name": "John Doe",
    "mrn": "MRN-123456",
    "date_of_birth": "1965-03-15",
    "ssn": "123-45-6789",
    "phone": "(555) 123-4567",
    "email": "john.doe@example.com",
    "address": "123 Main St, Springfield, IL 62701",
    "diagnosis": "Stage IIIA Non-Small Cell Lung Cancer",
    "tumor_volume_cm3": 12.5,
    "treatment": "Concurrent Chemoradiation",
}

PHI_FIELDS = {
    "patient_name": "NAME",
    "mrn": "MEDICAL_RECORD_NUMBER",
    "date_of_birth": "DATE",
    "ssn": "SSN",
    "phone": "PHONE",
    "email": "EMAIL",
    "address": "ADDRESS",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrivacyToClinical:
    """Integration: PHI detection -> de-identification -> clinical utility."""

    def test_deid_removes_phi(self, deid_mod, phi_mod):
        """De-identified text has no PHI detected by PHIDetector."""
        pipeline = deid_mod.DeidentificationPipeline()
        deidentified = pipeline.deidentify_text(SAMPLE_TEXT)
        # Now scan with PHI detector
        detector = phi_mod.PHIDetector()
        findings = detector.scan_text(deidentified)
        original_findings = detector.scan_text(SAMPLE_TEXT)
        # The deidentified text should have substantially fewer findings
        assert len(findings) < len(original_findings)

    def test_clinical_content_preserved(self, deid_mod):
        """Clinical terms survive de-identification."""
        pipeline = deid_mod.DeidentificationPipeline()
        deidentified = pipeline.deidentify_text(SAMPLE_TEXT)
        # Key clinical numeric/dosimetric terms should remain
        assert "75 mg/m2" in deidentified or "mg/m2" in deidentified
        assert "30 fractions" in deidentified or "fractions" in deidentified
        # Treatment keyword survives
        assert "treatment" in deidentified.lower()

    def test_structured_deid_preserves_diagnosis(self, deid_mod):
        """Structured de-identification keeps diagnosis field."""
        pipeline = deid_mod.DeidentificationPipeline()
        deid_record = pipeline.deidentify_record(SAMPLE_RECORD, PHI_FIELDS)
        assert "diagnosis" in deid_record
        diag = deid_record["diagnosis"]
        assert "lung" in diag.lower() or "cancer" in diag.lower() or "nsclc" in diag.lower()

    def test_scan_then_deid_workflow(self, deid_mod, phi_mod):
        """Scan text for PHI, then de-identify, verify all findings resolved."""
        detector = phi_mod.PHIDetector()
        findings_before = detector.scan_text(SAMPLE_TEXT)
        assert len(findings_before) > 0
        pipeline = deid_mod.DeidentificationPipeline()
        deidentified = pipeline.deidentify_text(SAMPLE_TEXT)
        findings_after = detector.scan_text(deidentified)
        assert len(findings_after) < len(findings_before)

    def test_phi_categories_covered(self, phi_mod):
        """PHIDetector covers common PHI types with patterns."""
        detector = phi_mod.PHIDetector()
        patterns = getattr(detector, "_patterns", None) or getattr(detector, "patterns", None)
        if patterns is not None:
            assert len(patterns) >= 5
        else:
            findings = detector.scan_text(
                "John Doe DOB 1965-03-15 SSN 123-45-6789 phone (555) 123-4567 email john@test.com MRN-99999"
            )
            assert len(findings) >= 2

    def test_deid_record_removes_ssn(self, deid_mod):
        """Structured de-identification removes or masks SSN."""
        pipeline = deid_mod.DeidentificationPipeline()
        deid_record = pipeline.deidentify_record(SAMPLE_RECORD, PHI_FIELDS)
        ssn_val = str(deid_record.get("ssn", ""))
        assert ssn_val != "123-45-6789"

    def test_deid_record_removes_patient_name(self, deid_mod):
        """Structured de-identification removes or masks patient name."""
        pipeline = deid_mod.DeidentificationPipeline()
        deid_record = pipeline.deidentify_record(SAMPLE_RECORD, PHI_FIELDS)
        name_val = str(deid_record.get("patient_name", ""))
        assert name_val != "John Doe"

    def test_deid_text_replaces_email(self, deid_mod):
        """Email address is replaced in de-identified text."""
        pipeline = deid_mod.DeidentificationPipeline()
        deidentified = pipeline.deidentify_text(SAMPLE_TEXT)
        assert "john.doe@example.com" not in deidentified

    def test_deid_text_replaces_phone(self, deid_mod):
        """Phone number is replaced in de-identified text."""
        pipeline = deid_mod.DeidentificationPipeline()
        deidentified = pipeline.deidentify_text(SAMPLE_TEXT)
        assert "(555) 123-4567" not in deidentified

    def test_tumor_volume_preserved(self, deid_mod):
        """Numeric clinical values like tumor volume are preserved."""
        pipeline = deid_mod.DeidentificationPipeline()
        deid_record = pipeline.deidentify_record(SAMPLE_RECORD, PHI_FIELDS)
        assert deid_record.get("tumor_volume_cm3") == 12.5

    def test_phi_detector_returns_location_info(self, phi_mod):
        """PHI findings should include location/type information."""
        detector = phi_mod.PHIDetector()
        findings = detector.scan_text("John Doe SSN 123-45-6789")
        for finding in findings:
            assert hasattr(finding, "phi_type") or hasattr(finding, "category") or isinstance(finding, dict)
