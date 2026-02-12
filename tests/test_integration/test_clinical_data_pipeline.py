"""Integration test: Clinical Data Pipeline.

Validates: DICOM constants -> deidentification config -> PHI detection patterns.
Tests the clinical data flow without actual DICOM files or servers.
"""

from __future__ import annotations

import re

import pytest

from tests.conftest import load_module


@pytest.fixture()
def dicom_mod():
    return load_module("dicom_inspector", "tools/dicom-inspector/dicom_inspector.py")


@pytest.fixture()
def deid_mod():
    return load_module("deidentification_pipeline", "privacy/de-identification/deidentification_pipeline.py")


@pytest.fixture()
def phi_mod():
    return load_module("phi_detector", "privacy/phi-pii-management/phi_detector.py")


class TestClinicalDataPipeline:
    def test_dicom_phi_tags_map_to_phi_categories(self, dicom_mod, phi_mod):
        """DICOM PHI tags should relate to PHI detection categories."""
        dicom_phi_names = set(dicom_mod.PHI_TAGS.values())
        phi_categories = [e.name for e in phi_mod.PHICategory]
        # At minimum, NAME should be in both
        assert "NAME" in phi_categories
        has_name_in_dicom = any("name" in n.lower() for n in dicom_phi_names)
        assert has_name_in_dicom

    def test_deidentification_config_has_all_methods(self, deid_mod):
        """Deidentification should support Safe Harbor and Expert Determination."""
        methods = [e.name for e in deid_mod.DeidentificationMethod]
        assert "SAFE_HARBOR" in methods
        assert "EXPERT_DETERMINATION" in methods

    def test_safe_harbor_removes_ssn_pattern(self, deid_mod):
        """SSN pattern should be detected and transformed by Safe Harbor."""
        config = deid_mod.DeidentificationConfig()
        transformer = deid_mod.SafeHarborTransformer(config=config)
        text = "Patient SSN is 123-45-6789"
        result, log = transformer.transform_text(text)
        assert "123-45-6789" not in result

    def test_phi_detector_pattern_coverage(self, phi_mod):
        """PHI detector should have regex patterns for key identifier types."""
        assert len(phi_mod.PHI_PATTERNS) >= 10
        # Should cover SSN, email, phone at minimum
        covered_categories = set(phi_mod.PHI_PATTERNS.keys())
        assert phi_mod.PHICategory.SSN in covered_categories
        assert phi_mod.PHICategory.EMAIL in covered_categories

    def test_end_to_end_phi_detect_and_deidentify(self, phi_mod, deid_mod):
        """Detect PHI in text, then deidentify it."""
        text = "Patient John Doe, SSN 123-45-6789, email john@example.com"

        # Step 1: Detect
        detector = phi_mod.PHIDetector()
        findings = detector.scan_text(text)
        assert len(findings) > 0

        # Step 2: Deidentify
        config = deid_mod.DeidentificationConfig()
        transformer = deid_mod.SafeHarborTransformer(config=config)
        cleaned, log = transformer.transform_text(text)
        assert "123-45-6789" not in cleaned
        assert "john@example.com" not in cleaned
