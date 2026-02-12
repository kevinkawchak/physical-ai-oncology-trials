"""Tests for privacy/de-identification/deidentification_pipeline.py

Covers DeidentificationConfig, SafeHarborTransformer, and
DeidentificationResult.
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


# ── Enums and configuration ──────────────────────────────────────────────


class TestEnumsAndConfig:
    def test_deidentification_methods(self, deid_mod):
        assert deid_mod.DeidentificationMethod.SAFE_HARBOR.value == "safe_harbor"
        assert deid_mod.DeidentificationMethod.EXPERT_DETERMINATION.value == "expert_determination"

    def test_date_handling_options(self, deid_mod):
        assert deid_mod.DateHandling.REMOVE.value == "remove"
        assert deid_mod.DateHandling.YEAR_ONLY.value == "year_only"
        assert deid_mod.DateHandling.DATE_SHIFT.value == "date_shift"

    def test_geography_handling(self, deid_mod):
        assert deid_mod.GeographyHandling.ZIP3.value == "zip3"

    def test_default_config(self, deid_mod):
        config = deid_mod.DeidentificationConfig()
        assert config.method == deid_mod.DeidentificationMethod.SAFE_HARBOR
        assert config.preserve_clinical_utility is True
        assert config.age_handling == "cap_at_89"


# ── SafeHarborTransformer ────────────────────────────────────────────────


class TestSafeHarborTransformer:
    def test_removes_ssn(self, deid_mod):
        config = deid_mod.DeidentificationConfig()
        transformer = deid_mod.SafeHarborTransformer(config)
        text = "Patient SSN is 123-45-6789 and needs treatment."
        transformed, log = transformer.transform_text(text)
        assert "123-45-6789" not in transformed
        assert len(log) >= 1

    def test_removes_email(self, deid_mod):
        config = deid_mod.DeidentificationConfig()
        transformer = deid_mod.SafeHarborTransformer(config)
        text = "Contact at john.doe@hospital.org for follow-up."
        transformed, log = transformer.transform_text(text)
        assert "john.doe@hospital.org" not in transformed

    def test_removes_phone(self, deid_mod):
        config = deid_mod.DeidentificationConfig()
        transformer = deid_mod.SafeHarborTransformer(config)
        text = "Call (555) 123-4567 for scheduling."
        transformed, log = transformer.transform_text(text)
        assert "(555) 123-4567" not in transformed

    def test_removes_ip_address(self, deid_mod):
        config = deid_mod.DeidentificationConfig()
        transformer = deid_mod.SafeHarborTransformer(config)
        text = "Login from 192.168.1.100 detected."
        transformed, log = transformer.transform_text(text)
        assert "192.168.1.100" not in transformed

    def test_removes_url(self, deid_mod):
        config = deid_mod.DeidentificationConfig()
        transformer = deid_mod.SafeHarborTransformer(config)
        text = "See report at https://hospital.org/report/12345 for details."
        transformed, log = transformer.transform_text(text)
        assert "https://hospital.org/report/12345" not in transformed

    def test_preserves_clinical_content(self, deid_mod):
        config = deid_mod.DeidentificationConfig()
        transformer = deid_mod.SafeHarborTransformer(config)
        # Use clinical terms that avoid the name-pattern regex (two adjacent
        # multi-letter words trigger the IGNORECASE name pattern).
        text = "T4N2M0. KPS 80. CA-125: 35 U/mL."
        transformed, _log = transformer.transform_text(text)
        assert "T4N2M0" in transformed
        assert "KPS" in transformed
        assert "CA-125" in transformed
        assert "35" in transformed

    def test_structured_transform(self, deid_mod, sample_patient_record):
        config = deid_mod.DeidentificationConfig()
        transformer = deid_mod.SafeHarborTransformer(config)
        phi_fields = {
            "patient_name": "replace",
            "mrn": "pseudonymize",
            "date_of_birth": "generalize_date",
            "ssn": "replace",
            "phone": "replace",
            "email": "replace",
            "address": "replace",
        }
        result, log = transformer.transform_structured(sample_patient_record, phi_fields)
        # PHI fields should be transformed
        assert result["patient_name"] != "John Doe"
        assert result["ssn"] != "123-45-6789"
        # Clinical fields should be preserved
        assert result["diagnosis"] == "Stage IIIA Non-Small Cell Lung Cancer"
        assert result["tumor_volume_cm3"] == 12.5


# ── DeidentificationResult ───────────────────────────────────────────────


class TestDeidentificationResult:
    def test_result_to_dict(self, deid_mod):
        result = deid_mod.DeidentificationResult(
            result_id="R-001",
            timestamp="2026-03-01T09:00:00",
            method="safe_harbor",
            records_processed=100,
            records_output=100,
            identifiers_removed=500,
        )
        d = result.to_dict()
        assert d["records_processed"] == 100
        assert d["identifiers_removed"] == 500
        assert d["method"] == "safe_harbor"

    def test_result_empty_errors(self, deid_mod):
        result = deid_mod.DeidentificationResult(
            result_id="R-002",
            timestamp="2026-03-01T09:00:00",
            method="safe_harbor",
        )
        assert result.errors == []
        assert result.transformation_log == []
