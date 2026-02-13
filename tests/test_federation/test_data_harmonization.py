"""Tests for federation.data_harmonization module.

Covers DICOMNormalizer, VocabularyHarmonizer, FHIRResourceMapper,
and DataHarmonizationEngine.

License: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from federation.data_harmonization import (
    DICOMMetadata,
    DICOMNormalizer,
    DataHarmonizationEngine,
    FHIRResourceMapper,
    HarmonizationStatus,
    VocabularyHarmonizer,
    VocabularyMapping,
)


# ---------------------------------------------------------------------------
# DICOMNormalizer tests
# ---------------------------------------------------------------------------


class TestDICOMNormalizer:
    def test_modality_normalization(self):
        normalizer = DICOMNormalizer()
        meta = DICOMMetadata(modality="MRI", slice_thickness=1.0, pixel_spacing=(0.5, 0.5))
        normalized, warnings = normalizer.normalize(meta)
        assert normalized.modality == "MR"

    def test_pet_normalization(self):
        normalizer = DICOMNormalizer()
        meta = DICOMMetadata(modality="PET", slice_thickness=3.0, pixel_spacing=(4.0, 4.0))
        normalized, _ = normalizer.normalize(meta)
        assert normalized.modality == "PT"

    def test_body_part_normalization(self):
        normalizer = DICOMNormalizer()
        meta = DICOMMetadata(modality="CT", body_part="THORAX", slice_thickness=1.0, pixel_spacing=(0.7, 0.7))
        normalized, _ = normalizer.normalize(meta)
        assert normalized.body_part == "CHEST"

    def test_body_part_abdomen(self):
        normalizer = DICOMNormalizer()
        meta = DICOMMetadata(modality="CT", body_part="ABD", slice_thickness=1.0, pixel_spacing=(1.0, 1.0))
        normalized, _ = normalizer.normalize(meta)
        assert normalized.body_part == "ABDOMEN"

    def test_missing_slice_thickness_warning(self):
        normalizer = DICOMNormalizer()
        meta = DICOMMetadata(modality="CT", slice_thickness=0, pixel_spacing=(1.0, 1.0))
        normalized, warnings = normalizer.normalize(meta)
        assert any("slice thickness" in w.lower() for w in warnings)

    def test_missing_pixel_spacing_warning(self):
        normalizer = DICOMNormalizer()
        meta = DICOMMetadata(modality="CT", slice_thickness=1.0, pixel_spacing=(0.0, 0.0))
        _, warnings = normalizer.normalize(meta)
        assert any("pixel spacing" in w.lower() for w in warnings)

    def test_patient_position_uppercase(self):
        normalizer = DICOMNormalizer()
        meta = DICOMMetadata(modality="CT", patient_position="hfs", slice_thickness=1.0, pixel_spacing=(1.0, 1.0))
        normalized, _ = normalizer.normalize(meta)
        assert normalized.patient_position == "HFS"


# ---------------------------------------------------------------------------
# VocabularyHarmonizer tests
# ---------------------------------------------------------------------------


class TestVocabularyHarmonizer:
    def test_icd10_to_snomed_breast(self):
        harmonizer = VocabularyHarmonizer()
        mapping = harmonizer.map_icd10_to_snomed("C50.9")
        assert mapping is not None
        assert mapping.target_system == "SNOMED-CT"
        assert "breast" in mapping.target_display.lower()

    def test_icd10_to_snomed_lung(self):
        harmonizer = VocabularyHarmonizer()
        mapping = harmonizer.map_icd10_to_snomed("C34.9")
        assert mapping is not None
        assert "lung" in mapping.target_display.lower()

    def test_unknown_icd10_returns_none(self):
        harmonizer = VocabularyHarmonizer()
        mapping = harmonizer.map_icd10_to_snomed("Z99.99")
        assert mapping is None

    def test_loinc_for_cea(self):
        harmonizer = VocabularyHarmonizer()
        loinc = harmonizer.get_loinc_for_marker("CEA")
        assert loinc is not None
        assert loinc["loinc"] == "2039-6"

    def test_loinc_case_insensitive(self):
        harmonizer = VocabularyHarmonizer()
        loinc = harmonizer.get_loinc_for_marker("psa")
        assert loinc is not None

    def test_custom_mapping(self):
        harmonizer = VocabularyHarmonizer()
        custom = VocabularyMapping(
            source_system="LOCAL", source_code="XYZ", target_system="SNOMED-CT", target_code="12345"
        )
        harmonizer.add_custom_mapping(custom)
        result = harmonizer.harmonize_code("LOCAL", "XYZ")
        assert result is not None
        assert result.target_code == "12345"

    def test_harmonize_code_icd10(self):
        harmonizer = VocabularyHarmonizer()
        result = harmonizer.harmonize_code("icd-10", "C61")
        assert result is not None
        assert "prostate" in result.target_display.lower()


# ---------------------------------------------------------------------------
# FHIRResourceMapper tests
# ---------------------------------------------------------------------------


class TestFHIRResourceMapper:
    def test_create_condition(self):
        mapper = FHIRResourceMapper()
        resource = mapper.create_condition("Patient/001", "C50.9", "2026-01-15")
        assert resource.resource_type == "Condition"
        assert resource.subject == "Patient/001"
        assert resource.code["code"] == "C50.9"

    def test_create_observation(self):
        mapper = FHIRResourceMapper()
        resource = mapper.create_observation("Patient/001", "CEA", 5.2)
        assert resource.resource_type == "Observation"
        assert resource.code.get("system") == "http://loinc.org"
        assert resource.value["valueQuantity"]["value"] == 5.2

    def test_create_medication_statement(self):
        mapper = FHIRResourceMapper()
        resource = mapper.create_medication_statement("Patient/001", "Pembrolizumab", "200mg IV q3w")
        assert resource.resource_type == "MedicationStatement"
        assert resource.code["display"] == "Pembrolizumab"


# ---------------------------------------------------------------------------
# DataHarmonizationEngine tests
# ---------------------------------------------------------------------------


class TestDataHarmonizationEngine:
    def test_harmonize_dicom_series(self):
        engine = DataHarmonizationEngine(trial_id="TEST")
        batch = [
            DICOMMetadata(modality="MRI", body_part="brain", slice_thickness=1.0, pixel_spacing=(0.5, 0.5)),
            DICOMMetadata(modality="CT", body_part="chest", slice_thickness=1.25, pixel_spacing=(0.7, 0.7)),
        ]
        result = engine.harmonize_dicom_series("s1", batch)
        assert result.records_processed == 2
        assert result.records_harmonized == 2
        assert result.status == HarmonizationStatus.COMPLETED

    def test_harmonize_diagnoses(self):
        engine = DataHarmonizationEngine()
        diagnoses = [
            {"code": "C50.9", "patient_ref": "P/1"},
            {"code": "C34.9", "patient_ref": "P/2"},
        ]
        resources, result = engine.harmonize_diagnoses("s1", diagnoses)
        assert len(resources) == 2
        assert result.vocabulary_mappings_applied == 2

    def test_harmonize_lab_results(self):
        engine = DataHarmonizationEngine()
        labs = [
            {"marker_name": "CEA", "value": 5.0, "patient_ref": "P/1"},
            {"marker_name": "PSA", "value": 4.0, "patient_ref": "P/2"},
        ]
        resources, result = engine.harmonize_lab_results("s1", labs)
        assert len(resources) == 2
        assert result.vocabulary_mappings_applied == 2

    def test_harmonization_summary(self):
        engine = DataHarmonizationEngine(trial_id="SUM-TEST")
        engine.harmonize_dicom_series(
            "s1", [DICOMMetadata(modality="CT", slice_thickness=1.0, pixel_spacing=(1.0, 1.0))]
        )
        summary = engine.get_harmonization_summary()
        assert summary["trial_id"] == "SUM-TEST"
        assert summary["total_events"] == 1
