#!/usr/bin/env python3
"""Example 05: Cross-Site Data Normalization.

Demonstrates DICOM metadata normalization, FHIR R4 resource mapping,
and vocabulary harmonization (ICD-10, SNOMED CT, LOINC) across
heterogeneous site systems.

Key Concepts:
    - DICOM modality and body part normalization
    - ICD-10 to SNOMED CT vocabulary mapping
    - LOINC coding for tumor markers
    - FHIR R4 Condition and Observation resource creation
    - Cross-site data harmonization engine

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.

Version: 1.0.0
Last Updated: February 2026
License: MIT
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from federation.data_harmonization import (
    DICOMMetadata,
    DICOMNormalizer,
    DataHarmonizationEngine,
    FHIRResourceMapper,
    VocabularyHarmonizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulated Site Data
# ---------------------------------------------------------------------------

SITE_A_DICOM = [
    DICOMMetadata(
        study_uid="1.2.840.001",
        modality="MRI",
        body_part="brain",
        manufacturer="Siemens",
        slice_thickness=1.0,
        pixel_spacing=(0.5, 0.5),
        patient_position="HFS",
    ),
    DICOMMetadata(
        study_uid="1.2.840.002",
        modality="PET",
        body_part="wholebody",
        manufacturer="GE Medical",
        slice_thickness=3.27,
        pixel_spacing=(4.07, 4.07),
        patient_position="hfs",
    ),
]

SITE_B_DICOM = [
    DICOMMetadata(
        study_uid="1.2.840.101",
        modality="CT",
        body_part="THORAX",
        manufacturer="Philips",
        slice_thickness=1.25,
        pixel_spacing=(0.68, 0.68),
    ),
    DICOMMetadata(
        study_uid="1.2.840.102",
        modality="MR",
        body_part="ABD",
        manufacturer="Siemens",
        slice_thickness=3.0,
        pixel_spacing=(1.2, 1.2),
    ),
    DICOMMetadata(
        study_uid="1.2.840.103",
        modality="RTSTRUCT",
        body_part="Lung",
        manufacturer="Varian",
        slice_thickness=0,
        pixel_spacing=(0.0, 0.0),
    ),
]


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------


def demo_dicom_normalization() -> None:
    """Demonstrate DICOM metadata normalization."""
    print("\n--- DICOM Metadata Normalization ---")

    normalizer = DICOMNormalizer()

    print("\n  Site A (2 studies):")
    for metadata in SITE_A_DICOM:
        normalized, warnings = normalizer.normalize(metadata)
        print(
            f"    {metadata.modality} -> {normalized.modality}, "
            f"{metadata.body_part} -> {normalized.body_part}, "
            f"pos={normalized.patient_position}"
        )
        if warnings:
            for w in warnings:
                print(f"      WARNING: {w}")

    print("\n  Site B (3 studies):")
    for metadata in SITE_B_DICOM:
        normalized, warnings = normalizer.normalize(metadata)
        print(f"    {metadata.modality} -> {normalized.modality}, {metadata.body_part} -> {normalized.body_part}")
        if warnings:
            for w in warnings:
                print(f"      WARNING: {w}")


def demo_vocabulary_harmonization() -> None:
    """Demonstrate cross-vocabulary mapping."""
    print("\n--- Vocabulary Harmonization ---")

    harmonizer = VocabularyHarmonizer()

    icd10_codes = ["C50.9", "C34.9", "C18.9", "C61", "C71.9", "C25.9"]
    print("\n  ICD-10 to SNOMED CT mappings:")
    for code in icd10_codes:
        mapping = harmonizer.map_icd10_to_snomed(code)
        if mapping is not None:
            print(f"    {code} ({mapping.source_display}) -> SNOMED {mapping.target_code} ({mapping.target_display})")

    print("\n  LOINC tumor marker codes:")
    markers = ["CEA", "PSA", "CA125", "CA19-9", "AFP", "HER2"]
    for marker in markers:
        loinc = harmonizer.get_loinc_for_marker(marker)
        if loinc is not None:
            print(f"    {marker}: LOINC {loinc['loinc']} ({loinc['display']}) [{loinc['unit']}]")


def demo_fhir_resource_mapping() -> None:
    """Demonstrate FHIR R4 resource creation."""
    print("\n--- FHIR R4 Resource Mapping ---")

    mapper = FHIRResourceMapper()

    print("\n  Condition resources (from ICD-10):")
    conditions = [
        ("Patient/001", "C50.9", "2026-01-15"),
        ("Patient/002", "C34.9", "2025-11-20"),
        ("Patient/003", "C61", "2026-02-01"),
    ]
    for patient_ref, code, onset in conditions:
        resource = mapper.create_condition(patient_ref, code, onset)
        print(
            f"    {resource.resource_type}/{resource.resource_id}: "
            f"code={resource.code.get('code', 'N/A')}, "
            f"display={resource.code.get('display', 'N/A')}, "
            f"subject={resource.subject}"
        )

    print("\n  Observation resources (tumor markers):")
    observations = [
        ("Patient/001", "CEA", 5.2),
        ("Patient/002", "PSA", 8.7),
        ("Patient/003", "CA125", 42.0),
    ]
    for patient_ref, marker, value in observations:
        resource = mapper.create_observation(patient_ref, marker, value)
        print(
            f"    {resource.resource_type}/{resource.resource_id}: "
            f"code={resource.code.get('code', 'N/A')}, "
            f"value={resource.value}"
        )

    print("\n  MedicationStatement resources:")
    meds = [
        ("Patient/001", "Pembrolizumab", "200mg IV q3w"),
        ("Patient/002", "Carboplatin", "AUC 5 IV q3w"),
    ]
    for patient_ref, med_name, dose in meds:
        resource = mapper.create_medication_statement(patient_ref, med_name, dose)
        print(
            f"    {resource.resource_type}/{resource.resource_id}: "
            f"medication={resource.code.get('display', '')}, "
            f"dose={resource.value.get('dosage', '')}"
        )


def demo_cross_site_harmonization() -> None:
    """Demonstrate full cross-site data harmonization."""
    print("\n--- Cross-Site Data Harmonization ---")

    engine = DataHarmonizationEngine(trial_id="ONCO-HARMON-001")

    dicom_result_a = engine.harmonize_dicom_series("site_a", SITE_A_DICOM)
    dicom_result_b = engine.harmonize_dicom_series("site_b", SITE_B_DICOM)
    print(f"\n  DICOM: Site A={dicom_result_a.records_harmonized}, Site B={dicom_result_b.records_harmonized}")

    diagnoses = [
        {"code": "C50.9", "patient_ref": "Patient/101"},
        {"code": "C34.9", "patient_ref": "Patient/102"},
        {"code": "C18.9", "patient_ref": "Patient/103"},
        {"code": "C61", "patient_ref": "Patient/104"},
    ]
    fhir_conditions, dx_result = engine.harmonize_diagnoses("site_a", diagnoses)
    print(
        f"  Diagnoses: {dx_result.records_harmonized} harmonized, "
        f"{dx_result.vocabulary_mappings_applied} vocabulary mappings"
    )

    labs = [
        {"marker_name": "CEA", "value": 5.2, "patient_ref": "Patient/101"},
        {"marker_name": "PSA", "value": 4.1, "patient_ref": "Patient/104"},
        {"marker_name": "CA125", "value": 35.0, "patient_ref": "Patient/102"},
        {"marker_name": "AFP", "value": 12.0, "patient_ref": "Patient/103"},
    ]
    fhir_obs, lab_result = engine.harmonize_lab_results("site_b", labs)
    print(
        f"  Labs: {lab_result.records_harmonized} harmonized, {lab_result.vocabulary_mappings_applied} LOINC mappings"
    )

    summary = engine.get_harmonization_summary()
    print(f"\n  Harmonization summary: {summary}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all data harmonization demonstrations."""
    print("=" * 70)
    print("Example 05: Cross-Site Data Normalization")
    print("=" * 70)

    demo_dicom_normalization()
    demo_vocabulary_harmonization()
    demo_fhir_resource_mapping()
    demo_cross_site_harmonization()

    print("\n" + "=" * 70)
    print("Data harmonization demonstration completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
