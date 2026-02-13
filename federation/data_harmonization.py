#!/usr/bin/env python3
"""Cross-site data harmonization for DICOM/FHIR interoperability.

DICOM metadata normalization, FHIR R4 resource mapping, and vocabulary
harmonization (ICD-10, SNOMED CT, LOINC) across heterogeneous site
systems in multi-site oncology trials.

Framework Dependencies:
    - NumPy 1.24.0+

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    Data harmonization mappings are illustrative and must be validated
    by qualified clinical data managers before any production use.

Version: 1.0.0
Last Updated: February 2026
License: MIT
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DataStandard(Enum):
    """Supported clinical data standards."""

    DICOM = "dicom"
    FHIR_R4 = "fhir_r4"
    HL7_V2 = "hl7_v2"
    CDISC_ODM = "cdisc_odm"


class VocabularySystem(Enum):
    """Clinical vocabulary/terminology systems."""

    ICD10 = "icd10"
    SNOMED_CT = "snomed_ct"
    LOINC = "loinc"
    RXNORM = "rxnorm"
    CPT = "cpt"


class HarmonizationStatus(Enum):
    """Status of a harmonization operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class DICOMMetadata:
    """Standardized DICOM metadata representation.

    Attributes:
        study_uid: Study Instance UID.
        series_uid: Series Instance UID.
        sop_uid: SOP Instance UID.
        modality: Imaging modality (CT, MR, PT, etc.).
        body_part: Body part examined.
        manufacturer: Equipment manufacturer.
        slice_thickness: Slice thickness in mm.
        pixel_spacing: Pixel spacing in mm.
        acquisition_date: Acquisition date string.
        patient_position: Patient position code.
        additional_tags: Extra DICOM tags.
    """

    study_uid: str = ""
    series_uid: str = ""
    sop_uid: str = ""
    modality: str = ""
    body_part: str = ""
    manufacturer: str = ""
    slice_thickness: float = 0.0
    pixel_spacing: Tuple[float, float] = (0.0, 0.0)
    acquisition_date: str = ""
    patient_position: str = "HFS"
    additional_tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FHIRResource:
    """Simplified FHIR R4 resource representation.

    Attributes:
        resource_type: FHIR resource type (Patient, Observation, etc.).
        resource_id: Resource identifier.
        status: Resource status.
        code: Primary code (system, code, display).
        subject: Reference to subject (patient).
        effective_datetime: Effective date/time.
        value: Resource value (quantity, codeable concept, etc.).
        extensions: FHIR extensions.
    """

    resource_type: str = ""
    resource_id: str = ""
    status: str = "final"
    code: Dict[str, str] = field(default_factory=dict)
    subject: str = ""
    effective_datetime: str = ""
    value: Dict[str, Any] = field(default_factory=dict)
    extensions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VocabularyMapping:
    """Mapping between clinical vocabulary codes.

    Attributes:
        source_system: Source vocabulary system.
        source_code: Source code.
        source_display: Source display text.
        target_system: Target vocabulary system.
        target_code: Target code.
        target_display: Target display text.
        equivalence: Mapping equivalence level.
    """

    source_system: str = ""
    source_code: str = ""
    source_display: str = ""
    target_system: str = ""
    target_code: str = ""
    target_display: str = ""
    equivalence: str = "equivalent"


@dataclass
class HarmonizationResult:
    """Result of a data harmonization operation.

    Attributes:
        result_id: Unique result identifier.
        source_standard: Original data standard.
        target_standard: Harmonized data standard.
        records_processed: Number of records processed.
        records_harmonized: Successfully harmonized records.
        records_failed: Failed harmonization records.
        vocabulary_mappings_applied: Number of vocabulary mappings.
        warnings: List of harmonization warnings.
        status: Overall harmonization status.
    """

    result_id: str = ""
    source_standard: str = ""
    target_standard: str = ""
    records_processed: int = 0
    records_harmonized: int = 0
    records_failed: int = 0
    vocabulary_mappings_applied: int = 0
    warnings: List[str] = field(default_factory=list)
    status: HarmonizationStatus = HarmonizationStatus.PENDING


# ---------------------------------------------------------------------------
# Vocabulary Harmonizer
# ---------------------------------------------------------------------------

# Illustrative cross-vocabulary mappings for oncology
ICD10_TO_SNOMED: Dict[str, VocabularyMapping] = {
    "C50.9": VocabularyMapping(
        "ICD-10", "C50.9", "Breast cancer, unspecified", "SNOMED-CT", "254837009", "Malignant neoplasm of breast"
    ),
    "C34.9": VocabularyMapping(
        "ICD-10", "C34.9", "Lung cancer, unspecified", "SNOMED-CT", "93880001", "Primary malignant neoplasm of lung"
    ),
    "C18.9": VocabularyMapping(
        "ICD-10", "C18.9", "Colon cancer, unspecified", "SNOMED-CT", "363406005", "Malignant tumor of colon"
    ),
    "C61": VocabularyMapping(
        "ICD-10", "C61", "Prostate cancer", "SNOMED-CT", "399068003", "Malignant tumor of prostate"
    ),
    "C71.9": VocabularyMapping(
        "ICD-10", "C71.9", "Brain cancer, unspecified", "SNOMED-CT", "126952004", "Neoplasm of brain"
    ),
    "C25.9": VocabularyMapping(
        "ICD-10", "C25.9", "Pancreatic cancer", "SNOMED-CT", "363418001", "Malignant tumor of pancreas"
    ),
}

LOINC_TUMOR_MARKERS: Dict[str, Dict[str, str]] = {
    "CEA": {"loinc": "2039-6", "display": "Carcinoembryonic Ag [Mass/Vol]", "unit": "ng/mL"},
    "PSA": {"loinc": "2857-1", "display": "Prostate specific Ag [Mass/Vol]", "unit": "ng/mL"},
    "CA125": {"loinc": "10334-1", "display": "Cancer Ag 125 [Units/Vol]", "unit": "U/mL"},
    "CA19-9": {"loinc": "24108-3", "display": "Cancer Ag 19-9 [Units/Vol]", "unit": "U/mL"},
    "AFP": {"loinc": "1834-1", "display": "Alpha-1-fetoprotein [Mass/Vol]", "unit": "ng/mL"},
    "HER2": {"loinc": "48676-1", "display": "HER2 [Interpretation]", "unit": ""},
}


class VocabularyHarmonizer:
    """Harmonizes clinical vocabulary across sites.

    Maps between ICD-10, SNOMED CT, LOINC, and RxNorm codes
    used by different site EHR systems.
    """

    def __init__(self):
        self.icd10_to_snomed = ICD10_TO_SNOMED
        self.loinc_markers = LOINC_TUMOR_MARKERS
        self.custom_mappings: Dict[str, VocabularyMapping] = {}
        self.mapping_log: List[Dict[str, Any]] = []

    def map_icd10_to_snomed(self, icd10_code: str) -> Optional[VocabularyMapping]:
        """Map an ICD-10 code to SNOMED CT.

        Args:
            icd10_code: ICD-10 diagnosis code.

        Returns:
            VocabularyMapping if found, else None.
        """
        mapping = self.icd10_to_snomed.get(icd10_code)
        if mapping is not None:
            self.mapping_log.append({"source": icd10_code, "target": mapping.target_code, "type": "icd10_to_snomed"})
        return mapping

    def get_loinc_for_marker(self, marker_name: str) -> Optional[Dict[str, str]]:
        """Get LOINC code for a tumor marker.

        Args:
            marker_name: Common tumor marker name.

        Returns:
            LOINC code information, or None if not found.
        """
        return self.loinc_markers.get(marker_name.upper())

    def add_custom_mapping(self, mapping: VocabularyMapping) -> None:
        """Add a custom vocabulary mapping.

        Args:
            mapping: Custom mapping to add.
        """
        key = f"{mapping.source_system}:{mapping.source_code}"
        self.custom_mappings[key] = mapping

    def harmonize_code(self, system: str, code: str) -> Optional[VocabularyMapping]:
        """Attempt to harmonize a code using all available mappings.

        Args:
            system: Source vocabulary system.
            code: Source code.

        Returns:
            Mapping if found, else None.
        """
        custom_key = f"{system}:{code}"
        if custom_key in self.custom_mappings:
            return self.custom_mappings[custom_key]

        if system.lower() in ("icd-10", "icd10"):
            return self.map_icd10_to_snomed(code)

        return None


# ---------------------------------------------------------------------------
# DICOM Normalizer
# ---------------------------------------------------------------------------


class DICOMNormalizer:
    """Normalizes DICOM metadata across sites.

    Handles manufacturer-specific tag variations, unit conversions,
    and missing value imputation for cross-site consistency.
    """

    MODALITY_NORMALIZATION = {
        "CT": "CT",
        "MR": "MR",
        "MRI": "MR",
        "PT": "PT",
        "PET": "PT",
        "NM": "NM",
        "US": "US",
        "DX": "DX",
        "CR": "CR",
        "XA": "XA",
        "MG": "MG",
        "RTSTRUCT": "RTSTRUCT",
        "RTPLAN": "RTPLAN",
        "RTDOSE": "RTDOSE",
    }

    BODY_PART_NORMALIZATION = {
        "CHEST": "CHEST",
        "THORAX": "CHEST",
        "LUNG": "CHEST",
        "ABDOMEN": "ABDOMEN",
        "ABD": "ABDOMEN",
        "PELVIS": "PELVIS",
        "HEAD": "HEAD",
        "BRAIN": "HEAD",
        "BREAST": "BREAST",
        "EXTREMITY": "EXTREMITY",
        "SPINE": "SPINE",
        "NECK": "NECK",
        "WHOLEBODY": "WHOLE_BODY",
        "WHOLE BODY": "WHOLE_BODY",
    }

    def normalize(self, metadata: DICOMMetadata) -> Tuple[DICOMMetadata, List[str]]:
        """Normalize DICOM metadata to a standard format.

        Args:
            metadata: Raw DICOM metadata.

        Returns:
            Tuple of (normalized_metadata, warnings).
        """
        warnings = []
        normalized = DICOMMetadata(
            study_uid=metadata.study_uid,
            series_uid=metadata.series_uid,
            sop_uid=metadata.sop_uid,
            acquisition_date=metadata.acquisition_date,
            manufacturer=metadata.manufacturer,
            additional_tags=dict(metadata.additional_tags),
        )

        upper_modality = metadata.modality.upper().strip()
        normalized.modality = self.MODALITY_NORMALIZATION.get(upper_modality, upper_modality)
        if upper_modality not in self.MODALITY_NORMALIZATION:
            warnings.append(f"Unknown modality '{metadata.modality}' kept as-is")

        upper_body = metadata.body_part.upper().strip()
        normalized.body_part = self.BODY_PART_NORMALIZATION.get(upper_body, upper_body)

        if metadata.slice_thickness > 0:
            normalized.slice_thickness = metadata.slice_thickness
        else:
            normalized.slice_thickness = 0.0
            warnings.append("Missing or zero slice thickness")

        px, py = metadata.pixel_spacing
        if px > 0 and py > 0:
            normalized.pixel_spacing = (px, py)
        else:
            normalized.pixel_spacing = (0.0, 0.0)
            warnings.append("Missing or zero pixel spacing")

        normalized.patient_position = metadata.patient_position.upper().strip() if metadata.patient_position else "HFS"

        return normalized, warnings


# ---------------------------------------------------------------------------
# FHIR Resource Mapper
# ---------------------------------------------------------------------------


class FHIRResourceMapper:
    """Maps site-specific data to FHIR R4 resources.

    Creates standardized FHIR R4 resources from heterogeneous
    site data formats for cross-site interoperability.
    """

    def __init__(self, vocabulary_harmonizer: Optional[VocabularyHarmonizer] = None):
        self.vocab = vocabulary_harmonizer or VocabularyHarmonizer()

    def create_condition(
        self,
        patient_ref: str,
        icd10_code: str,
        onset_date: str = "",
        clinical_status: str = "active",
    ) -> FHIRResource:
        """Create a FHIR Condition resource from an ICD-10 code.

        Args:
            patient_ref: Patient reference.
            icd10_code: ICD-10 diagnosis code.
            onset_date: Onset date string.
            clinical_status: Clinical status.

        Returns:
            FHIR Condition resource.
        """
        snomed = self.vocab.map_icd10_to_snomed(icd10_code)

        code_info: Dict[str, str] = {"system": "http://hl7.org/fhir/sid/icd-10", "code": icd10_code}
        if snomed is not None:
            code_info["display"] = snomed.target_display

        return FHIRResource(
            resource_type="Condition",
            resource_id=str(uuid.uuid4())[:8],
            status=clinical_status,
            code=code_info,
            subject=patient_ref,
            effective_datetime=onset_date,
        )

    def create_observation(
        self,
        patient_ref: str,
        marker_name: str,
        value: float,
        date: str = "",
    ) -> FHIRResource:
        """Create a FHIR Observation resource for a tumor marker.

        Args:
            patient_ref: Patient reference.
            marker_name: Tumor marker name (CEA, PSA, etc.).
            value: Measured value.
            date: Observation date.

        Returns:
            FHIR Observation resource.
        """
        loinc_info = self.vocab.get_loinc_for_marker(marker_name)

        code_info: Dict[str, str] = {"display": marker_name}
        unit = ""
        if loinc_info is not None:
            code_info = {
                "system": "http://loinc.org",
                "code": loinc_info["loinc"],
                "display": loinc_info["display"],
            }
            unit = loinc_info.get("unit", "")

        return FHIRResource(
            resource_type="Observation",
            resource_id=str(uuid.uuid4())[:8],
            status="final",
            code=code_info,
            subject=patient_ref,
            effective_datetime=date,
            value={"valueQuantity": {"value": value, "unit": unit}},
        )

    def create_medication_statement(
        self,
        patient_ref: str,
        medication_name: str,
        dose: str = "",
        status: str = "active",
    ) -> FHIRResource:
        """Create a FHIR MedicationStatement resource.

        Args:
            patient_ref: Patient reference.
            medication_name: Medication name.
            dose: Dosage string.
            status: Medication status.

        Returns:
            FHIR MedicationStatement resource.
        """
        return FHIRResource(
            resource_type="MedicationStatement",
            resource_id=str(uuid.uuid4())[:8],
            status=status,
            code={"display": medication_name},
            subject=patient_ref,
            value={"dosage": dose} if dose else {},
        )


# ---------------------------------------------------------------------------
# Data Harmonization Engine
# ---------------------------------------------------------------------------


class DataHarmonizationEngine:
    """Orchestrates cross-site data harmonization.

    Combines DICOM normalization, FHIR mapping, and vocabulary
    harmonization for multi-site trial data consistency.

    Args:
        trial_id: Clinical trial identifier.
    """

    def __init__(self, trial_id: str = ""):
        self.trial_id = trial_id or str(uuid.uuid4())[:8]
        self.dicom_normalizer = DICOMNormalizer()
        self.vocab_harmonizer = VocabularyHarmonizer()
        self.fhir_mapper = FHIRResourceMapper(self.vocab_harmonizer)
        self.harmonization_log: List[Dict[str, Any]] = []

    def harmonize_dicom_series(self, site_id: str, metadata_list: List[DICOMMetadata]) -> HarmonizationResult:
        """Harmonize a batch of DICOM metadata from a site.

        Args:
            site_id: Source site identifier.
            metadata_list: List of DICOM metadata to harmonize.

        Returns:
            HarmonizationResult with processing summary.
        """
        result = HarmonizationResult(
            result_id=str(uuid.uuid4())[:8],
            source_standard="DICOM",
            target_standard="DICOM_normalized",
            status=HarmonizationStatus.IN_PROGRESS,
        )

        for metadata in metadata_list:
            result.records_processed += 1
            _, warnings = self.dicom_normalizer.normalize(metadata)
            if warnings:
                result.warnings.extend(warnings)
            result.records_harmonized += 1

        result.status = HarmonizationStatus.COMPLETED
        self._log_event("dicom_harmonized", {"site_id": site_id, "records": result.records_processed})

        logger.info("Harmonized %d DICOM records from site %s", result.records_processed, site_id)
        return result

    def harmonize_diagnoses(
        self, site_id: str, diagnoses: List[Dict[str, str]]
    ) -> Tuple[List[FHIRResource], HarmonizationResult]:
        """Harmonize diagnosis codes from a site.

        Args:
            site_id: Source site identifier.
            diagnoses: List of diagnosis records with 'code' and 'patient_ref'.

        Returns:
            Tuple of (FHIR Condition resources, HarmonizationResult).
        """
        result = HarmonizationResult(
            result_id=str(uuid.uuid4())[:8],
            source_standard="ICD-10",
            target_standard="FHIR_R4",
            status=HarmonizationStatus.IN_PROGRESS,
        )
        resources = []

        for dx in diagnoses:
            result.records_processed += 1
            code = dx.get("code", "")
            patient_ref = dx.get("patient_ref", "")

            condition = self.fhir_mapper.create_condition(patient_ref, code)
            resources.append(condition)
            result.records_harmonized += 1

            mapping = self.vocab_harmonizer.map_icd10_to_snomed(code)
            if mapping is not None:
                result.vocabulary_mappings_applied += 1

        result.status = HarmonizationStatus.COMPLETED
        self._log_event(
            "diagnoses_harmonized",
            {"site_id": site_id, "records": result.records_processed, "mappings": result.vocabulary_mappings_applied},
        )

        logger.info("Harmonized %d diagnoses from site %s", result.records_processed, site_id)
        return resources, result

    def harmonize_lab_results(
        self, site_id: str, labs: List[Dict[str, Any]]
    ) -> Tuple[List[FHIRResource], HarmonizationResult]:
        """Harmonize lab results from a site.

        Args:
            site_id: Source site identifier.
            labs: List of lab results with marker_name, value, patient_ref.

        Returns:
            Tuple of (FHIR Observation resources, HarmonizationResult).
        """
        result = HarmonizationResult(
            result_id=str(uuid.uuid4())[:8],
            source_standard="site_specific",
            target_standard="FHIR_R4",
            status=HarmonizationStatus.IN_PROGRESS,
        )
        resources = []

        for lab in labs:
            result.records_processed += 1
            marker = lab.get("marker_name", "")
            value = lab.get("value", 0.0)
            patient_ref = lab.get("patient_ref", "")

            obs = self.fhir_mapper.create_observation(patient_ref, marker, value)
            resources.append(obs)
            result.records_harmonized += 1

            if self.vocab_harmonizer.get_loinc_for_marker(marker) is not None:
                result.vocabulary_mappings_applied += 1

        result.status = HarmonizationStatus.COMPLETED
        self._log_event(
            "labs_harmonized",
            {"site_id": site_id, "records": result.records_processed},
        )

        logger.info("Harmonized %d lab results from site %s", result.records_processed, site_id)
        return resources, result

    def get_harmonization_summary(self) -> Dict[str, Any]:
        """Generate a summary of all harmonization operations.

        Returns:
            Dictionary with harmonization metrics.
        """
        return {
            "trial_id": self.trial_id,
            "total_events": len(self.harmonization_log),
            "event_types": list({e["event_type"] for e in self.harmonization_log}),
        }

    def _log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Append an event to the harmonization log."""
        self.harmonization_log.append(
            {
                "event_id": str(uuid.uuid4())[:8],
                "event_type": event_type,
                "details": details,
            }
        )


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = DataHarmonizationEngine(trial_id="ONCO-FED-001")

    dicom_batch = [
        DICOMMetadata(modality="MRI", body_part="brain", slice_thickness=1.0, pixel_spacing=(0.5, 0.5)),
        DICOMMetadata(modality="PET", body_part="wholebody", slice_thickness=3.0, pixel_spacing=(4.0, 4.0)),
        DICOMMetadata(modality="CT", body_part="thorax", slice_thickness=1.25, pixel_spacing=(0.7, 0.7)),
    ]
    result = engine.harmonize_dicom_series("site_001", dicom_batch)
    print(f"DICOM harmonization: {result.records_harmonized}/{result.records_processed}")

    diagnoses = [
        {"code": "C50.9", "patient_ref": "patient/001"},
        {"code": "C34.9", "patient_ref": "patient/002"},
        {"code": "C61", "patient_ref": "patient/003"},
    ]
    fhir_conditions, dx_result = engine.harmonize_diagnoses("site_001", diagnoses)
    print(
        f"Diagnosis harmonization: {dx_result.records_harmonized} records, {dx_result.vocabulary_mappings_applied} mappings"
    )

    labs = [
        {"marker_name": "CEA", "value": 5.2, "patient_ref": "patient/001"},
        {"marker_name": "PSA", "value": 4.1, "patient_ref": "patient/003"},
    ]
    fhir_obs, lab_result = engine.harmonize_lab_results("site_001", labs)
    print(f"Lab harmonization: {lab_result.records_harmonized} records")
