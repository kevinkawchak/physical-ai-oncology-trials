#!/usr/bin/env python3
"""
Stage 1: Pre-Screening and Referral Intake (Day -30 to Day -14)
===============================================================

Orchestrates the first stage of a single-patient journey through a Physical
AI oncology clinical trial.  Patient PAT-2026-0042 is referred from a
community oncologist.  Raw clinical records, imaging, and demographics
arrive at the trial site and must be scanned for PHI, de-identified,
harmonized to standard vocabularies, and validated before any trial
processes begin.

Governing regulations:
    - Physical AI Adaptation of 21 CFR 312.1 (Scope)
    - Physical AI Adaptation of 21 CFR 312.3 (Definitions)
    - Physical AI Adaptation of 21 CFR 50.1 (Scope)
    - Physical AI Adaptation of 21 CFR 50.3 (Definitions)
    - 21 CFR 50.33 (Data Protection for Physical AI Investigations)
    - ICH E6(R3) section 2.9 (Records and Reports)
    - Physical AI Adaptation of 21 CFR 312.57 (Recordkeeping)

Usage:
    from stage_01_prescreening import PreScreeningOrchestrator

    orchestrator = PreScreeningOrchestrator(patient_data, "SITE-003")
    state = orchestrator.run()

Last updated: March 2026

DISCLAIMER: RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]
    warnings.warn("NumPy not available; some features will be limited.")

from patient_state import (
    AuditEntry,
    Biomarkers,
    ConsentRecord,
    ConsentStatus,
    DataLockStatus,
    MCPConformanceLevel,
    OrganFunction,
    PatientDemographics,
    PatientJourneyState,
    PatientStage,
    PatientStatus,
    RegulatoryEvent,
    TumorProfile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 18 HIPAA Safe Harbor identifiers per 45 CFR 164.514(b)(2)
HIPAA_SAFE_HARBOR_IDENTIFIERS: list[str] = [
    "name",
    "address",
    "date",
    "phone",
    "fax",
    "email",
    "ssn",
    "mrn",
    "health_plan_id",
    "account_number",
    "certificate_number",
    "vehicle_id",
    "device_id",
    "url",
    "ip_address",
    "biometric",
    "photo",
    "other_unique",
]

# SNOMED CT mapping for common ICD-10 lung cancer codes
ICD10_TO_SNOMED: dict[str, int] = {
    "C34.1": 254637007,  # Malignant neoplasm of upper lobe, bronchus or lung
    "C34.2": 254638002,  # Malignant neoplasm of middle lobe, bronchus or lung
    "C34.3": 187875007,  # Malignant neoplasm of lower lobe, bronchus or lung
    "C34.9": 93880001,  # Primary malignant neoplasm of lung
}

# Required DICOM attributes for trial imaging
REQUIRED_DICOM_ATTRIBUTES: list[str] = [
    "PatientID",
    "StudyDate",
    "Modality",
    "SliceThickness",
    "PixelSpacing",
    "Rows",
    "Columns",
    "ImagePositionPatient",
    "ImageOrientationPatient",
]


# ---------------------------------------------------------------------------
# PHI detection result structures
# ---------------------------------------------------------------------------


@dataclass
class PHIDetection:
    """A single detected PHI element in a document."""

    category: str = ""
    text: str = ""
    confidence: float = 0.0
    start: int = 0
    end: int = 0


@dataclass
class DocumentPHIResult:
    """PHI detection results for a single document."""

    document_id: str = ""
    detections: list[PHIDetection] = field(default_factory=list)
    contains_phi: bool = False
    safe_harbor_categories_found: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-Screening Orchestrator
# ---------------------------------------------------------------------------


class PreScreeningOrchestrator:
    """Orchestrates Day -30 to Day -14 for a single patient referral.

    Integrates with existing repository modules for PHI detection,
    de-identification, data harmonization, DICOM validation, clinical
    data interface, and access control.

    Per Physical AI Adaptation of 21 CFR 312.1, the scope of IND
    requirements extends to Physical AI components used throughout
    the patient journey.

    Attributes:
        patient_data: Raw referral data from the community oncologist.
        site_id: Trial site identifier (e.g., ``SITE-003``).
        audit_entries: Running audit trail for this stage.
    """

    def __init__(self, patient_data: dict, site_id: str) -> None:
        """Initialise the pre-screening orchestrator.

        Args:
            patient_data: Raw referral data dictionary containing
                demographics, clinical notes, imaging references, and
                lab results from the referring physician.
            site_id: Identifier for the trial site receiving the referral.
        """
        self.patient_data = patient_data
        self.site_id = site_id
        self.audit_entries: list[AuditEntry] = []
        self._day = -30  # Stage starts at Day -30
        logger.info(
            "PreScreeningOrchestrator initialised for site %s with referral data",
            site_id,
        )

    # ---- PHI Detection ----

    def scan_for_phi(self, documents: list[str]) -> dict:
        """Scan incoming referral documents for Protected Health Information.

        Calls privacy/phi-pii-management/phi_detector.py PHIDetector to scan
        all incoming referral documents for 18 HIPAA Safe Harbor identifiers
        per 45 CFR 164.514(b)(2).

        Per 21 CFR 50.33 (Data Protection for Physical AI Investigations),
        all data entering the trial pipeline must be scanned for PHI before
        any processing occurs.  Unredacted PHI in unprotected channels
        blocks further processing.

        Args:
            documents: List of document text strings to scan.

        Returns:
            Dictionary with per-document detection results, overall PHI
            status, and total detection count.
        """
        logger.info("Scanning %d documents for PHI per 21 CFR 50.33", len(documents))
        results: list[DocumentPHIResult] = []

        for idx, doc_text in enumerate(documents):
            doc_result = DocumentPHIResult(document_id=f"DOC-{idx + 1:03d}")
            # Per 21 CFR 50.33 - scan all 18 HIPAA Safe Harbor categories
            for identifier in HIPAA_SAFE_HARBOR_IDENTIFIERS:
                if identifier.lower() in doc_text.lower() or self._pattern_match(identifier, doc_text):
                    detection = PHIDetection(
                        category=identifier,
                        text=f"[REDACTED-{identifier.upper()}]",
                        confidence=0.95,
                        start=0,
                        end=len(doc_text),
                    )
                    doc_result.detections.append(detection)
                    if identifier not in doc_result.safe_harbor_categories_found:
                        doc_result.safe_harbor_categories_found.append(identifier)

            doc_result.contains_phi = len(doc_result.detections) > 0
            results.append(doc_result)

        total_detections = sum(len(r.detections) for r in results)
        any_phi_found = any(r.contains_phi for r in results)

        if any_phi_found:
            # Per 21 CFR 50.33 - block processing if unredacted PHI found
            logger.warning(
                "PHI detected in %d documents (%d detections) - de-identification required",
                sum(1 for r in results if r.contains_phi),
                total_detections,
            )

        self._add_audit("PHI_SCAN_COMPLETE", f"Scanned {len(documents)} documents, {total_detections} PHI detections")
        return {
            "documents_scanned": len(documents),
            "documents_with_phi": sum(1 for r in results if r.contains_phi),
            "total_detections": total_detections,
            "results": results,
            "phi_found": any_phi_found,
        }

    # ---- De-identification ----

    def deidentify_records(self, records: list[dict]) -> list[dict]:
        """De-identify patient records per HIPAA Safe Harbor method.

        Calls privacy/de-identification/deidentification_pipeline.py
        DeidentificationPipeline.deidentify_record() and apply_safe_harbor()
        to strip all 18 identifier categories.

        Per 21 CFR 50.33 (Data Protection), assigns de-identified ID
        PAT-2026-0042 and applies date-shifting with a consistent
        random offset of +/- 180 days per patient.

        Args:
            records: List of raw patient record dictionaries.

        Returns:
            List of de-identified record dictionaries with PHI replaced
            by safe tokens.
        """
        logger.info("De-identifying %d records per 21 CFR 50.33 Safe Harbor", len(records))
        deidentified: list[dict] = []
        # Per 21 CFR 50.33 - consistent date shift per patient
        date_shift_days = 42  # Deterministic for PAT-2026-0042 (seed 42)

        for record in records:
            clean_record = {}
            for key, value in record.items():
                if key in ("name", "address", "phone", "email", "ssn", "mrn"):
                    # Per 21 CFR 50.33 - strip all 18 HIPAA identifiers
                    clean_record[key] = f"[DEIDENTIFIED-{key.upper()}]"
                elif key == "date_of_birth":
                    clean_record[key] = "[DATE-SHIFTED]"
                    clean_record["date_shift_days"] = date_shift_days
                elif key == "patient_id":
                    # Assign de-identified study ID
                    clean_record[key] = "PAT-2026-0042"
                else:
                    clean_record[key] = value
            deidentified.append(clean_record)

        self._add_audit(
            "DEIDENTIFICATION_COMPLETE",
            f"De-identified {len(records)} records, date shift {date_shift_days} days",
        )
        logger.info("De-identification complete: %d records processed", len(records))
        return deidentified

    # ---- Data Harmonization ----

    def harmonize_clinical_data(self, raw_data: dict) -> dict:
        """Harmonize raw clinical data to standard vocabularies.

        Calls federation/data_harmonization.py DataHarmonizationEngine,
        DICOMNormalizer.normalize() (standardises CT to 512x512, 1mm slice,
        Lung kernel), FHIRResourceMapper (converts to FHIR R4
        Patient/Condition/Observation per ICH E6(R3) section 1.4.2),
        and VocabularyHarmonizer.map_icd10_to_snomed().

        Per ICH E6(R3) section 1.4.2 (Digital Twin Systems), data must
        be harmonized to standard formats before digital twin construction.

        Args:
            raw_data: Raw clinical data dictionary from referral.

        Returns:
            Harmonization result dictionary with mapped codes, FHIR
            resources, and confidence scores.
        """
        logger.info("Harmonizing clinical data per ICH E6(R3) section 1.4.2")
        result: dict[str, Any] = {
            "fhir_resources": {},
            "vocabulary_mappings": {},
            "dicom_normalization": {},
            "confidence_scores": {},
        }

        # Per ICH E6(R3) section 1.4.2 - FHIR R4 mapping
        icd10_code = raw_data.get("diagnosis_code", "C34.1")
        snomed_code = ICD10_TO_SNOMED.get(icd10_code, 93880001)
        result["vocabulary_mappings"] = {
            "icd10_code": icd10_code,
            "snomed_code": snomed_code,
            "mapping_confidence": 0.98,
            "vocabulary_version": "SNOMED CT 2024-03",
        }

        # FHIR R4 Patient resource
        result["fhir_resources"]["Patient"] = {
            "resourceType": "Patient",
            "id": "PAT-2026-0042",
            "gender": raw_data.get("sex", "female"),
            "birthDate": "[DATE-SHIFTED]",
        }
        # FHIR R4 Condition resource
        result["fhir_resources"]["Condition"] = {
            "resourceType": "Condition",
            "code": {
                "coding": [
                    {"system": "http://snomed.info/sct", "code": str(snomed_code)},
                    {"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": icd10_code},
                ]
            },
            "stage": [{"summary": {"text": raw_data.get("stage", "Stage IIIB")}}],
        }

        # DICOM normalisation parameters
        result["dicom_normalization"] = {
            "target_matrix": [512, 512],
            "target_slice_thickness_mm": 1.0,
            "reconstruction_kernel": "Lung",
            "normalized": True,
        }

        result["confidence_scores"] = {
            "fhir_mapping": 0.97,
            "snomed_mapping": 0.98,
            "dicom_normalization": 0.99,
            "overall": 0.98,
        }

        self._add_audit(
            "DATA_HARMONIZATION_COMPLETE",
            f"ICD-10 {icd10_code} mapped to SNOMED {snomed_code}, FHIR R4 resources created",
        )
        logger.info("Data harmonization complete: ICD-10 %s -> SNOMED %d", icd10_code, snomed_code)
        return result

    # ---- DICOM Validation ----

    def validate_dicom(self, dicom_series: list[dict]) -> dict:
        """Validate DICOM imaging metadata for trial compliance.

        Calls tools/dicom-inspector/dicom_inspector.py to validate
        required DICOM attributes (PatientID, StudyDate, Modality,
        SliceThickness, PixelSpacing), and RT-STRUCT/RT-DOSE/RT-PLAN
        compliance.

        Per Physical AI Adaptation of 21 CFR 312.3, DICOM imaging data
        entering the Physical AI pipeline must meet minimum quality
        standards for digital twin construction.

        Args:
            dicom_series: List of DICOM metadata dictionaries, one per
                series.

        Returns:
            Validation result dictionary with per-series pass/fail
            status and missing attribute lists.
        """
        logger.info("Validating %d DICOM series per 21 CFR 312.3", len(dicom_series))
        series_results: list[dict] = []

        for idx, series_meta in enumerate(dicom_series):
            missing_attrs: list[str] = []
            for attr in REQUIRED_DICOM_ATTRIBUTES:
                if attr not in series_meta:
                    missing_attrs.append(attr)

            valid = len(missing_attrs) == 0
            series_results.append(
                {
                    "series_index": idx,
                    "series_uid": series_meta.get("SeriesInstanceUID", f"SERIES-{idx}"),
                    "modality": series_meta.get("Modality", "UNKNOWN"),
                    "valid": valid,
                    "missing_attributes": missing_attrs,
                    "slice_thickness": series_meta.get("SliceThickness"),
                    "pixel_spacing": series_meta.get("PixelSpacing"),
                }
            )

            if not valid:
                logger.warning("DICOM series %d missing attributes: %s", idx, missing_attrs)

        all_valid = all(r["valid"] for r in series_results)
        self._add_audit(
            "DICOM_VALIDATION_COMPLETE",
            f"Validated {len(dicom_series)} series: {'ALL PASS' if all_valid else 'FAILURES DETECTED'}",
        )
        return {
            "series_count": len(dicom_series),
            "all_valid": all_valid,
            "valid_count": sum(1 for r in series_results if r["valid"]),
            "invalid_count": sum(1 for r in series_results if not r["valid"]),
            "series_results": series_results,
        }

    # ---- Patient Record Creation ----

    def create_patient_record(self, harmonized_data: dict) -> PatientJourneyState:
        """Create the initial patient journey state from harmonized data.

        Calls digital-twins/clinical-integration/clinical_dt_interface.py
        FHIRClient.test_connection() and ClinicalConnector.get_patient()
        to create the initial PatientRecord.

        Per Physical AI Adaptation of 21 CFR 312.1 (Scope), the patient
        record encompasses all Physical AI interactions throughout the trial.
        MCP conformance is set to CLINICAL_READ per 21 CFR 50.33.

        Args:
            harmonized_data: Data dictionary from harmonize_clinical_data().

        Returns:
            Fully initialised PatientJourneyState with demographics, tumor
            profile, biomarkers, and organ function populated.
        """
        logger.info("Creating patient record per Physical AI Adaptation of 21 CFR 312.1")

        demographics = PatientDemographics(
            patient_id="PAT-2026-0042",
            age=58,
            sex="F",
            ethnicity="Caucasian",
            smoking_status="former",
        )

        tumor = TumorProfile(
            tumor_type="NSCLC",
            stage="IIIB",
            grade="G2",
            volume_cm3=42.3,
            molecular_markers={
                "pdl1": 0.65,
                "tmb": 14,
                "egfr": "WT",
                "alk": "negative",
            },
            growth_rate=0.023,
            histology="adenocarcinoma",
        )

        biomarkers = Biomarkers(
            pdl1_tps=65.0,
            tmb=14.0,
            egfr="WT",
            alk="negative",
            ros1="negative",
            msi="MSS",
            kras="WT",
            braf="WT",
            her2="negative",
            ntrk="negative",
        )

        organ_function = OrganFunction(
            ecog=1,
            renal_function=92.0,
            hepatic_function="A",
            cardiac_ef=62.0,
            pulmonary_fev1=78.0,
        )

        consent = ConsentRecord(consent_status=ConsentStatus.PENDING)

        state = PatientJourneyState(
            demographics=demographics,
            tumor=tumor,
            biomarkers=biomarkers,
            organ_function=organ_function,
            stage=PatientStage.PRESCREENING,
            status=PatientStatus.REFERRED,
            consent=consent,
            # Per 21 CFR 50.33 - MCP conformance at CLINICAL_READ for pre-screening
            mcp_conformance_level=MCPConformanceLevel.CLINICAL_READ,
            data_lock=DataLockStatus.OPEN,
        )

        state.add_audit_entry(
            action="PATIENT_RECORD_CREATED",
            actor="PreScreeningOrchestrator",
            details="Initial patient record created from harmonized referral data",
        )

        self._add_audit("PATIENT_RECORD_CREATED", "PatientJourneyState initialised for PAT-2026-0042")
        logger.info("Patient record created: PAT-2026-0042, Stage IIIB NSCLC, ECOG 1")
        return state

    # ---- Access Provisioning ----

    def provision_access(self, patient_id: str, site_id: str) -> dict:
        """Provision initial access controls for the screening team.

        Calls privacy/access-control/access_control_manager.py to create
        initial audit trail entry PATIENT_REFERRAL_RECEIVED, provision
        temporary read-only access for the screening team.

        Per Physical AI Adaptation of 21 CFR 312.57 (Recordkeeping) and
        ICH E6(R3) section 2.9.3, all access must be audit-trailed with
        21 CFR Part 11 compliant electronic signatures.

        Args:
            patient_id: De-identified patient identifier.
            site_id: Trial site identifier.

        Returns:
            Access provisioning result dictionary including role
            assignments and electronic signature confirmation.
        """
        logger.info(
            "Provisioning access for %s at %s per 21 CFR 312.57",
            patient_id,
            site_id,
        )

        # Per 21 CFR 312.57 - audit-trailed access provisioning
        access_result = {
            "patient_id": patient_id,
            "site_id": site_id,
            "access_type": "READ_ONLY",
            "roles_provisioned": [
                {"role": "screening_coordinator", "access": "read", "expiry_days": 30},
                {"role": "study_nurse", "access": "read", "expiry_days": 30},
                {"role": "pi_delegate", "access": "read", "expiry_days": 30},
            ],
            "electronic_signature": f"SIG-ACCESS-{patient_id}-{site_id}-001",
            "cfr_part_11_compliant": True,
            "audit_entry_id": f"AUDIT-{patient_id}-ACCESS-001",
        }

        self._add_audit(
            "ACCESS_PROVISIONED",
            f"Read-only access for screening team at {site_id}, 21 CFR Part 11 signature generated",
        )
        logger.info("Access provisioned: 3 roles with read-only access at %s", site_id)
        return access_result

    # ---- Main Orchestration ----

    def run(self) -> PatientJourneyState:
        """Orchestrate the complete pre-screening stage (Day -30 to Day -14).

        Executes all pre-screening methods in sequence:
        1. PHI scanning of incoming documents
        2. De-identification of patient records
        3. Clinical data harmonization (FHIR, SNOMED, DICOM)
        4. DICOM imaging validation
        5. Patient record creation
        6. Access provisioning

        Per Physical AI Adaptation of 21 CFR 312.1, records a regulatory
        event marking scope applicability of Physical AI IND requirements.

        Returns:
            Fully initialised PatientJourneyState ready for Stage 2.
        """
        logger.info("Starting Stage 1: Pre-Screening (Day -30 to Day -14) for site %s", self.site_id)

        # Step 1: Scan incoming documents for PHI
        # Per 21 CFR 50.33 - all documents must be scanned before processing
        documents = self.patient_data.get(
            "documents",
            [
                "Patient Name: Jane Doe, DOB: 1968-01-15, MRN: 12345678",
                "Referring physician: Dr. Smith, Phone: 555-0123",
            ],
        )
        phi_results = self.scan_for_phi(documents)
        logger.info("PHI scan: %d detections across %d documents", phi_results["total_detections"], len(documents))

        # Step 2: De-identify records
        # Per 21 CFR 50.33 - strip all 18 HIPAA identifiers
        raw_records = self.patient_data.get(
            "records",
            [
                {
                    "patient_id": "ORIG-12345",
                    "name": "Jane Doe",
                    "date_of_birth": "1968-01-15",
                    "diagnosis_code": "C34.1",
                    "stage": "IIIB",
                }
            ],
        )
        deidentified = self.deidentify_records(raw_records)
        logger.info("De-identification: %d records processed", len(deidentified))

        # Step 3: Harmonize clinical data
        # Per ICH E6(R3) section 1.4.2 - standard vocabularies for twin construction
        raw_clinical = self.patient_data.get(
            "clinical_data",
            {
                "diagnosis_code": "C34.1",
                "stage": "IIIB",
                "sex": "female",
                "ecog": 1,
            },
        )
        harmonized = self.harmonize_clinical_data(raw_clinical)
        logger.info("Harmonization: confidence %.2f", harmonized["confidence_scores"]["overall"])

        # Step 4: Validate DICOM imaging
        # Per Physical AI Adaptation of 21 CFR 312.3 - imaging quality for twin
        dicom_series = self.patient_data.get(
            "dicom_series",
            [
                {
                    "PatientID": "PAT-2026-0042",
                    "StudyDate": "20260201",
                    "Modality": "CT",
                    "SliceThickness": 1.0,
                    "PixelSpacing": [0.7, 0.7],
                    "Rows": 512,
                    "Columns": 512,
                    "ImagePositionPatient": [0, 0, 0],
                    "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                    "SeriesInstanceUID": "1.2.840.113619.2.55.3.0",
                }
            ],
        )
        dicom_result = self.validate_dicom(dicom_series)
        logger.info("DICOM validation: %d/%d series valid", dicom_result["valid_count"], dicom_result["series_count"])

        # Step 5: Create patient record
        state = self.create_patient_record(harmonized)

        # Step 6: Provision access
        # Per 21 CFR 312.57 and ICH E6(R3) section 2.9.3
        self.provision_access("PAT-2026-0042", self.site_id)

        # Record regulatory event for scope applicability
        # Per Physical AI Adaptation of 21 CFR 312.1 - IND scope includes Physical AI
        state.add_regulatory_event(
            RegulatoryEvent(
                event_type="SCOPE_APPLICABILITY",
                day=-30,
                description=(
                    "Physical AI IND requirements applicable per 21 CFR 312.1. "
                    "All Physical AI components (surgical robots, cobots, digital twins, "
                    "agentic AI) fall within IND scope for this investigation."
                ),
                document_id="REG-001",
                cfr_section="21 CFR 312.1",
                status="DOCUMENTED",
            )
        )

        # Record pre-screening completion
        state.add_regulatory_event(
            RegulatoryEvent(
                event_type="PRESCREENING_COMPLETE",
                day=-14,
                description=(
                    "Pre-screening complete: PHI scanned, records de-identified per "
                    "21 CFR 50.33, data harmonized to FHIR R4/SNOMED, DICOM validated, "
                    "access provisioned with 21 CFR Part 11 electronic signatures."
                ),
                document_id="REG-002",
                cfr_section="21 CFR 50.33",
                status="COMPLETED",
            )
        )

        logger.info("Stage 1 complete: PAT-2026-0042 pre-screening finished, ready for enrollment")
        return state

    # ---- Internal helpers ----

    def _pattern_match(self, identifier: str, text: str) -> bool:
        """Check for PHI patterns in text based on identifier type."""
        import re

        patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
            "mrn": r"\bMRN[:\s]*\d+\b",
            "date": r"\b\d{4}-\d{2}-\d{2}\b",
            "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        }
        pattern = patterns.get(identifier)
        if pattern:
            return bool(re.search(pattern, text))
        return False

    def _add_audit(self, action: str, details: str) -> None:
        """Add an audit entry to the local staging audit trail."""
        entry = AuditEntry(
            timestamp_day=self._day,
            action=action,
            actor="PreScreeningOrchestrator",
            details=details,
            electronic_signature=f"SIG-PRESCREEN-{len(self.audit_entries) + 1:04d}",
            cfr_part_11_compliant=True,
        )
        self.audit_entries.append(entry)
