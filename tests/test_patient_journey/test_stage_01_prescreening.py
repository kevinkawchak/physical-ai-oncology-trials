#!/usr/bin/env python3
"""
Tests for Stage 1: Pre-Screening & Referral Intake
===================================================

Validates the patient state data model and pre-screening orchestrator
for Day -30 to Day -14 of the patient journey.  Uses the importlib
load_module pattern from tests/conftest.py for hyphenated directory
imports.

Last updated: March 2026
DISCLAIMER: RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module loading (following tests/conftest.py load_module pattern)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_module(name: str, relative_path: str):
    """Load a Python module by file-path from the project root."""
    if name in sys.modules:
        return sys.modules[name]
    filepath = PROJECT_ROOT / relative_path
    if not filepath.exists():
        pytest.skip(f"Source file not found: {relative_path}")
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        sys.modules.pop(name, None)
        pytest.skip(f"Module {name!r} requires dependency not installed: {exc}")
    return mod


# Load patient_state and stage_01_prescreening modules
patient_state_mod = load_module("patient_state", "patient-journey/patient_state.py")
stage_01_mod = load_module("stage_01_prescreening", "patient-journey/stage_01_prescreening.py")

# Import classes from loaded modules
PatientJourneyState = patient_state_mod.PatientJourneyState
PatientStage = patient_state_mod.PatientStage
PatientStatus = patient_state_mod.PatientStatus
PatientDemographics = patient_state_mod.PatientDemographics
TumorProfile = patient_state_mod.TumorProfile
Biomarkers = patient_state_mod.Biomarkers
OrganFunction = patient_state_mod.OrganFunction
ConsentRecord = patient_state_mod.ConsentRecord
ConsentStatus = patient_state_mod.ConsentStatus
DataLockStatus = patient_state_mod.DataLockStatus
TreatmentArm = patient_state_mod.TreatmentArm
ResponseCategory = patient_state_mod.ResponseCategory
AESeverity = patient_state_mod.AESeverity
PhysicalAIClassification = patient_state_mod.PhysicalAIClassification
USLReadinessLevel = patient_state_mod.USLReadinessLevel
MCPConformanceLevel = patient_state_mod.MCPConformanceLevel
AdverseEvent = patient_state_mod.AdverseEvent
RegulatoryEvent = patient_state_mod.RegulatoryEvent
PreScreeningOrchestrator = stage_01_mod.PreScreeningOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def seed_rng():
    """Seed NumPy RNG for deterministic tests."""
    np.random.seed(42)


@pytest.fixture()
def sample_referral_data() -> dict:
    """Synthetic patient referral data matching PAT-2026-0042."""
    return {
        "patient_id": "ORIG-12345",
        "name": "Jane Doe",
        "date_of_birth": "1968-01-15",
        "sex": "F",
        "age": 58,
        "diagnosis_code": "C34.1",
        "stage": "IIIB",
        "ecog": 1,
        "pdl1_tps": 65.0,
        "tmb": 14.0,
        "site_id": "SITE-003",
        "documents": [
            "Patient Name: Jane Doe, DOB: 1968-01-15, MRN: 12345678",
            "Referring physician: Dr. Smith, Phone: 555-0123, email: dr.smith@hospital.org",
        ],
        "records": [
            {
                "patient_id": "ORIG-12345",
                "name": "Jane Doe",
                "date_of_birth": "1968-01-15",
                "diagnosis_code": "C34.1",
                "stage": "IIIB",
            }
        ],
        "clinical_data": {
            "diagnosis_code": "C34.1",
            "stage": "IIIB",
            "sex": "female",
            "ecog": 1,
        },
        "dicom_series": [
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
    }


@pytest.fixture()
def sample_documents() -> list[str]:
    """List of strings containing synthetic PHI."""
    return [
        "Patient Name: Jane Doe, DOB: 1968-01-15, MRN: 12345678",
        "Referring physician: Dr. Smith, Phone: 555-0123, email: dr.smith@hospital.org",
        "No PHI in this document. Clinical notes only.",
    ]


@pytest.fixture()
def sample_dicom_metadata() -> list[dict]:
    """DICOM metadata dictionaries for validation testing."""
    return [
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
        }
    ]


# ---------------------------------------------------------------------------
# Tests — Patient State Data Model
# ---------------------------------------------------------------------------


class TestPatientState:
    """Tests for the central patient state data model."""

    def test_patient_state_creation(self):
        """Verify PatientJourneyState initialises with correct defaults."""
        state = PatientJourneyState()
        assert state.stage == PatientStage.PRESCREENING
        assert state.status == PatientStatus.REFERRED
        assert state.data_lock == DataLockStatus.OPEN
        assert state.mcp_conformance_level == MCPConformanceLevel.CORE
        assert state.treatment_arm is None
        assert state.enrollment_day is None
        assert state.surgical_record is None
        assert state.digital_twin is None
        assert len(state.adverse_events) == 0
        assert len(state.audit_trail) == 0

    def test_patient_stage_enum_values(self):
        """Verify all 10 stages are present in PatientStage enum."""
        stages = list(PatientStage)
        assert len(stages) == 10
        expected = [
            "PRESCREENING",
            "ENROLLMENT",
            "DIGITAL_TWIN_CONSTRUCTION",
            "ROBOT_QUALIFICATION",
            "SURGERY",
            "RECOVERY",
            "IMMUNOTHERAPY",
            "FEDERATION",
            "SURVEILLANCE",
            "CLOSEOUT",
        ]
        assert [s.value for s in stages] == expected

    def test_physical_ai_classification_enum(self):
        """Verify all 7 robot types from ICH E6(R3) section 1.2 are present."""
        classifications = list(PhysicalAIClassification)
        assert len(classifications) == 7
        expected = [
            "SURGICAL_ROBOT",
            "COBOT",
            "HUMANOID",
            "THERAPEUTIC",
            "DIAGNOSTIC",
            "ASSISTIVE",
            "REHABILITATIVE",
        ]
        assert [c.value for c in classifications] == expected

    def test_usl_readiness_level_enum(self):
        """Verify USL readiness levels: FOUNDATIONAL/INTERMEDIATE/ADVANCED/CLINICAL_GRADE."""
        levels = list(USLReadinessLevel)
        assert len(levels) == 4
        assert levels[0] == USLReadinessLevel.FOUNDATIONAL
        assert levels[1] == USLReadinessLevel.INTERMEDIATE
        assert levels[2] == USLReadinessLevel.ADVANCED
        assert levels[3] == USLReadinessLevel.CLINICAL_GRADE

    def test_mcp_conformance_level_enum(self):
        """Verify all 5 MCP conformance levels per 21 CFR 50.33."""
        levels = list(MCPConformanceLevel)
        assert len(levels) == 5
        expected = ["CORE", "CLINICAL_READ", "IMAGING", "FEDERATED_SITE", "ROBOT_PROCEDURE"]
        assert [lv.value for lv in levels] == expected


# ---------------------------------------------------------------------------
# Tests — Pre-Screening Orchestrator
# ---------------------------------------------------------------------------


class TestPreScreeningOrchestrator:
    """Tests for the Stage 1 pre-screening orchestrator."""

    def test_phi_detection_finds_identifiers(self, sample_documents):
        """Verify synthetic PHI is detected in referral documents."""
        orch = PreScreeningOrchestrator({"documents": sample_documents}, "SITE-003")
        result = orch.scan_for_phi(sample_documents)
        assert result["phi_found"] is True
        assert result["documents_with_phi"] >= 2
        assert result["total_detections"] > 0

    def test_deidentification_removes_phi(self, sample_referral_data):
        """Verify de-identified records contain no PHI fields."""
        orch = PreScreeningOrchestrator(sample_referral_data, "SITE-003")
        records = sample_referral_data["records"]
        deidentified = orch.deidentify_records(records)
        assert len(deidentified) == len(records)
        for rec in deidentified:
            if "name" in rec:
                assert "DEIDENTIFIED" in rec["name"]
            if "patient_id" in rec:
                assert rec["patient_id"] == "PAT-2026-0042"

    def test_data_harmonization_maps_codes(self, sample_referral_data):
        """Verify ICD-10 to SNOMED mapping is correct."""
        orch = PreScreeningOrchestrator(sample_referral_data, "SITE-003")
        result = orch.harmonize_clinical_data(sample_referral_data["clinical_data"])
        mappings = result["vocabulary_mappings"]
        assert mappings["icd10_code"] == "C34.1"
        assert mappings["snomed_code"] == 254637007
        assert mappings["mapping_confidence"] >= 0.95

    def test_dicom_validation_passes_valid(self, sample_dicom_metadata):
        """Verify valid DICOM metadata passes validation."""
        orch = PreScreeningOrchestrator({}, "SITE-003")
        result = orch.validate_dicom(sample_dicom_metadata)
        assert result["all_valid"] is True
        assert result["valid_count"] == 1
        assert result["invalid_count"] == 0

    def test_dicom_validation_fails_invalid(self):
        """Verify missing DICOM attributes are flagged."""
        orch = PreScreeningOrchestrator({}, "SITE-003")
        invalid_series = [{"PatientID": "PAT-2026-0042", "Modality": "CT"}]
        result = orch.validate_dicom(invalid_series)
        assert result["all_valid"] is False
        assert result["invalid_count"] == 1
        assert len(result["series_results"][0]["missing_attributes"]) > 0

    def test_patient_record_creation(self, sample_referral_data):
        """Verify PatientJourneyState is populated correctly."""
        orch = PreScreeningOrchestrator(sample_referral_data, "SITE-003")
        harmonized = orch.harmonize_clinical_data(sample_referral_data["clinical_data"])
        state = orch.create_patient_record(harmonized)
        assert state.demographics.patient_id == "PAT-2026-0042"
        assert state.demographics.age == 58
        assert state.demographics.sex == "F"
        assert state.tumor.stage == "IIIB"
        assert state.tumor.volume_cm3 == 42.3
        assert state.biomarkers.pdl1_tps == 65.0
        assert state.biomarkers.tmb == 14.0
        assert state.organ_function.ecog == 1
        assert state.stage == PatientStage.PRESCREENING
        assert state.status == PatientStatus.REFERRED
        assert state.mcp_conformance_level == MCPConformanceLevel.CLINICAL_READ

    def test_access_provisioning(self, sample_referral_data):
        """Verify audit entry is created with Part 11 compliance."""
        orch = PreScreeningOrchestrator(sample_referral_data, "SITE-003")
        result = orch.provision_access("PAT-2026-0042", "SITE-003")
        assert result["cfr_part_11_compliant"] is True
        assert result["access_type"] == "READ_ONLY"
        assert len(result["roles_provisioned"]) == 3
        assert "SIG-ACCESS" in result["electronic_signature"]

    def test_prescreening_run_end_to_end(self, sample_referral_data):
        """Verify full run() method produces valid state."""
        orch = PreScreeningOrchestrator(sample_referral_data, "SITE-003")
        state = orch.run()
        assert state.demographics.patient_id == "PAT-2026-0042"
        assert state.stage == PatientStage.PRESCREENING
        assert state.status == PatientStatus.REFERRED
        assert len(state.regulatory_events) >= 2
        assert len(state.audit_trail) >= 2

    def test_advance_stage_valid_transition(self, sample_referral_data):
        """Verify PRESCREENING to ENROLLMENT is allowed."""
        orch = PreScreeningOrchestrator(sample_referral_data, "SITE-003")
        state = orch.run()
        state.advance_stage(PatientStage.ENROLLMENT)
        assert state.stage == PatientStage.ENROLLMENT

    def test_advance_stage_invalid_transition(self, sample_referral_data):
        """Verify PRESCREENING to SURGERY raises error."""
        orch = PreScreeningOrchestrator(sample_referral_data, "SITE-003")
        state = orch.run()
        with pytest.raises(ValueError, match="Illegal stage transition"):
            state.advance_stage(PatientStage.SURGERY)
