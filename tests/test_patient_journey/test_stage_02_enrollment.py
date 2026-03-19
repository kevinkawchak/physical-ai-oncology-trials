#!/usr/bin/env python3
"""
Tests for Stage 2: Eligibility Screening & Enrollment
=====================================================

Validates the enrollment orchestrator for Day -14 to Day 0 of the
patient journey, including eligibility checking, informed consent with
Physical AI appendix, IRB review, GCP validation, and randomization.

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
# Module loading
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


patient_state_mod = load_module("patient_state", "patient-journey/patient_state.py")
stage_01_mod = load_module("stage_01_prescreening", "patient-journey/stage_01_prescreening.py")
stage_02_mod = load_module("stage_02_enrollment", "patient-journey/stage_02_enrollment.py")

PatientJourneyState = patient_state_mod.PatientJourneyState
PatientStage = patient_state_mod.PatientStage
PatientStatus = patient_state_mod.PatientStatus
TreatmentArm = patient_state_mod.TreatmentArm
ConsentStatus = patient_state_mod.ConsentStatus
MCPConformanceLevel = patient_state_mod.MCPConformanceLevel
PreScreeningOrchestrator = stage_01_mod.PreScreeningOrchestrator
EnrollmentOrchestrator = stage_02_mod.EnrollmentOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def seed_rng():
    """Seed NumPy RNG for deterministic tests."""
    np.random.seed(42)


@pytest.fixture()
def enrolled_patient_state() -> PatientJourneyState:
    """Patient state after Stage 1 completion."""
    referral_data = {
        "documents": ["Patient Name: Jane Doe, MRN: 12345678"],
        "records": [{"patient_id": "ORIG-12345", "name": "Jane Doe", "diagnosis_code": "C34.1", "stage": "IIIB"}],
        "clinical_data": {"diagnosis_code": "C34.1", "stage": "IIIB", "sex": "female", "ecog": 1},
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
            }
        ],
    }
    orch = PreScreeningOrchestrator(referral_data, "SITE-003")
    return orch.run()


@pytest.fixture()
def trial_config() -> dict:
    """Trial protocol configuration."""
    return {
        "protocol_id": "PHYS-AI-NSCLC-001",
        "criteria": {
            "inclusion": {
                "confirmed_nsclc": True,
                "stage_iiib_or_iv": True,
                "ecog_0_or_1": True,
                "pdl1_gte_50": True,
                "adequate_organ_function": True,
                "age_gte_18": True,
            },
            "exclusion": {
                "prior_immunotherapy": False,
                "autoimmune_disease": False,
                "active_brain_metastases": False,
            },
        },
        "arms": ["EXPERIMENTAL", "CONTROL"],
        "stratification": ["stage", "pdl1_level", "ecog", "smoking_status"],
    }


@pytest.fixture()
def mock_eligibility_criteria() -> dict:
    """Mock eligibility criteria for testing."""
    return {
        "inclusion": {
            "confirmed_nsclc": True,
            "stage_iiib_or_iv": True,
            "ecog_0_or_1": True,
            "pdl1_gte_50": True,
            "adequate_organ_function": True,
            "age_gte_18": True,
        },
        "exclusion": {
            "prior_immunotherapy": False,
            "autoimmune_disease": False,
            "active_brain_metastases": False,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnrollmentOrchestrator:
    """Tests for the Stage 2 enrollment orchestrator."""

    def test_eligibility_all_criteria_met(self, enrolled_patient_state, trial_config):
        """PAT-2026-0042 passes all inclusion, fails no exclusion."""
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        result = orch.check_eligibility(trial_config["criteria"])
        assert result["eligible"] is True
        assert result["all_inclusion_met"] is True
        assert result["no_exclusion_triggered"] is True

    def test_eligibility_fails_low_pdl1(self, enrolled_patient_state, trial_config):
        """Patient with PD-L1 < 50% is rejected."""
        enrolled_patient_state.biomarkers.pdl1_tps = 10.0
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        result = orch.check_eligibility(trial_config["criteria"])
        assert result["eligible"] is False
        assert result["inclusion_results"]["pdl1_gte_50"] is False

    def test_eligibility_fails_prior_immunotherapy(self, enrolled_patient_state, trial_config):
        """Exclusion triggered for prior immunotherapy."""
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        criteria = trial_config["criteria"].copy()
        criteria["exclusion"] = {"prior_immunotherapy": True}
        result = orch.check_eligibility(criteria)
        assert result["no_exclusion_triggered"] is False

    def test_duplicate_check_no_conflict(self, enrolled_patient_state, trial_config):
        """No duplicate enrollment found across sites."""
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        result = orch.check_duplicate_enrollment("PAT-2026-0042")
        assert result["duplicate_found"] is False
        assert len(result["sites_queried"]) >= 5

    def test_protocol_compliance_verified(self, enrolled_patient_state, trial_config):
        """RAG agent confirms protocol compliance."""
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        result = orch.verify_protocol_compliance(trial_config)
        assert result["compliant"] is True
        assert all(result["consent_elements_ich_e6r3"].values())

    def test_consent_includes_six_ai_elements(self, enrolled_patient_state, trial_config):
        """Consent doc contains all ICH E6(R3) section 2.8.5 elements (a)-(f)."""
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        compliance = orch.verify_protocol_compliance(trial_config)
        elements = compliance["consent_elements_ich_e6r3"]
        assert len(elements) == 6
        assert all(elements.values())

    def test_consent_record_physical_ai_fields(self, enrolled_patient_state, trial_config):
        """Consent record has physical_ai_appendix, robot_types, autonomy, override."""
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        consent = orch.generate_consent("PAT-2026-0042", "SITE-003")
        assert consent.physical_ai_appendix_included is True
        assert "da Vinci Xi" in consent.robot_types_disclosed
        assert "Franka Panda" in consent.robot_types_disclosed
        assert consent.autonomy_levels_disclosed is True
        assert consent.override_rights_disclosed is True

    def test_consent_documentation_system_summary(self, enrolled_patient_state, trial_config):
        """Physical AI System Summary per 21 CFR 50.27 is present."""
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        consent = orch.generate_consent("PAT-2026-0042", "SITE-003")
        assert consent.consent_status == ConsentStatus.OBTAINED
        assert consent.version == "1.0"
        assert consent.physical_ai_appendix_included is True

    def test_irb_review_physical_ai(self, enrolled_patient_state, trial_config):
        """21 CFR 50.31 IRB review includes USL scores and safety profiles."""
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        result = orch.submit_for_irb_review({})
        assert result["status"] == "APPROVED"
        elements = result["physical_ai_elements_reviewed"]
        assert elements["safety_profiles"] is True
        assert elements["usl_scores"] is True
        assert elements["cybersecurity_assessment"] is True

    def test_gcp_validation_passes(self, enrolled_patient_state, trial_config):
        """Enrollment docs pass GCP checks per ICH E6(R3) sections 2.1-2.4."""
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        result = orch.validate_gcp({})
        assert result["gcp_compliant"] is True
        assert "2.1" in result["sections_verified"]

    def test_randomization_assigns_arm(self, enrolled_patient_state, trial_config):
        """Randomization returns valid TreatmentArm enum per section 2.7."""
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        arm = orch.randomize_patient()
        assert arm == TreatmentArm.EXPERIMENTAL
        assert isinstance(arm, TreatmentArm)

    def test_access_upgrade_creates_audit(self, enrolled_patient_state, trial_config):
        """Audit entry for enrollment logged per 21 CFR 312.57."""
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        result = orch.upgrade_access("PAT-2026-0042", TreatmentArm.EXPERIMENTAL)
        assert result["cfr_part_11_compliant"] is True
        assert result["audit_entry"] == "PATIENT_ENROLLED"
        assert len(result["roles_upgraded"]) == 4

    def test_enrollment_run_end_to_end(self, enrolled_patient_state, trial_config):
        """Full pipeline from eligible patient to randomized."""
        orch = EnrollmentOrchestrator(enrolled_patient_state, trial_config)
        state = orch.run(enrolled_patient_state)
        assert state.stage == PatientStage.ENROLLMENT
        assert state.status == PatientStatus.RANDOMIZED
        assert state.treatment_arm == TreatmentArm.EXPERIMENTAL
        assert state.enrollment_day == 0
        assert state.consent.consent_status == ConsentStatus.OBTAINED
        assert state.consent.physical_ai_appendix_included is True
        assert len(state.regulatory_events) >= 4
