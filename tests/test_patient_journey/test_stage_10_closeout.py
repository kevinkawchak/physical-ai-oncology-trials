#!/usr/bin/env python3
"""
Tests for Stage 10: Trial Closeout (Month 36+)
================================================

Validates the closeout orchestrator: HARD_LOCK with Part 11 signatures,
final de-identification review with re-identification risk < 0.04%,
Clinical Study Report with Physical AI device data, GCP audit with
Subpart J compliance, Physical AI data archival, patient status COMPLETED,
and end-to-end closeout run.

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
stage_03_mod = load_module("stage_03_digital_twin", "patient-journey/stage_03_digital_twin.py")
stage_04_mod = load_module("stage_04_robot_qualification", "patient-journey/stage_04_robot_qualification.py")
stage_05_mod = load_module("stage_05_surgery", "patient-journey/stage_05_surgery.py")
stage_06_mod = load_module("stage_06_recovery", "patient-journey/stage_06_recovery.py")
stage_07_mod = load_module("stage_07_immunotherapy", "patient-journey/stage_07_immunotherapy.py")
stage_08_mod = load_module("stage_08_federation", "patient-journey/stage_08_federation.py")
stage_10_mod = load_module("stage_10_closeout", "patient-journey/stage_10_closeout.py")

PatientStage = patient_state_mod.PatientStage
PatientStatus = patient_state_mod.PatientStatus
DataLockStatus = patient_state_mod.DataLockStatus
MCPConformanceLevel = patient_state_mod.MCPConformanceLevel
ImagingTimepoint = patient_state_mod.ImagingTimepoint
ResponseCategory = patient_state_mod.ResponseCategory
RegulatoryEvent = patient_state_mod.RegulatoryEvent
CloseoutOrchestrator = stage_10_mod.CloseoutOrchestrator


@pytest.fixture(autouse=True)
def seed_rng():
    np.random.seed(42)


def _build_referral_data() -> dict:
    """Standard referral data for PAT-2026-0042."""
    return {
        "documents": ["Patient Name: Jane Doe, MRN: 12345678"],
        "records": [
            {
                "patient_id": "PAT-2026-0042",
                "name": "Jane Doe",
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
            }
        ],
    }


def _advance_through_stage_9(state):
    """Advance state through Stage 9 (surveillance) by simulation."""
    # Simulate federation (Stage 8)
    fed_config = {
        "strategy": "federated_averaging",
        "num_sites": 5,
        "epsilon": 1.0,
        "delta": 1e-5,
        "max_norm": 1.0,
    }
    fed_orch = stage_08_mod.FederationOrchestrator(state, fed_config)
    state = fed_orch.run(state, total_rounds=70)

    # Simulate surveillance (Stage 9)
    base_day = 730
    for quarter in range(1, 5):
        state.imaging_timepoints.append(ImagingTimepoint(
            day=base_day + quarter * 90,
            modality="CT",
            tumor_volume_cm3=0.0,
            response_category=ResponseCategory.CR,
            notes=f"Surveillance Q{quarter}: NED",
        ))
    state.advance_stage(PatientStage.SURVEILLANCE)
    state.status = PatientStatus.SURVEILLANCE
    return state


@pytest.fixture()
def closeout_state():
    """Patient state after Stage 9, ready for closeout."""
    referral = _build_referral_data()
    state = stage_01_mod.PreScreeningOrchestrator(referral, "SITE-003").run()
    state = stage_02_mod.EnrollmentOrchestrator(state, {"criteria": {}}).run(state)
    state = stage_03_mod.DigitalTwinOrchestrator(state).run(state)
    state = stage_04_mod.RobotQualificationOrchestrator(state, {}).run(state)
    state = stage_05_mod.SurgeryOrchestrator(state).run(state)
    state = stage_06_mod.RecoveryOrchestrator(state).run(state)
    state = stage_07_mod.ImmunotherapyOrchestrator(state, {}).run(state)
    state = _advance_through_stage_9(state)
    return state


class TestCloseoutOrchestrator:
    """Tests for Stage 10 trial closeout."""

    def test_data_lock_hard(self, closeout_state):
        """Data lock returns HARD_LOCK status per 21 CFR 312.57."""
        orch = CloseoutOrchestrator(closeout_state)
        result = orch.lock_patient_data()
        assert result["lock_status"] == "HARD_LOCK"
        assert result["part_11_compliant"] is True
        assert result["compliant"] is True
        assert "record_hash" in result
        assert result["hash_algorithm"] == "SHA-256"

    def test_data_lock_read_only(self, closeout_state):
        """HARD_LOCK renders the database read-only."""
        orch = CloseoutOrchestrator(closeout_state)
        result = orch.lock_patient_data()
        assert result["read_only"] is True
        assert result["modifications_blocked"] is True
        # Verify electronic signatures are present
        sigs = result["electronic_signatures"]
        assert "principal_investigator" in sigs
        assert "data_manager" in sigs
        assert "quality_assurance" in sigs

    def test_final_deidentification_clean(self, closeout_state):
        """De-identification review reports all fields CLEAN."""
        orch = CloseoutOrchestrator(closeout_state)
        result = orch.final_deidentification_review()
        assert result["all_clean"] is True
        assert result["phi_residual"] is False
        assert result["review_status"] == "PASSED"
        assert result["compliant"] is True

    def test_reidentification_risk_below_threshold(self, closeout_state):
        """Re-identification risk below 0.04% per 21 CFR 50.33."""
        orch = CloseoutOrchestrator(closeout_state)
        result = orch.final_deidentification_review()
        assert result["below_threshold"] is True
        assert result["overall_risk"] < stage_10_mod.REIDENTIFICATION_RISK_THRESHOLD
        assert result["risk_threshold"] == 0.04

    def test_regulatory_package_physical_ai_data(self, closeout_state):
        """CSR includes Physical AI device performance addenda."""
        orch = CloseoutOrchestrator(closeout_state)
        result = orch.generate_regulatory_package()
        assert result["physical_ai_data_included"] is True
        assert result["compliant"] is True
        addenda = result["physical_ai_addenda"]
        assert "robot_system_identification" in addenda
        assert "usl_score_trajectory" in addenda
        assert "digital_twin_validation" in addenda
        assert "surgical_performance" in addenda
        assert "federation_summary" in addenda
        assert addenda["subpart_j_compliant"] is True

    def test_final_validation_report(self, closeout_state):
        """Regulatory package addresses 84+ regulatory sections."""
        orch = CloseoutOrchestrator(closeout_state)
        result = orch.generate_regulatory_package()
        assert result["regulatory_sections_addressed"] >= 84
        assert "csr_sections" in result
        assert len(result["csr_sections"]) >= 12

    def test_gcp_audit_pass_with_physical_ai(self, closeout_state):
        """Final GCP audit returns GCP COMPLIANT: PASS."""
        orch = CloseoutOrchestrator(closeout_state)
        result = orch.run_final_gcp_audit()
        assert result["overall_verdict"] == "GCP COMPLIANT: PASS"
        assert result["compliant"] is True
        assert result["total_findings"] == 0
        assert result["critical_findings"] == 0
        assert result["domains_passed"] == result["total_domains"]

    def test_subpart_j_compliance(self, closeout_state):
        """GCP audit confirms Subpart J Physical AI compliance."""
        orch = CloseoutOrchestrator(closeout_state)
        result = orch.run_final_gcp_audit()
        assert result["subpart_j_compliant"] is True
        pai_audit = result["physical_ai_audit"]
        assert pai_audit["subpart_j_compliance"] == "PASS"
        assert pai_audit["device_accountability"] == "PASS"
        assert pai_audit["sensor_calibration_records"] == "PASS"
        assert pai_audit["cybersecurity_validation"] == "PASS"

    def test_physical_ai_data_archived(self, closeout_state):
        """Physical AI data archived: 8 categories, 15-year retention."""
        orch = CloseoutOrchestrator(closeout_state)
        result = orch.archive_physical_ai_data()
        assert result["total_categories"] == 8
        assert result["categories_archived"] == 8
        assert result["retention_years"] == 15
        assert result["encryption"] == "AES-256-GCM"
        assert result["integrity_verified"] is True
        assert result["inspection_ready"] is True
        assert result["compliant"] is True
        # Verify each category has integrity hash
        for entry in result["archive_manifest"]:
            assert entry["archived"] is True
            assert "integrity_hash" in entry

    def test_patient_status_completed(self, closeout_state):
        """Patient finalize returns COMPLETED status."""
        orch = CloseoutOrchestrator(closeout_state)
        # Run prerequisite methods
        orch.lock_patient_data()
        orch.final_deidentification_review()
        orch.generate_regulatory_package()
        orch.run_final_gcp_audit()
        orch.archive_physical_ai_data()
        orch.calculate_trial_contribution()
        result = orch.finalize_patient_record()
        assert result["final_status"] == "COMPLETED"
        assert result["all_checks_passed"] is True
        assert result["total_stages_completed"] == 10
        assert result["event_free_at_36_months"] is True

    def test_closeout_run_end_to_end(self, closeout_state):
        """Full closeout run produces COMPLETED status, HARD_LOCK, CLOSEOUT stage."""
        orch = CloseoutOrchestrator(closeout_state)
        state = orch.run(closeout_state)
        assert state.stage == PatientStage.CLOSEOUT
        assert state.status == PatientStatus.COMPLETED
        assert state.data_lock == DataLockStatus.HARD_LOCK
        # Verify outcome recorded
        assert state.outcome is not None
        assert state.outcome["best_response"] == "CR"
        assert state.outcome["event_free_months"] == 36
        assert state.outcome["bayesian_p_superior"] == 0.97
        assert state.outcome["federated_hr"] == 0.62
        # Verify regulatory events from closeout
        closeout_events = [
            e for e in state.regulatory_events
            if e.event_type in (
                "DATA_HARD_LOCK", "DEIDENTIFICATION_REVIEW",
                "CSR_GENERATED", "GCP_AUDIT_FINAL",
                "PHYSICAL_AI_DATA_ARCHIVED", "TRIAL_CONTRIBUTION_CALCULATED",
            )
        ]
        assert len(closeout_events) >= 6
