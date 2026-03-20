#!/usr/bin/env python3
"""
Tests for Master Journey Orchestrator — Full 10-Stage Patient Journey
=======================================================================

Validates the MasterJourneyOrchestrator: creation, initial patient setup,
individual stage execution (stages 1-10), full end-to-end journey,
journey report generation, and completeness of the STAGE_REGULATORY_MAP
and STAGE_MODULE_MAP.

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
master_mod = load_module("master_journey", "patient-journey/master_journey.py")

PatientStage = patient_state_mod.PatientStage
PatientStatus = patient_state_mod.PatientStatus
DataLockStatus = patient_state_mod.DataLockStatus
MCPConformanceLevel = patient_state_mod.MCPConformanceLevel
ImagingTimepoint = patient_state_mod.ImagingTimepoint
ResponseCategory = patient_state_mod.ResponseCategory
FederationContribution = patient_state_mod.FederationContribution
MasterJourneyOrchestrator = master_mod.MasterJourneyOrchestrator
STAGE_REGULATORY_MAP = master_mod.STAGE_REGULATORY_MAP
STAGE_MODULE_MAP = master_mod.STAGE_MODULE_MAP


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


@pytest.fixture()
def trial_protocol():
    """Standard trial protocol configuration."""
    return {
        "site_id": "SITE-003",
        "treatment_protocol": {},
        "federation_config": {
            "strategy": "federated_averaging",
            "num_sites": 5,
            "epsilon": 1.0,
            "delta": 1e-5,
            "max_norm": 1.0,
        },
        "eligibility_criteria": {},
    }


@pytest.fixture()
def referral_data():
    """Standard referral data fixture."""
    return _build_referral_data()


class TestMasterJourneyOrchestrator:
    """Tests for the Master Journey Orchestrator."""

    def test_master_orchestrator_creation(self, trial_protocol):
        """MasterJourneyOrchestrator initialises with protocol config."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        assert orch.trial_protocol == trial_protocol
        assert orch._site_id == "SITE-003"
        assert orch._federation_config["strategy"] == "federated_averaging"

    def test_initial_patient_creation(self, trial_protocol, referral_data):
        """create_initial_patient returns REFERRED status."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.create_initial_patient(referral_data)
        assert state.status == PatientStatus.REFERRED
        assert state.demographics.patient_id == "PAT-2026-0042"

    def test_stage_01_prescreening_runs(self, trial_protocol, referral_data):
        """Stage 1 pre-screening produces valid state."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.run_stage_01_prescreening(referral_data)
        assert state.stage == PatientStage.PRESCREENING
        assert state.demographics.patient_id != ""
        assert orch._stage_results.get("stage_01") == "COMPLETED"

    def test_stage_02_enrollment_runs(self, trial_protocol, referral_data):
        """Stage 2 enrollment produces enrolled state."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.run_stage_01_prescreening(referral_data)
        state = orch.run_stage_02_enrollment(state)
        assert state.stage == PatientStage.ENROLLMENT
        assert orch._stage_results.get("stage_02") == "COMPLETED"

    def test_stage_03_digital_twin_runs(self, trial_protocol, referral_data):
        """Stage 3 digital twin construction produces calibrated twin."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.run_stage_01_prescreening(referral_data)
        state = orch.run_stage_02_enrollment(state)
        state = orch.run_stage_03_digital_twin(state)
        assert state.stage == PatientStage.DIGITAL_TWIN_CONSTRUCTION
        assert state.digital_twin is not None
        assert orch._stage_results.get("stage_03") == "COMPLETED"

    def test_stage_04_robot_qualification_runs(self, trial_protocol, referral_data):
        """Stage 4 robot qualification produces deployment-ready robot."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.run_stage_01_prescreening(referral_data)
        state = orch.run_stage_02_enrollment(state)
        state = orch.run_stage_03_digital_twin(state)
        state = orch.run_stage_04_robot_qualification(state)
        assert state.stage == PatientStage.ROBOT_QUALIFICATION
        assert len(state.robot_qualifications) > 0
        assert orch._stage_results.get("stage_04") == "COMPLETED"

    def test_stage_05_surgery_runs(self, trial_protocol, referral_data):
        """Stage 5 surgery produces surgical record."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.run_stage_01_prescreening(referral_data)
        state = orch.run_stage_02_enrollment(state)
        state = orch.run_stage_03_digital_twin(state)
        state = orch.run_stage_04_robot_qualification(state)
        state = orch.run_stage_05_surgery(state)
        assert state.stage == PatientStage.SURGERY
        assert state.surgical_record is not None
        assert orch._stage_results.get("stage_05") == "COMPLETED"

    def test_stage_06_recovery_runs(self, trial_protocol, referral_data):
        """Stage 6 recovery produces recovered state."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.run_stage_01_prescreening(referral_data)
        state = orch.run_stage_02_enrollment(state)
        state = orch.run_stage_03_digital_twin(state)
        state = orch.run_stage_04_robot_qualification(state)
        state = orch.run_stage_05_surgery(state)
        state = orch.run_stage_06_recovery(state)
        assert state.stage == PatientStage.RECOVERY
        assert orch._stage_results.get("stage_06") == "COMPLETED"

    def test_stage_07_immunotherapy_runs(self, trial_protocol, referral_data):
        """Stage 7 immunotherapy produces treatment cycles."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.run_stage_01_prescreening(referral_data)
        state = orch.run_stage_02_enrollment(state)
        state = orch.run_stage_03_digital_twin(state)
        state = orch.run_stage_04_robot_qualification(state)
        state = orch.run_stage_05_surgery(state)
        state = orch.run_stage_06_recovery(state)
        state = orch.run_stage_07_immunotherapy(state)
        assert state.stage == PatientStage.IMMUNOTHERAPY
        assert len(state.treatment_cycles) > 0
        assert orch._stage_results.get("stage_07") == "COMPLETED"

    def test_stage_08_federation_runs(self, trial_protocol, referral_data):
        """Stage 8 federation produces 70 contributions."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.run_stage_01_prescreening(referral_data)
        state = orch.run_stage_02_enrollment(state)
        state = orch.run_stage_03_digital_twin(state)
        state = orch.run_stage_04_robot_qualification(state)
        state = orch.run_stage_05_surgery(state)
        state = orch.run_stage_06_recovery(state)
        state = orch.run_stage_07_immunotherapy(state)
        state = orch.run_stage_08_federation(state)
        assert state.stage == PatientStage.FEDERATION
        assert len(state.federation_contributions) == 70
        assert orch._stage_results.get("stage_08") == "COMPLETED"

    def test_stage_09_surveillance_runs(self, trial_protocol, referral_data):
        """Stage 9 surveillance produces quarterly imaging."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.run_stage_01_prescreening(referral_data)
        state = orch.run_stage_02_enrollment(state)
        state = orch.run_stage_03_digital_twin(state)
        state = orch.run_stage_04_robot_qualification(state)
        state = orch.run_stage_05_surgery(state)
        state = orch.run_stage_06_recovery(state)
        state = orch.run_stage_07_immunotherapy(state)
        state = orch.run_stage_08_federation(state)
        imaging_before = len(state.imaging_timepoints)
        state = orch.run_stage_09_surveillance(state)
        assert state.stage == PatientStage.SURVEILLANCE
        assert state.status == PatientStatus.SURVEILLANCE
        # Should have added surveillance imaging
        assert len(state.imaging_timepoints) > imaging_before
        assert orch._stage_results.get("stage_09") == "COMPLETED"

    def test_stage_10_closeout_runs(self, trial_protocol, referral_data):
        """Stage 10 closeout produces COMPLETED status."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.run_stage_01_prescreening(referral_data)
        state = orch.run_stage_02_enrollment(state)
        state = orch.run_stage_03_digital_twin(state)
        state = orch.run_stage_04_robot_qualification(state)
        state = orch.run_stage_05_surgery(state)
        state = orch.run_stage_06_recovery(state)
        state = orch.run_stage_07_immunotherapy(state)
        state = orch.run_stage_08_federation(state)
        state = orch.run_stage_09_surveillance(state)
        state = orch.run_stage_10_closeout(state)
        assert state.stage == PatientStage.CLOSEOUT
        assert state.status == PatientStatus.COMPLETED
        assert state.data_lock == DataLockStatus.HARD_LOCK
        assert orch._stage_results.get("stage_10") == "COMPLETED"

    def test_full_journey_end_to_end(self, trial_protocol, referral_data):
        """Full journey runs all 10 stages to COMPLETED with HARD_LOCK."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.run_full_journey(referral_data)
        assert state.stage == PatientStage.CLOSEOUT
        assert state.status == PatientStatus.COMPLETED
        assert state.data_lock == DataLockStatus.HARD_LOCK
        assert len(orch._stage_results) == 10
        for stage_key, status in orch._stage_results.items():
            assert status == "COMPLETED", f"{stage_key} not completed"
        # Verify outcome
        assert state.outcome is not None
        assert state.outcome["best_response"] == "CR"
        assert state.outcome["event_free_months"] == 36

    def test_journey_report_metrics(self, trial_protocol, referral_data):
        """Journey report contains all required metrics."""
        orch = MasterJourneyOrchestrator(trial_protocol)
        state = orch.run_full_journey(referral_data)
        report = orch.generate_journey_report(state)
        # Top-level fields
        assert report["final_stage"] == "CLOSEOUT"
        assert report["final_status"] == "COMPLETED"
        assert report["data_lock"] == "HARD_LOCK"
        assert report["stages_completed"] == 10
        # Clinical metrics
        cm = report["clinical_metrics"]
        assert cm["best_response"] == "CR"
        assert cm["event_free_months"] == 36
        assert cm["treatment_cycles"] > 0
        assert cm["device_related_aes"] == 0
        # Regulatory compliance
        rc = report["regulatory_compliance"]
        assert rc["total_regulatory_sections"] >= 84
        assert rc["gcp_compliant"] is True
        assert rc["subpart_j_compliant"] is True
        # Physical AI performance
        pai = report["physical_ai_performance"]
        assert pai["digital_twin_calibrated"] is True
        assert pai["robots_qualified"] > 0
        assert pai["federation_rounds"] == 70
        # Trial contribution
        tc = report["trial_contribution"]
        assert tc["bayesian_p_superior"] == 0.97
        assert tc["federated_hr"] == 0.62

    def test_stage_regulatory_map_complete(self):
        """STAGE_REGULATORY_MAP covers all 10 PatientStage values."""
        all_stages = set(PatientStage)
        mapped_stages = set(STAGE_REGULATORY_MAP.keys())
        assert mapped_stages == all_stages, f"Missing stages: {all_stages - mapped_stages}"
        # Each stage has at least one regulatory section
        for stage, sections in STAGE_REGULATORY_MAP.items():
            assert len(sections) >= 1, f"No sections for {stage}"

    def test_stage_module_map_complete(self):
        """STAGE_MODULE_MAP covers all 10 PatientStage values."""
        all_stages = set(PatientStage)
        mapped_stages = set(STAGE_MODULE_MAP.keys())
        assert mapped_stages == all_stages, f"Missing stages: {all_stages - mapped_stages}"
        # Each stage has at least one module
        for stage, modules in STAGE_MODULE_MAP.items():
            assert len(modules) >= 1, f"No modules for {stage}"
