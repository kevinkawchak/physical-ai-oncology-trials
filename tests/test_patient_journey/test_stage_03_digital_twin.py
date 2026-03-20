#!/usr/bin/env python3
"""
Tests for Stage 3: Digital Twin Construction
=============================================

Validates the digital twin orchestrator for Day 0 to Day 7 of the
patient journey, including tumor twin creation, toxicity prediction,
treatment simulation, dose calculation, TME modelling, ASME V&V 40
validation, real-time synchronisation, and ongoing consent notification.

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

PatientJourneyState = patient_state_mod.PatientJourneyState
PatientStage = patient_state_mod.PatientStage
DigitalTwinState = patient_state_mod.DigitalTwinState
PreScreeningOrchestrator = stage_01_mod.PreScreeningOrchestrator
EnrollmentOrchestrator = stage_02_mod.EnrollmentOrchestrator
DigitalTwinOrchestrator = stage_03_mod.DigitalTwinOrchestrator


@pytest.fixture(autouse=True)
def seed_rng():
    np.random.seed(42)


@pytest.fixture()
def enrolled_state() -> PatientJourneyState:
    """Patient state after Stage 2 (enrolled and randomized)."""
    referral = {
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
    state = PreScreeningOrchestrator(referral, "SITE-003").run()
    trial_config = {"criteria": {}, "arms": ["EXPERIMENTAL", "CONTROL"]}
    state = EnrollmentOrchestrator(state, trial_config).run(state)
    return state


class TestDigitalTwinOrchestrator:
    """Tests for Stage 3 digital twin construction."""

    def test_tumor_twin_creation(self, enrolled_state):
        """Twin created with correct patient parameters."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.create_tumor_twin()
        assert result["twin_config"]["patient_id"] == "PAT-2026-0042"
        assert result["twin_config"]["age"] == 58
        assert result["twin_config"]["baseline_volume_cm3"] == 42.3

    def test_growth_model_calibration(self, enrolled_state):
        """Calibration: volume 42.3 cm3, rate 0.023/day."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.create_tumor_twin()
        cal = result["calibration"]
        assert cal["baseline_volume_cm3"] == 42.3
        assert cal["proliferation_rate_per_day"] == 0.023
        assert cal["calibrated"] is True

    def test_multiple_growth_models_fitted(self, enrolled_state):
        """All 4 model types present per ICH E6(R3) section 1.4.2."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.create_tumor_twin()
        models = result["model_results"]
        assert len(models) == 4
        expected = ["ReactionDiffusionModel", "LogisticGrowthModel", "GompertzGrowthModel", "MechanisticModel"]
        for name in expected:
            assert name in models

    def test_toxicity_prediction_organ_specific(self, enrolled_state):
        """Hepatic, pulmonary, thyroid, dermatologic risks returned."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.model_toxicity()
        preds = result["organ_predictions"]
        assert "hepatic" in preds
        assert "pulmonary" in preds
        assert "thyroid" in preds
        assert "dermatologic" in preds
        assert preds["hepatic"]["risk_grade_gte_2"] == 0.12

    def test_toxicity_ctcae_grading(self, enrolled_state):
        """All grades are valid CTCAE integers."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.model_toxicity()
        for organ_data in result["organ_predictions"].values():
            grade = organ_data["ctcae_grade_predicted"]
            assert isinstance(grade, int)
            assert 0 <= grade <= 5

    def test_treatment_simulation_three_arms(self, enrolled_state):
        """3 arms simulated with PFS estimates per 21 CFR 312.21."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.simulate_treatments()
        arms = result["arm_predictions"]
        assert len(arms) == 3
        assert arms["Arm_A_experimental"]["predicted_pfs_months"] == 14.2
        assert arms["Arm_B_control"]["predicted_pfs_months"] == 12.7
        assert arms["Arm_C_mono"]["predicted_pfs_months"] == 8.9

    def test_treatment_response_for_assigned_arm(self, enrolled_state):
        """PR with negative volume change for assigned arm."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.simulate_treatments()
        arm_a = result["arm_predictions"]["Arm_A_experimental"]
        assert arm_a["predicted_response"] == "PR"
        assert arm_a["volume_change_pct"] < 0

    def test_dose_calculation_bed_eqd2(self, enrolled_state):
        """BED and EQD2 values are physically reasonable."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.calculate_doses()
        assert result["bed_gy"] == 72.0
        assert result["eqd2_gy"] == 60.0
        assert 0 < result["tcp"] < 1
        assert 0 < result["ntcp_lung"] < 1

    def test_tme_subtype_classification(self, enrolled_state):
        """TME subtype is inflamed for PD-L1 65%."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.model_tumor_microenvironment()
        assert result["tme_profile"]["tme_subtype"] == "inflamed"
        assert result["tme_profile"]["pdl1_tps"] == 65.0

    def test_pseudoprogression_window(self, enrolled_state):
        """Pseudoprogression predicted between weeks 4-8."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.model_tumor_microenvironment()
        window = result["immune_dynamics"]["pseudoprogression_window_weeks"]
        assert window[0] == 4
        assert window[1] == 8

    def test_vv40_validation_passes(self, enrolled_state):
        """Validation score meets 21 CFR 312.402 threshold."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.validate_twin()
        assert result["vv40_compliant"] is True
        assert result["overall_validation_score"] >= 0.9
        assert result["verification"]["unit_tests_passed"] == 40

    def test_realtime_sync_configuration(self, enrolled_state):
        """Sync rate and state vector configured per ICH E6(R3) section 1.4.1."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.establish_realtime_sync()
        assert result["sync_rate_intraop_hz"] == 30
        assert result["configured"] is True
        assert "tumor_position" in result["state_vector"]
        assert result["estimator"] == "extended_kalman_filter"

    def test_patient_notification_twin_creation(self, enrolled_state):
        """21 CFR 50.32 ongoing consent documented."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        result = orch.notify_patient_twin_creation()
        assert result["documented"] is True
        assert result["prior_consent_verified"] is True
        assert result["cfr_section"] == "21 CFR 50.32"

    def test_digital_twin_run_end_to_end(self, enrolled_state):
        """Full run produces valid DigitalTwinState."""
        orch = DigitalTwinOrchestrator(enrolled_state)
        state = orch.run(enrolled_state)
        assert state.stage == PatientStage.DIGITAL_TWIN_CONSTRUCTION
        assert state.digital_twin is not None
        assert state.digital_twin.calibrated is True
        assert state.digital_twin.validation_score >= 0.9
        assert len(state.imaging_timepoints) >= 1
        assert len(state.regulatory_events) >= 6
