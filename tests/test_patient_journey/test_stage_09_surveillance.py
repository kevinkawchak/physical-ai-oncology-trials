#!/usr/bin/env python3
"""
Tests for Stage 9: Treatment Completion & Surveillance (Month 24 to Month 36)
==============================================================================

Validates the surveillance orchestrator: treatment completion recording (CR,
35/35 cycles), digital twin transition to recurrence detection, surveillance
imaging with declining risk, long-term safety monitoring, follow-up data
collection with PROs (EORTC QLQ-C30 + LC13), and end-to-end run producing
the outcome dict with pfs_days, os_days, response_type CR, max_toxicity_grade 2.

Last updated: March 2026
DISCLAIMER: RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
stage_09_mod = load_module("stage_09_surveillance", "patient-journey/stage_09_surveillance.py")

PatientStage = patient_state_mod.PatientStage
PatientStatus = patient_state_mod.PatientStatus
MCPConformanceLevel = patient_state_mod.MCPConformanceLevel
ResponseCategory = patient_state_mod.ResponseCategory
ImagingTimepoint = patient_state_mod.ImagingTimepoint
TreatmentCycle = patient_state_mod.TreatmentCycle
SurveillanceOrchestrator = stage_09_mod.SurveillanceOrchestrator


def _build_state_through_stage_8():
    """Build patient state through Stages 1-8 for surveillance testing."""
    referral = {
        "documents": ["Patient Name: Jane Doe, MRN: 12345678"],
        "records": [
            {
                "patient_id": "ORIG-12345",
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
    state = stage_01_mod.PreScreeningOrchestrator(referral, "SITE-003").run()
    state = stage_02_mod.EnrollmentOrchestrator(state, {"criteria": {}}).run(state)
    state = stage_03_mod.DigitalTwinOrchestrator(state).run(state)
    state = stage_04_mod.RobotQualificationOrchestrator(state, {}).run(state)
    state = stage_05_mod.SurgeryOrchestrator(state).run(state)
    state = stage_06_mod.RecoveryOrchestrator(state).run(state)
    state = stage_07_mod.ImmunotherapyOrchestrator(state, {}).run(state)
    federation_config = {
        "strategy": "federated_averaging",
        "num_sites": 5,
        "epsilon": 1.0,
        "delta": 1e-5,
        "max_norm": 1.0,
    }
    state = stage_08_mod.FederationOrchestrator(state, federation_config).run(
        state, total_rounds=70
    )
    return state


@pytest.fixture()
def surveillance_state():
    """Patient state after Stage 8 (federation complete), ready for surveillance."""
    return _build_state_through_stage_8()


class TestSurveillanceOrchestrator:
    """Tests for Stage 9 treatment completion and surveillance."""

    def test_treatment_completion_cr(self, surveillance_state):
        """Treatment completion records CR per § 50.32 and § 2.9."""
        orch = SurveillanceOrchestrator(surveillance_state)
        result = orch.record_treatment_completion()
        assert result["response_type"] == "CR"
        assert result["treatment_completed"] is True
        assert result["no_evidence_of_disease"] is True
        assert result["patient_notified"] is True
        assert result["compliant"] is True
        assert "21 CFR 50.32" in result["cfr_sections"]

    def test_all_35_cycles_completed(self, surveillance_state):
        """All 35 planned cycles were administered."""
        orch = SurveillanceOrchestrator(surveillance_state)
        result = orch.record_treatment_completion()
        assert result["total_cycles_planned"] == 35
        assert result["all_cycles_completed"] is True
        assert result["total_cycles_administered"] >= 35

    def test_patient_outcome_censored(self, surveillance_state):
        """Patient outcome is censored (event-free) after surveillance."""
        orch = SurveillanceOrchestrator(surveillance_state)
        state = orch.run(surveillance_state)
        assert state.outcome is not None
        assert state.outcome["censored"] is True
        assert state.outcome["event_free"] is True
        assert state.outcome["recurrence_detected"] is False
        assert state.outcome["response_type"] == "CR"

    def test_ongoing_consent_completion_notification(self, surveillance_state):
        """Patient is notified of treatment completion per § 50.32."""
        orch = SurveillanceOrchestrator(surveillance_state)
        result = orch.record_treatment_completion()
        consent = result["consent_notification"]
        assert consent["type"] == "TREATMENT_COMPLETION"
        assert consent["per_section"] == "21 CFR 50.32"
        assert "35 cycles completed" in consent["message"] or "surveillance" in consent["message"]

    def test_surveillance_twin_transition(self, surveillance_state):
        """Digital twin transitions to recurrence detection per § 1.4.2."""
        orch = SurveillanceOrchestrator(surveillance_state)
        result = orch.transition_to_surveillance_twin()
        assert result["previous_mode"] == "treatment_response"
        assert result["new_mode"] == "recurrence_detection"
        assert result["model_recalibrated"] is True
        assert result["validation_status"] == "PASSED"
        assert result["cfr_section"] == "ICH E6(R3) section 1.4.2"
        assert result["compliant"] is True
        # Recurrence detection config should include ctDNA monitoring
        config = result["recurrence_detection_config"]
        assert config["ctdna_monitoring"] is True
        assert len(config["imaging_biomarkers"]) >= 3

    def test_imaging_schedule(self, surveillance_state):
        """Surveillance imaging follows defined schedule per § 312.87."""
        orch = SurveillanceOrchestrator(surveillance_state)
        imaging_inputs = [
            {"month": 27, "modality": "CT", "findings": "NED"},
            {"month": 30, "modality": "CT", "findings": "NED"},
            {"month": 33, "modality": "CT", "findings": "NED"},
            {"month": 36, "modality": "CT", "findings": "NED"},
        ]
        timepoints = orch.process_surveillance_imaging(imaging_inputs)
        assert len(timepoints) == 4
        for tp in timepoints:
            assert isinstance(tp, ImagingTimepoint)
            assert tp.tumor_volume_cm3 == 0.0
            assert tp.response_category == ResponseCategory.CR
            assert "NED" in tp.notes
            assert "312.87" in tp.notes

    def test_recurrence_risk_declining(self, surveillance_state):
        """Recurrence risk declines: Month 30 = 5%, Month 36 = 3%."""
        orch = SurveillanceOrchestrator(surveillance_state)
        imaging_inputs = [
            {"month": 30, "modality": "CT", "findings": "NED"},
            {"month": 36, "modality": "CT", "findings": "NED"},
        ]
        timepoints = orch.process_surveillance_imaging(imaging_inputs)
        # Month 30 note should contain 5%
        assert "5" in timepoints[0].notes or "5.0" in timepoints[0].notes
        # Month 36 note should contain 3%
        assert "3" in timepoints[1].notes or "3.0" in timepoints[1].notes

    def test_long_term_safety_monitoring(self, surveillance_state):
        """Long-term safety monitors 6 categories per § 312.88."""
        orch = SurveillanceOrchestrator(surveillance_state)
        result = orch.monitor_long_term_safety()
        assert result["categories_monitored"] == 6
        assert result["device_related_late_effects"] == 0
        assert result["physical_ai_related_late_effects"] == 0
        assert result["overall_safety_status"] == "ACCEPTABLE"
        assert result["cfr_section"] == "21 CFR 312.88"
        assert result["compliant"] is True
        # Thyroid should be MANAGED (known hypothyroidism from cycle 6)
        thyroid = [
            c for c in result["category_results"]
            if c["category"] == "thyroid_function"
        ]
        assert len(thyroid) == 1
        assert thyroid[0]["status"] == "MANAGED"
        assert thyroid[0]["max_grade"] == 1

    def test_physical_ai_long_term_outcomes(self, surveillance_state):
        """Physical AI long-term outcomes show zero device-related events."""
        orch = SurveillanceOrchestrator(surveillance_state)
        safety = orch.monitor_long_term_safety()
        assert safety["device_related_late_effects"] == 0
        assert safety["physical_ai_related_late_effects"] == 0
        # Follow-up data with PROs
        visits = [
            {
                "month": 30,
                "clinical_assessment": {"recurrence_status": "NED", "ecog_ps": 0},
                "pro_scores": {"qlq_c30_ghs": 78.0, "qlq_c30_pf": 85.0},
            },
            {
                "month": 36,
                "clinical_assessment": {"recurrence_status": "NED", "ecog_ps": 0},
                "pro_scores": {"qlq_c30_ghs": 80.0, "qlq_c30_pf": 88.0},
            },
        ]
        fu = orch.collect_follow_up_data(visits)
        assert fu["total_visits"] == 2
        assert fu["pro_summary"]["all_instruments_completed"] is True
        assert "EORTC_QLQ_C30" in fu["pro_summary"]["instruments"]
        assert "EORTC_QLQ_LC13" in fu["pro_summary"]["instruments"]
        assert fu["all_consent_reaffirmed"] is True

    def test_surveillance_run_end_to_end(self, surveillance_state):
        """Full surveillance run produces valid state and outcome."""
        orch = SurveillanceOrchestrator(surveillance_state)
        state = orch.run(surveillance_state)

        # Stage and status
        assert state.stage == PatientStage.SURVEILLANCE
        assert state.status == PatientStatus.SURVEILLANCE

        # Outcome dict
        assert state.outcome is not None
        assert state.outcome["pfs_days"] == 730
        assert state.outcome["os_days"] == 1095
        assert state.outcome["response_type"] == "CR"
        assert state.outcome["max_toxicity_grade"] == 2
        assert state.outcome["event_free"] is True
        assert state.outcome["censored"] is True
        assert state.outcome["surveillance_complete"] is True

        # Regulatory events recorded
        completion_events = [
            e for e in state.regulatory_events
            if e.event_type == "TREATMENT_COMPLETION"
        ]
        assert len(completion_events) >= 1

        twin_events = [
            e for e in state.regulatory_events
            if e.event_type == "SURVEILLANCE_TWIN_ACTIVE"
        ]
        assert len(twin_events) >= 1

        safety_events = [
            e for e in state.regulatory_events
            if e.event_type == "LONG_TERM_SAFETY_REPORT"
        ]
        assert len(safety_events) >= 1

        imaging_events = [
            e for e in state.regulatory_events
            if e.event_type == "SURVEILLANCE_IMAGING_COMPLETE"
        ]
        assert len(imaging_events) >= 1

        fu_events = [
            e for e in state.regulatory_events
            if e.event_type == "FOLLOW_UP_DATA_COLLECTED"
        ]
        assert len(fu_events) >= 1

        # Imaging timepoints added to state
        surv_imaging = [
            tp for tp in state.imaging_timepoints
            if "surveillance" in tp.notes.lower()
        ]
        assert len(surv_imaging) == 4
