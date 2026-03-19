#!/usr/bin/env python3
"""
Tests for Stage 6: Post-Operative Recovery (Day 14 to Day 28)
==============================================================

Validates the recovery orchestrator including DT sync transition,
drain management, pathology integration, adverse event tracking,
Physical AI causality assessment, and regulatory compliance.

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

PatientStage = patient_state_mod.PatientStage
PatientStatus = patient_state_mod.PatientStatus
AESeverity = patient_state_mod.AESeverity
AdverseEvent = patient_state_mod.AdverseEvent
MCPConformanceLevel = patient_state_mod.MCPConformanceLevel
RecoveryOrchestrator = stage_06_mod.RecoveryOrchestrator


@pytest.fixture(autouse=True)
def seed_rng():
    np.random.seed(42)


@pytest.fixture()
def surgery_complete_state():
    """Patient state after Stage 5 (surgery complete)."""
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
    state = stage_01_mod.PreScreeningOrchestrator(referral, "SITE-003").run()
    state = stage_02_mod.EnrollmentOrchestrator(state, {"criteria": {}}).run(state)
    state = stage_03_mod.DigitalTwinOrchestrator(state).run(state)
    state = stage_04_mod.RobotQualificationOrchestrator(state, {}).run(state)
    state = stage_05_mod.SurgeryOrchestrator(state).run(state)
    return state


class TestRecoveryOrchestrator:
    """Tests for Stage 6 post-operative recovery."""

    def test_dt_sync_transition_post_procedure(self, surgery_complete_state):
        """DT sync transitions from 30 Hz to event-driven per 21 CFR 50.30."""
        orch = RecoveryOrchestrator(surgery_complete_state)
        result = orch.transition_dt_sync()
        assert result["previous_sync_rate_hz"] == 30
        assert result["new_sync_mode"] == "event_driven"
        assert result["new_sync_rate_hz"] == 0
        assert result["realtime_data_archived"] is True
        assert result["compliant"] is True
        assert len(result["trigger_events"]) >= 4

    def test_drain_output_recommendations(self, surgery_complete_state):
        """Drain management follows clinical thresholds per ICH E6(R3) section 2.3.2."""
        orch = RecoveryOrchestrator(surgery_complete_state)
        result = orch.monitor_recovery_metrics([])
        recs = result["drain_recommendations"]
        # Day 15 (post-op day 1): 210 mL > 100 mL -> continue
        day15_rec = next(r for r in recs if r["day"] == 15)
        assert day15_rec["recommendation"] == "continue_drain"
        assert day15_rec["drain_output_ml"] == 210
        # Day 17 (post-op day 3): 45 mL < 100 mL -> remove
        day17_rec = next(r for r in recs if r["day"] == 17)
        assert day17_rec["recommendation"] == "remove_drain"
        assert day17_rec["drain_output_ml"] == 45
        # Drain removed on Day 17
        assert result["drain_removed_day"] == 17

    def test_pathology_integration(self, surgery_complete_state):
        """Pathology report integrates pT2aN2M0 with negative margins."""
        orch = RecoveryOrchestrator(surgery_complete_state)
        result = orch.integrate_pathology({})
        assert result["pathologic_stage"] == "pT2aN2M0"
        assert result["margins"] == "negative"
        assert result["margin_distance_mm"] == 8
        assert result["pdl1_tps_percent"] == 72
        assert result["pdl1_confirmed"] is True
        assert result["integrated"] is True

    def test_recurrence_risk_model(self, surgery_complete_state):
        """Recurrence risk is 35% at 18 months for pT2aN2M0."""
        orch = RecoveryOrchestrator(surgery_complete_state)
        result = orch.integrate_pathology({})
        assert result["recurrence_risk_18mo"] == 0.35
        assert result["risk_category"] == "intermediate_high"
        assert result["treatment_implications"]["adjuvant_immunotherapy"] == "recommended"
        assert result["treatment_implications"]["recommended_agent"] == "pembrolizumab"

    def test_adverse_event_creation(self, surgery_complete_state):
        """Adverse event for Day 16 atrial fibrillation created correctly."""
        orch = RecoveryOrchestrator(surgery_complete_state)
        ae_list = orch.track_adverse_events([])
        assert len(ae_list) == 1
        ae = ae_list[0]
        assert ae.event_id == "AE-PAT0042-001"
        assert ae.description == "New-onset atrial fibrillation"
        assert ae.onset_day == 16
        assert ae.organ_system == "cardiac"

    def test_ae_severity_moderate(self, surgery_complete_state):
        """Atrial fibrillation is Grade 2 MODERATE per CTCAE v5.0."""
        orch = RecoveryOrchestrator(surgery_complete_state)
        ae_list = orch.track_adverse_events([])
        ae = ae_list[0]
        assert ae.severity == AESeverity.MODERATE
        assert ae.ctcae_grade == 2
        assert ae.resolved is True
        assert ae.resolution_day == 20

    def test_physical_ai_causality_five_categories(self, surgery_complete_state):
        """All 5 Physical AI AE categories (a)-(e) are assessed."""
        orch = RecoveryOrchestrator(surgery_complete_state)
        ae_list = orch.track_adverse_events([])
        ae = ae_list[0]
        result = orch.assess_physical_ai_causality(ae)
        assert result["categories_assessed"] == 5
        cats = result["category_assessments"]
        assert "(a) mechanical_injury" in cats
        assert "(b) software_malfunction" in cats
        assert "(c) sensor_failure" in cats
        assert "(d) communication_failure" in cats
        assert "(e) cybersecurity_breach" in cats
        # All categories must be assessed
        for cat in cats.values():
            assert cat["assessed"] is True

    def test_ae_unrelated_to_device(self, surgery_complete_state):
        """AF is unrelated to Physical AI per causality assessment."""
        orch = RecoveryOrchestrator(surgery_complete_state)
        ae_list = orch.track_adverse_events([])
        ae = ae_list[0]
        result = orch.assess_physical_ai_causality(ae)
        assert result["overall_conclusion"] == "unrelated"
        assert result["related_to_physical_ai"] is False
        assert result["requires_fda_device_report"] is False
        # All 5 categories unrelated
        for cat in result["category_assessments"].values():
            assert cat["related"] is False

    def test_ae_irb_reporting_timeline(self, surgery_complete_state):
        """AE reported to IRB but not FDA (Grade 2, not device-related)."""
        orch = RecoveryOrchestrator(surgery_complete_state)
        ae_list = orch.track_adverse_events([])
        ae = ae_list[0]
        assert ae.reported_to_irb is True
        assert ae.reported_to_fda is False
        assert ae.related_to_device is False

    def test_ae_compliance_check(self, surgery_complete_state):
        """AE compliance validation passes per 21 CFR 312.64."""
        orch = RecoveryOrchestrator(surgery_complete_state)
        ae_list = orch.track_adverse_events([])
        result = orch.validate_ae_compliance(ae_list)
        assert result["total_events"] == 1
        assert result["all_compliant"] is True
        assert result["cfr_section"] == "21 CFR 312.64"
        ev = result["event_compliance"][0]
        assert ev["irb_compliant"] is True
        assert ev["fda_compliant"] is True
        assert ev["overall_compliant"] is True
        # Grade 2 non-device AE: FDA not required
        assert ev["fda_reporting_required"] is False

    def test_recovery_run_end_to_end(self, surgery_complete_state):
        """Full recovery orchestration produces valid state."""
        orch = RecoveryOrchestrator(surgery_complete_state)
        state = orch.run(surgery_complete_state)
        assert state.stage == PatientStage.RECOVERY
        assert state.status == PatientStatus.RECOVERING
        assert len(state.adverse_events) >= 1
        assert state.adverse_events[-1].event_id == "AE-PAT0042-001"
        assert state.adverse_events[-1].ctcae_grade == 2
        assert state.biomarkers.pdl1_tps == 72.0
        # Surgical record pathology updated
        assert state.surgical_record is not None
        assert state.surgical_record.pathology_stage == "pT2aN2M0"
        # Regulatory events recorded
        reg_types = [e.event_type for e in state.regulatory_events]
        assert "DT_SYNC_TRANSITION" in reg_types
        assert "PATHOLOGY_INTEGRATION" in reg_types
        assert "AE_REPORTING" in reg_types

    def test_discharge_planning_after_clear_cxr(self, surgery_complete_state):
        """Discharge planning initiated after Day 5 CXR clear."""
        orch = RecoveryOrchestrator(surgery_complete_state)
        result = orch.monitor_recovery_metrics([])
        assert result["cxr_clear"] is True
        assert result["discharge_planning_initiated"] is True
        assert result["compliant"] is True
