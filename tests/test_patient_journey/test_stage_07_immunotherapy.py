#!/usr/bin/env python3
"""
Tests for Stage 7: Adjuvant Immunotherapy (Day 28 to Month 24)
================================================================

Validates the immunotherapy orchestrator for 35 cycles of pembrolizumab
200 mg IV q3w, including treatment initialisation, cycle execution,
cumulative toxicity tracking, digital twin updates, adaptive agent
recommendations, annual IND reporting, and regulatory monitoring.

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
stage_07_mod = load_module("stage_07_immunotherapy", "patient-journey/stage_07_immunotherapy.py")

PatientStage = patient_state_mod.PatientStage
PatientStatus = patient_state_mod.PatientStatus
MCPConformanceLevel = patient_state_mod.MCPConformanceLevel
RegulatoryEvent = patient_state_mod.RegulatoryEvent
ResponseCategory = patient_state_mod.ResponseCategory
AESeverity = patient_state_mod.AESeverity
ImmunotherapyOrchestrator = stage_07_mod.ImmunotherapyOrchestrator


@pytest.fixture(autouse=True)
def seed_rng():
    np.random.seed(42)


def _advance_through_recovery(state):
    """Simulate Stage 6 (Recovery) to advance state for Stage 7.

    Stage 6 module may not yet exist; we manually advance the state
    from SURGERY through RECOVERY, mirroring expected post-operative
    recovery milestones (wound healing, pathology finalisation, ECOG
    recovery, digital twin recalibration).
    """
    # Advance to RECOVERY stage
    state.advance_stage(PatientStage.RECOVERY)
    state.status = PatientStatus.RECOVERING

    # Post-surgical recovery milestones
    state.add_regulatory_event(RegulatoryEvent(
        event_type="RECOVERY_COMPLETE",
        day=27,
        description="Post-operative recovery: wound healing satisfactory, pathology finalised pT2aN2M0",
        document_id="RECOVERY-001",
        cfr_section="ICH E6(R3) section 2.3",
        status="COMPLETED",
    ))

    state.add_audit_entry(
        action="RECOVERY_COMPLETE",
        actor="test_fixture",
        details="Stage 6 Recovery simulated: wound OK, path final, ECOG 0-1, twin recalibrated",
    )

    return state


@pytest.fixture()
def recovered_state():
    """Patient state after Stage 6 (Recovery), ready for immunotherapy."""
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
    # Build through stages 1-5
    state = stage_01_mod.PreScreeningOrchestrator(referral, "SITE-003").run()
    state = stage_02_mod.EnrollmentOrchestrator(state, {"criteria": {}}).run(state)
    state = stage_03_mod.DigitalTwinOrchestrator(state).run(state)
    state = stage_04_mod.RobotQualificationOrchestrator(state, {}).run(state)
    state = stage_05_mod.SurgeryOrchestrator(state).run(state)

    # Simulate stage 6 recovery
    state = _advance_through_recovery(state)

    return state


class TestImmunotherapyOrchestrator:
    """Tests for Stage 7 adjuvant immunotherapy."""

    # ------------------------------------------------------------------
    # 1. Treatment initialisation
    # ------------------------------------------------------------------

    def test_treatment_initialization(self, recovered_state):
        """Treatment plan initialised with correct drug, dose, and PK/PD model."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        result = orch.initialize_treatment()

        plan = result["treatment_plan"]
        assert plan["drug"] == "pembrolizumab"
        assert plan["dose_mg"] == 200.0
        assert plan["route"] == "IV"
        assert plan["frequency"] == "q3w"
        assert plan["planned_cycles"] == 35
        assert plan["start_day"] == 28

        pk_pd = result["pk_pd_model"]
        assert pk_pd["pk_parameters"]["half_life_days"] == 26.0
        assert pk_pd["pd_parameters"]["emax"] == 0.95
        assert result["investigator_confirmed"] is True

        # Validate schedule length
        assert len(result["schedule"]) == 35

    # ------------------------------------------------------------------
    # 2. TME post-surgical enhancement
    # ------------------------------------------------------------------

    def test_tme_post_surgical_enhancement(self, recovered_state):
        """TME model updated post-surgery for immunotherapy response prediction."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        result = orch.enhance_tme_post_surgical()

        assert result["tme_updated"] is True
        assert "t_cell_infiltration" in result["compartments"]
        assert "pd_l1_expression" in result["compartments"]
        assert result["immune_activation_score"] > 0.0

    # ------------------------------------------------------------------
    # 3. Single cycle execution
    # ------------------------------------------------------------------

    def test_single_cycle_execution(self, recovered_state):
        """Single cycle executes with correct drug, dose, and day."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        labs = {"wbc": 6.2, "anc": 3.8, "platelets": 210, "hemoglobin": 12.8}
        cycle = orch.execute_cycle(1, labs)

        assert cycle.cycle_number == 1
        assert cycle.drug == "pembrolizumab"
        assert cycle.dose_mg == 200.0
        assert cycle.start_day == 28
        assert cycle.dose_modifications == "none"
        assert len(cycle.toxicities) == 0

    # ------------------------------------------------------------------
    # 4. Labs recorded in cycle
    # ------------------------------------------------------------------

    def test_cycle_labs_recorded(self, recovered_state):
        """Pre-treatment labs are recorded in the cycle record."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        labs = {"wbc": 6.2, "anc": 3.8, "platelets": 210, "creatinine": 0.9}
        cycle = orch.execute_cycle(3, labs)

        assert cycle.labs_pre["wbc"] == 6.2
        assert cycle.labs_pre["platelets"] == 210

    # ------------------------------------------------------------------
    # 5. Imaging time-point NED
    # ------------------------------------------------------------------

    def test_imaging_timepoint_ned(self, recovered_state):
        """Imaging at cycle 4 shows NED with response assessment."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        labs = {"wbc": 6.2, "anc": 3.8}
        imaging = {"modality": "CT_chest_abdomen_pelvis", "response": "NED", "tumor_volume_cm3": 0.0}
        cycle = orch.execute_cycle(4, labs, imaging)

        assert cycle.response_assessment == ResponseCategory.CR
        assert len(orch._imaging_results) == 1
        assert orch._imaging_results[0]["recurrence_risk_pct"] == 35.0

    # ------------------------------------------------------------------
    # 6. Hypothyroidism at cycle 6
    # ------------------------------------------------------------------

    def test_hypothyroidism_cycle_6(self, recovered_state):
        """Immune-related hypothyroidism (Grade 1) at cycle 6."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        labs = {"wbc": 6.2, "tsh": 4.2, "free_t4": 0.9}
        cycle = orch.execute_cycle(6, labs)

        assert len(cycle.toxicities) == 1
        ae = cycle.toxicities[0]
        assert ae.event_id == "AE-IMM-001"
        assert ae.description == "Immune-related hypothyroidism"
        assert ae.severity == AESeverity.MILD
        assert ae.ctcae_grade == 1
        assert ae.organ_system == "endocrine"

    # ------------------------------------------------------------------
    # 7. Ongoing consent notification at cycle 6
    # ------------------------------------------------------------------

    def test_ongoing_consent_notification_cycle_6(self, recovered_state):
        """Patient notified of hypothyroidism per 21 CFR 50.32."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        # Execute cycles 1-6
        for c in range(1, 7):
            labs = {"wbc": 6.2, "tsh": 2.8 if c < 6 else 4.2}
            orch.execute_cycle(c, labs)

        # Verify cumulative AE tracking includes notification
        toxicity = orch.track_cumulative_toxicity(orch._cycles_completed)
        assert toxicity["patient_notifications_complete"] is True
        assert toxicity["cfr_section"] == "21 CFR 50.32"
        ae_records = toxicity["adverse_events"]
        assert len(ae_records) == 1
        assert ae_records[0]["patient_notified"] is True

    # ------------------------------------------------------------------
    # 8. Rash at cycle 12
    # ------------------------------------------------------------------

    def test_rash_cycle_12(self, recovered_state):
        """Immune-related rash (Grade 1) at cycle 12."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        labs = {"wbc": 6.2}
        cycle = orch.execute_cycle(12, labs)

        assert len(cycle.toxicities) == 1
        ae = cycle.toxicities[0]
        assert ae.event_id == "AE-IMM-002"
        assert ae.description == "Immune-related maculopapular rash"
        assert ae.severity == AESeverity.MILD
        assert ae.ctcae_grade == 1
        assert ae.organ_system == "dermatologic"

    # ------------------------------------------------------------------
    # 9. Cumulative toxicity tracking
    # ------------------------------------------------------------------

    def test_cumulative_toxicity_tracking(self, recovered_state):
        """Cumulative toxicity correctly aggregates AEs across all cycles."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        # Run cycles 1-12 to capture both AEs
        for c in range(1, 13):
            labs = {"wbc": 6.2}
            orch.execute_cycle(c, labs)

        toxicity = orch.track_cumulative_toxicity(orch._cycles_completed)
        assert toxicity["total_aes"] == 2
        assert toxicity["max_grade"] == 1
        assert toxicity["dose_modifications_required"] is False
        assert toxicity["treatment_discontinued"] is False
        assert toxicity["cumulative_toxicity_acceptable"] is True

    # ------------------------------------------------------------------
    # 10. Recurrence risk decreasing
    # ------------------------------------------------------------------

    def test_recurrence_risk_decreasing(self, recovered_state):
        """Recurrence risk monotonically decreases at imaging time-points."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        imaging_cycles = [4, 8, 12, 17, 23, 29, 35]

        for c in imaging_cycles:
            labs = {"wbc": 6.2}
            imaging = {"modality": "CT", "response": "NED", "tumor_volume_cm3": 0.0}
            orch.execute_cycle(c, labs, imaging)
            orch.update_twin_at_imaging(imaging, c)

        # Check monotonic decrease
        risks = [u["recurrence_risk_pct"] for u in orch._twin_updates]
        assert len(risks) == 7
        for i in range(1, len(risks)):
            assert risks[i] < risks[i - 1], f"Risk not decreasing at index {i}: {risks}"
        assert risks[0] == 35.0
        assert risks[-1] == 3.0

    # ------------------------------------------------------------------
    # 11. Adaptive agent with human oversight
    # ------------------------------------------------------------------

    def test_adaptive_agent_human_oversight(self, recovered_state):
        """Adaptive agent recommendation requires human oversight per § 312.404."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        result = orch.run_adaptive_agent(8, {"cycle": 8, "labs": {"wbc": 6.2}})

        assert result["agent_type"] == "adaptive_treatment"
        assert result["cfr_section"] == "21 CFR 312.404"
        oversight = result["human_oversight"]
        assert oversight["investigator_reviewed"] is True
        assert oversight["investigator_approved"] is True
        assert oversight["override_applied"] is False
        assert result["confidence"] > 0.0

    # ------------------------------------------------------------------
    # 12. Simulation agent rash what-if
    # ------------------------------------------------------------------

    def test_simulation_agent_rash_whatif(self, recovered_state):
        """Simulation agent generates what-if scenario for rash at cycle 12+."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        result = orch.run_adaptive_agent(12, {"cycle": 12, "labs": {"wbc": 6.2}})

        assert len(result["whatif_scenarios"]) >= 1
        scenario = result["whatif_scenarios"][0]
        assert scenario["scenario"] == "rash_dose_hold"
        assert scenario["recommendation"] == "continue_with_topical_steroids"

    # ------------------------------------------------------------------
    # 13. Annual report with Physical AI data
    # ------------------------------------------------------------------

    def test_annual_report_physical_ai_data(self, recovered_state):
        """Annual IND report includes Physical AI performance data per § 312.33."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        # Run some cycles to populate data
        for c in range(1, 18):
            labs = {"wbc": 6.2}
            imaging = None
            if c in [4, 8, 12, 17]:
                imaging = {"modality": "CT", "response": "NED", "tumor_volume_cm3": 0.0}
                orch.update_twin_at_imaging(imaging, c)
            orch.execute_cycle(c, labs, imaging)
            if c in [4, 6, 8, 12, 17]:
                orch.run_adaptive_agent(c, {"cycle": c})

        report = orch.generate_annual_report()
        assert report["report_type"] == "ANNUAL_IND_REPORT"
        assert report["cfr_section"] == "21 CFR 312.33"
        assert report["cycles_completed"] == 17
        assert report["treatment_status"] == "ON_TREATMENT"

        ai_data = report["physical_ai_performance"]
        assert ai_data["digital_twin_updates"] >= 1
        assert ai_data["twin_validation_status"] == "ALL_PASSED"
        assert ai_data["adaptive_agent_recommendations"] >= 1
        assert ai_data["system_uptime_pct"] > 99.0

        ae_summary = report["adverse_events"]
        assert ae_summary["total"] == 2
        assert ae_summary["grade_1"] == 2
        assert ae_summary["grade_3_plus"] == 0

    # ------------------------------------------------------------------
    # 14. Regulatory monitoring
    # ------------------------------------------------------------------

    def test_regulatory_monitoring(self, recovered_state):
        """Regulatory monitoring covers all required sections per § 312.56."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        result = orch.monitor_regulatory_changes()

        assert result["monitoring_active"] is True
        assert result["protocol_amendments_pending"] == 0

        monitored_sections = [s["section"] for s in result["sections_monitored"]]
        assert "21 CFR 312.33" in monitored_sections
        assert "21 CFR 312.56" in monitored_sections
        assert "21 CFR 312.30" in monitored_sections
        assert "21 CFR 312.405" in monitored_sections
        assert "ICH E6(R3) section 3.1" in monitored_sections

        for section in result["sections_monitored"]:
            assert section["status"] in ("COMPLIANT", "NO_AMENDMENTS_NEEDED")

    # ------------------------------------------------------------------
    # 15. Treatment completion — 35 cycles
    # ------------------------------------------------------------------

    def test_treatment_completion_35_cycles(self, recovered_state):
        """All 35 cycles complete without dose modification or discontinuation."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        imaging_cycles = {4, 8, 12, 17, 23, 29, 35}

        for c in range(1, 36):
            labs = {"wbc": 6.2, "anc": 3.8}
            imaging = None
            if c in imaging_cycles:
                imaging = {"modality": "CT", "response": "NED", "tumor_volume_cm3": 0.0}
            orch.execute_cycle(c, labs, imaging)

        assert len(orch._cycles_completed) == 35

        # All cycles administered at full dose
        for cycle in orch._cycles_completed:
            assert cycle.dose_mg == 200.0
            assert cycle.dose_modifications == "none"

        toxicity = orch.track_cumulative_toxicity(orch._cycles_completed)
        assert toxicity["total_aes"] == 2
        assert toxicity["treatment_discontinued"] is False

    # ------------------------------------------------------------------
    # 16. End-to-end run
    # ------------------------------------------------------------------

    def test_immunotherapy_run_end_to_end(self, recovered_state):
        """Full immunotherapy orchestration produces valid state."""
        orch = ImmunotherapyOrchestrator(recovered_state, {})
        state = orch.run(recovered_state)

        # Stage and status
        assert state.stage == PatientStage.IMMUNOTHERAPY
        assert state.status == PatientStatus.ON_IMMUNOTHERAPY

        # Treatment cycles recorded
        assert len(state.treatment_cycles) == 35

        # Adverse events recorded on state (2 from immunotherapy)
        immuno_aes = [
            ae for ae in state.adverse_events
            if ae.event_id.startswith("AE-IMM")
        ]
        assert len(immuno_aes) == 2

        # Imaging time-points recorded
        immuno_imaging = [
            img for img in state.imaging_timepoints
            if img.modality == "CT" and img.response_category == ResponseCategory.CR
        ]
        assert len(immuno_imaging) == 7

        # Regulatory events recorded
        reg_types = [r.event_type for r in state.regulatory_events]
        assert "IMMUNOTHERAPY_COMPLETE" in reg_types
        assert "ANNUAL_IND_REPORT" in reg_types
        assert "REGULATORY_MONITORING" in reg_types

        # Audit trail updated
        audit_actions = [a.action for a in state.audit_trail]
        assert "IMMUNOTHERAPY_COMPLETE" in audit_actions
