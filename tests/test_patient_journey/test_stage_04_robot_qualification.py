#!/usr/bin/env python3
"""
Tests for Stage 4: Robot Qualification & Surgical Planning
==========================================================

Validates robot qualification per Physical AI Subpart J (21 CFR 312.400-405),
USL scoring per ICH E6(R3) section 1.5, and safety validation per 21 CFR 50.30.

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

PatientJourneyState = patient_state_mod.PatientJourneyState
PatientStage = patient_state_mod.PatientStage
PatientStatus = patient_state_mod.PatientStatus
PhysicalAIClassification = patient_state_mod.PhysicalAIClassification
USLReadinessLevel = patient_state_mod.USLReadinessLevel
RobotQualification = patient_state_mod.RobotQualification
RobotQualificationOrchestrator = stage_04_mod.RobotQualificationOrchestrator


@pytest.fixture(autouse=True)
def seed_rng():
    np.random.seed(42)


@pytest.fixture()
def twin_state() -> PatientJourneyState:
    """Patient state after Stage 3 (digital twin constructed)."""
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
    return state


class TestRobotQualification:
    """Tests for Stage 4 robot qualification."""

    def test_robot_classification_surgical(self, twin_state):
        """Da Vinci Xi classified SURGICAL_ROBOT per 21 CFR 312.401."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        result = orch.classify_robots()
        assert result["da_vinci_xi"]["classification"] == "SURGICAL_ROBOT"
        assert result["da_vinci_xi"]["risk_category"] == "Class III"

    def test_robot_classification_cobot(self, twin_state):
        """Franka Panda classified COBOT per 21 CFR 312.401."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        result = orch.classify_robots()
        assert result["franka_panda"]["classification"] == "COBOT"
        assert result["franka_panda"]["risk_category"] == "Class II"

    def test_framework_detection(self, twin_state):
        """Isaac Sim + MuJoCo detected per ICH E6(R3) section 1.4.1."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        result = orch.detect_frameworks("SITE-003")
        assert result["frameworks_detected"]["isaac_sim"]["available"] is True
        assert result["frameworks_detected"]["mujoco"]["available"] is True

    def test_model_conversion_davinci(self, twin_state):
        """URDF to MJCF + USD conversion successful."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        result = orch.convert_robot_models([])
        assert result["all_valid"] is True
        assert len(result["conversions"]) == 2

    def test_usl_surgical_scoring(self, twin_state):
        """Da Vinci composite 7.9, ADVANCED readiness per ICH E6(R3) section 1.5."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        qual = orch.score_surgical_robot()
        assert qual.usl_score == 7.9
        assert qual.usl_readiness == USLReadinessLevel.ADVANCED
        assert qual.robot_classification == PhysicalAIClassification.SURGICAL_ROBOT

    def test_usl_surgical_above_threshold(self, twin_state):
        """7.9 >= 7.0 surgical minimum."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        qual = orch.score_surgical_robot()
        assert qual.usl_score >= 7.0
        assert qual.deployment_ready is True

    def test_usl_cobot_scoring(self, twin_state):
        """Franka composite 7.2, ADVANCED readiness per ICH E6(R3) section 1.5."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        qual = orch.score_cobot()
        assert qual.usl_score == 7.2
        assert qual.usl_readiness == USLReadinessLevel.ADVANCED

    def test_usl_fail_below_threshold(self):
        """Score below threshold fails qualification."""
        qual = RobotQualification(usl_score=4.5, deployment_ready=False)
        assert qual.usl_score < 5.0
        assert qual.deployment_ready is False

    def test_simulation_benchmarks_run(self, twin_state):
        """100 simulation variations per 21 CFR 312.402."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        result = orch.run_simulation_benchmarks()
        assert result["total_simulations"] == 100
        assert result["pass_rate"] >= 0.95

    def test_cross_framework_divergence(self, twin_state):
        """Within tolerance per ICH E6(R3) section 1.4.1."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        result = orch.validate_cross_framework()
        assert result["within_tolerance"] is True
        assert result["positional_divergence_mm"] <= 0.3
        assert result["force_divergence_nm"] <= 0.02

    def test_safety_pre_procedure_matrix(self, twin_state):
        """21 CFR 50.30 pre-procedure safety matrix populated."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        result = orch.validate_safety()
        matrix = result["pre_procedure_safety_matrix"]
        assert matrix["force_limits"]["validated"] is True
        assert matrix["estop_within_spec"] is True
        assert matrix["iec_80601_compliant"] is True

    def test_safety_forbidden_operations(self, twin_state):
        """21 CFR 50.30 forbidden operations enforced."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        result = orch.validate_safety()
        forbidden = result["forbidden_operations"]
        assert len(forbidden) >= 4
        assert "autonomous_tissue_cutting_without_confirmation" in forbidden

    def test_cybersecurity_validated(self, twin_state):
        """21 CFR 312.403 requirements met."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        result = orch.validate_cybersecurity()
        assert result["compliant"] is True
        assert result["encrypted_communication"] is True
        assert result["mfa_enabled"] is True

    def test_deployment_readiness_lifecycle(self, twin_state):
        """21 CFR 312.405 lifecycle documented."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        result = orch.check_deployment_readiness()
        assert result["deployment_ready"] is True
        assert result["items_passed"] == 47

    def test_robot_qualification_run_end_to_end(self, twin_state):
        """Full run with regulatory events for Subpart J."""
        orch = RobotQualificationOrchestrator(twin_state, {})
        state = orch.run(twin_state)
        assert state.stage == PatientStage.ROBOT_QUALIFICATION
        assert state.status == PatientStatus.SURGERY_SCHEDULED
        assert len(state.robot_qualifications) == 2
        assert state.robot_qualifications[0].usl_score == 7.9
        assert state.robot_qualifications[1].usl_score == 7.2
        # Check Subpart J regulatory events
        subpart_j = [e for e in state.regulatory_events if e.event_type == "SUBPART_J_COMPLIANCE"]
        assert len(subpart_j) == 5
