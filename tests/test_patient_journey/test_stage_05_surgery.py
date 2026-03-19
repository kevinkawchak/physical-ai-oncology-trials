#!/usr/bin/env python3
"""
Tests for Stage 5: Surgery — Active Robot Operations
=====================================================

Validates the surgical orchestrator for Day 14, including ROS2 deployment,
safety monitoring, sensor fusion, shared autonomy, specimen handling, and
investigator oversight review.

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

PatientStage = patient_state_mod.PatientStage
PatientStatus = patient_state_mod.PatientStatus
MCPConformanceLevel = patient_state_mod.MCPConformanceLevel
SurgeryOrchestrator = stage_05_mod.SurgeryOrchestrator


@pytest.fixture(autouse=True)
def seed_rng():
    np.random.seed(42)


@pytest.fixture()
def qualified_state():
    """Patient state after Stage 4 (robots qualified)."""
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
    return state


class TestSurgeryOrchestrator:
    """Tests for Stage 5 surgery."""

    def test_ros2_state_machine_task_order_lifecycle(self, qualified_state):
        """Task-order lifecycle transitions correctly per 21 CFR 50.30."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.initialize_ros2_deployment()
        assert result["current_state"] == "READY"
        assert len(result["lifecycle_transitions"]) == 3

    def test_pre_procedure_safety_matrix(self, qualified_state):
        """All pre-procedure safety checks pass."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.execute_pre_procedure_safety_matrix()
        assert result["all_checks_passed"] is True
        assert result["estop_tested"] is True

    def test_surgical_twin_activation(self, qualified_state):
        """Digital twin overlay activated for surgery."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.activate_surgical_digital_twin()
        assert result["twin_active"] is True
        assert "tumor_boundaries" in result["overlay_elements"]

    def test_realtime_sync_rate(self, qualified_state):
        """Real-time sync at 30 Hz during surgery."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.start_realtime_sync()
        assert result["sync_rate_hz"] == 30
        assert result["active"] is True

    def test_sensor_fusion_pipeline(self, qualified_state):
        """All 4 sensor types active."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.run_sensor_fusion()
        sensors = result["sensors"]
        assert sensors["stereo_endoscope"]["active"] is True
        assert sensors["force_torque"]["rate_hz"] == 1000

    def test_shared_autonomy_level_2_human_oversight(self, qualified_state):
        """Level 2 autonomy: surgeon leads, AI assists."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.manage_shared_autonomy(autonomy_level=2)
        assert result["autonomy_level"] == 2
        assert result["surgeon_has_override"] is True

    def test_forbidden_operations_enforced(self, qualified_state):
        """Forbidden operations list enforced per 21 CFR 50.30."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.manage_shared_autonomy()
        forbidden = result["forbidden_operations_enforced"]
        assert len(forbidden) >= 4
        assert "autonomous_tissue_cutting_without_confirmation" in forbidden

    def test_autonomy_level_range(self, qualified_state):
        """Autonomy levels 0-4 are valid."""
        orch = SurgeryOrchestrator(qualified_state)
        for level in range(5):
            result = orch.manage_shared_autonomy(autonomy_level=level)
            assert result["autonomy_level"] == level

    def test_safety_monitoring_force_within_limits(self, qualified_state):
        """Max force within 15N tip limit."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.monitor_safety_realtime([])
        assert result["max_force_n"] <= 15.0
        assert result["boundary_violations"] == 0

    def test_safety_monitoring_force_warning_logged(self, qualified_state):
        """Force spike at T+45 min logged with warning."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.monitor_safety_realtime([])
        event = result["force_events"][0]
        assert event["force_n"] == 12.1
        assert event["warning_issued"] is True
        assert event["resolution_ms"] == 200

    def test_sim_vs_real_deviation_twin_accuracy(self, qualified_state):
        """Digital twin deviation within 5mm threshold."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.run_sim_vs_real()
        assert result["within_threshold"] is True
        assert result["max_deviation_mm"] <= 5.0

    def test_specimen_chain_of_custody_part_11(self, qualified_state):
        """Specimen chain-of-custody with 21 CFR Part 11 compliance."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.handle_specimens([])
        assert result["part_11_compliant"] is True
        assert result["specimens_collected"] == 3
        assert result["robot_handler"] == "FRANKA-PANDA-SITE003"

    def test_surgical_record_usl_and_autonomy(self, qualified_state):
        """Surgical record has USL 7.9 and autonomy level 2."""
        orch = SurgeryOrchestrator(qualified_state)
        record = orch.record_surgical_outcome()
        assert record.usl_score_at_procedure == 7.9
        assert record.autonomy_level == 2
        assert record.margins_status == "negative"
        assert record.lymph_nodes_sampled == 18

    def test_investigator_oversight_review(self, qualified_state):
        """Investigator review items (a)-(e) assessed."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.perform_investigator_oversight_review()
        assert result["overall_assessment"] == "APPROVED"
        assert len(result["review_items"]) == 5

    def test_mcp_conformance_robot_procedure(self, qualified_state):
        """MCP conformance upgraded to ROBOT_PROCEDURE."""
        orch = SurgeryOrchestrator(qualified_state)
        result = orch.coordinate_agents()
        assert result["mcp_conformance_level"] == "ROBOT_PROCEDURE"

    def test_surgery_run_end_to_end(self, qualified_state):
        """Full surgical procedure produces valid state."""
        orch = SurgeryOrchestrator(qualified_state)
        state = orch.run(qualified_state)
        assert state.stage == PatientStage.SURGERY
        assert state.status == PatientStatus.SURGERY_COMPLETE
        assert state.surgical_record is not None
        assert state.surgical_record.operative_time_min == 168
        assert state.tumor.volume_cm3 == 0.0
        assert state.mcp_conformance_level == MCPConformanceLevel.ROBOT_PROCEDURE
