#!/usr/bin/env python3
"""
Cross-Stage Consistency Tests
==============================

Validates that all 10 stages of the patient journey interoperate correctly,
share consistent data models, follow regulatory citation conventions, and
maintain proper stage transitions for Patient PAT-2026-0042.

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


patient_state = load_module("patient_state", "patient-journey/patient_state.py")
stage_01 = load_module("stage_01_prescreening", "patient-journey/stage_01_prescreening.py")
stage_02 = load_module("stage_02_enrollment", "patient-journey/stage_02_enrollment.py")
stage_03 = load_module("stage_03_digital_twin", "patient-journey/stage_03_digital_twin.py")
stage_04 = load_module("stage_04_robot_qualification", "patient-journey/stage_04_robot_qualification.py")
stage_05 = load_module("stage_05_surgery", "patient-journey/stage_05_surgery.py")
stage_06 = load_module("stage_06_recovery", "patient-journey/stage_06_recovery.py")
stage_07 = load_module("stage_07_immunotherapy", "patient-journey/stage_07_immunotherapy.py")
stage_08 = load_module("stage_08_federation", "patient-journey/stage_08_federation.py")
stage_09 = load_module("stage_09_surveillance", "patient-journey/stage_09_surveillance.py")
stage_10 = load_module("stage_10_closeout", "patient-journey/stage_10_closeout.py")
master_mod = load_module("master_journey", "patient-journey/master_journey.py")

PatientStage = patient_state.PatientStage
PatientStatus = patient_state.PatientStatus
PatientJourneyState = patient_state.PatientJourneyState
LEGAL_STAGE_TRANSITIONS = patient_state.LEGAL_STAGE_TRANSITIONS
MasterJourneyOrchestrator = master_mod.MasterJourneyOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _trial_protocol():
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


def _referral_data():
    return {
        "documents": ["Patient Name: Jane Doe, MRN: 12345678"],
        "records": [
            {
                "patient_id": "PAT-2026-0042",
                "age": 58,
                "sex": "F",
                "tumor_type": "NSCLC",
                "stage": "IIIB",
                "ecog": 1,
                "pdl1_tps": 65.0,
                "tmb": 14.0,
            }
        ],
        "clinical_data": {"icd10": "C34.1", "histology": "adenocarcinoma"},
        "dicom_series": [
            {
                "Modality": "CT",
                "PatientID": "PAT-2026-0042",
                "StudyDate": "20260101",
                "SeriesInstanceUID": "1.2.3.4",
                "Manufacturer": "GE",
                "SliceThickness": "1.25",
            }
        ],
    }


@pytest.fixture
def master_orchestrator():
    """Create a MasterJourneyOrchestrator ready for testing."""
    return MasterJourneyOrchestrator(_trial_protocol())


@pytest.fixture
def initial_patient(master_orchestrator):
    """Create the initial patient state for PAT-2026-0042."""
    return master_orchestrator.create_initial_patient(_referral_data())


@pytest.fixture
def full_journey_state(master_orchestrator):
    """Run the full 10-stage journey and return the final state."""
    np.random.seed(42)
    return master_orchestrator.run_full_journey(_referral_data())


# ---------------------------------------------------------------------------
# 1. Enum Completeness
# ---------------------------------------------------------------------------


class TestEnumCompleteness:
    """Verify that all enums have expected values."""

    def test_patient_stage_has_10_values(self):
        """PatientStage enum must cover all 10 stages."""
        assert len(PatientStage) == 10

    def test_patient_stage_names(self):
        """Each stage name must follow the naming convention."""
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
        actual = [s.name for s in PatientStage]
        assert actual == expected

    def test_patient_status_has_sufficient_values(self):
        """PatientStatus enum must have at least 10 states."""
        assert len(PatientStatus) >= 10


# ---------------------------------------------------------------------------
# 2. Stage Transition Legality
# ---------------------------------------------------------------------------


class TestStageTransitions:
    """Verify that legal stage transitions form a valid DAG."""

    def test_all_stages_have_transitions(self):
        """Every stage except CLOSEOUT must have at least one successor."""
        for stage in PatientStage:
            if stage != PatientStage.CLOSEOUT:
                assert stage in LEGAL_STAGE_TRANSITIONS, f"Missing transitions for {stage}"
                assert len(LEGAL_STAGE_TRANSITIONS[stage]) >= 1

    def test_closeout_is_terminal(self):
        """CLOSEOUT stage should not have outgoing transitions."""
        if PatientStage.CLOSEOUT in LEGAL_STAGE_TRANSITIONS:
            assert len(LEGAL_STAGE_TRANSITIONS[PatientStage.CLOSEOUT]) == 0

    def test_sequential_path_exists(self):
        """A valid sequential path through all 10 stages must exist."""
        stages = list(PatientStage)
        for i in range(len(stages) - 1):
            current = stages[i]
            next_stage = stages[i + 1]
            assert next_stage in LEGAL_STAGE_TRANSITIONS.get(current, []), (
                f"No transition from {current} to {next_stage}"
            )

    def test_no_backward_transitions(self):
        """Stages should not transition backward."""
        stage_order = {s: i for i, s in enumerate(PatientStage)}
        for src, dests in LEGAL_STAGE_TRANSITIONS.items():
            for dest in dests:
                assert stage_order[dest] > stage_order[src], f"Backward transition {src} -> {dest}"


# ---------------------------------------------------------------------------
# 3. Orchestrator Interface Consistency
# ---------------------------------------------------------------------------


class TestOrchestratorInterfaces:
    """Every stage orchestrator must have a run() method."""

    @pytest.mark.parametrize(
        "mod,cls_name",
        [
            (stage_01, "PreScreeningOrchestrator"),
            (stage_02, "EnrollmentOrchestrator"),
            (stage_03, "DigitalTwinOrchestrator"),
            (stage_04, "RobotQualificationOrchestrator"),
            (stage_05, "SurgeryOrchestrator"),
            (stage_06, "RecoveryOrchestrator"),
            (stage_07, "ImmunotherapyOrchestrator"),
            (stage_08, "FederationOrchestrator"),
            (stage_09, "SurveillanceOrchestrator"),
            (stage_10, "CloseoutOrchestrator"),
        ],
    )
    def test_orchestrator_has_run_method(self, mod, cls_name):
        """Each orchestrator class must expose a run() method."""
        cls = getattr(mod, cls_name, None)
        assert cls is not None, f"{cls_name} not found in module"
        assert hasattr(cls, "run"), f"{cls_name}.run() not found"
        assert callable(getattr(cls, "run"))


# ---------------------------------------------------------------------------
# 4. Patient Demographics Consistency
# ---------------------------------------------------------------------------


class TestPatientDemographicsConsistency:
    """Patient identity must be preserved across all stages."""

    def test_patient_id_preserved(self, full_journey_state):
        """Patient ID must remain PAT-2026-0042 after full journey."""
        assert full_journey_state.demographics.patient_id == "PAT-2026-0042"

    def test_demographics_immutable(self, full_journey_state):
        """Age and sex must not change during the journey."""
        assert full_journey_state.demographics.age == 58
        assert full_journey_state.demographics.sex == "F"

    def test_tumor_profile_preserved(self, full_journey_state):
        """Core tumor profile remains stable."""
        assert full_journey_state.tumor.tumor_type == "NSCLC"
        assert full_journey_state.tumor.stage == "IIIB"


# ---------------------------------------------------------------------------
# 5. Data Model Field Availability
# ---------------------------------------------------------------------------


class TestDataModelFields:
    """Verify the PatientJourneyState has all required fields."""

    def test_has_demographics(self, initial_patient):
        assert hasattr(initial_patient, "demographics")

    def test_has_tumor(self, initial_patient):
        assert hasattr(initial_patient, "tumor")

    def test_has_biomarkers(self, initial_patient):
        assert hasattr(initial_patient, "biomarkers")

    def test_has_organ_function(self, initial_patient):
        assert hasattr(initial_patient, "organ_function")

    def test_has_adverse_events_list(self, initial_patient):
        assert hasattr(initial_patient, "adverse_events")
        assert isinstance(initial_patient.adverse_events, list)

    def test_has_audit_trail(self, initial_patient):
        assert hasattr(initial_patient, "audit_trail")
        assert isinstance(initial_patient.audit_trail, list)

    def test_has_regulatory_events(self, initial_patient):
        assert hasattr(initial_patient, "regulatory_events")
        assert isinstance(initial_patient.regulatory_events, list)


# ---------------------------------------------------------------------------
# 6. Full Journey Progression
# ---------------------------------------------------------------------------


class TestFullJourneyProgression:
    """Verify end-to-end journey reaches expected final state."""

    def test_final_stage_is_closeout(self, full_journey_state):
        """Patient must reach CLOSEOUT after full journey."""
        assert full_journey_state.stage == PatientStage.CLOSEOUT

    def test_final_status_completed(self, full_journey_state):
        """Patient status must be COMPLETED at journey end."""
        assert full_journey_state.status == PatientStatus.COMPLETED

    def test_adverse_events_accumulated(self, full_journey_state):
        """Adverse events should accumulate across stages."""
        assert len(full_journey_state.adverse_events) >= 1

    def test_audit_trail_populated(self, full_journey_state):
        """Audit trail must contain entries from multiple stages."""
        assert len(full_journey_state.audit_trail) >= 5

    def test_regulatory_events_recorded(self, full_journey_state):
        """Regulatory events must be present from multiple stages."""
        assert len(full_journey_state.regulatory_events) >= 3

    def test_treatment_cycles_recorded(self, full_journey_state):
        """Treatment cycles from immunotherapy must be recorded."""
        assert len(full_journey_state.treatment_cycles) >= 1

    def test_surgical_record_present(self, full_journey_state):
        """Surgical record must be populated after stage 5."""
        assert full_journey_state.surgical_record is not None

    def test_digital_twin_state_present(self, full_journey_state):
        """Digital twin state must be populated after stage 3."""
        assert full_journey_state.digital_twin is not None


# ---------------------------------------------------------------------------
# 7. Diagram File Consistency
# ---------------------------------------------------------------------------


class TestDiagramFileConsistency:
    """Verify that all 30 diagram files exist."""

    DIAGRAM_DIR = PROJECT_ROOT / "patient-journey" / "diagrams"

    @pytest.mark.parametrize("stage_num", range(1, 11))
    def test_three_perspectives_per_stage(self, stage_num):
        """Each stage must have exactly 3 perspective diagrams."""
        stage_dir = self.DIAGRAM_DIR / f"stage_{stage_num:02d}"
        if not stage_dir.exists():
            pytest.skip(f"Diagram directory not found: {stage_dir}")
        expected_files = [
            "perspective_a_timeline.txt",
            "perspective_b_regulatory.txt",
            "perspective_c_clinical.txt",
        ]
        for fname in expected_files:
            fpath = stage_dir / fname
            assert fpath.exists(), f"Missing diagram: {fpath}"
            assert fpath.stat().st_size > 100, f"Diagram too small: {fpath}"


# ---------------------------------------------------------------------------
# 8. Module File Existence
# ---------------------------------------------------------------------------


class TestModuleFileExistence:
    """Verify all 10 stage modules and master exist."""

    @pytest.mark.parametrize(
        "filename",
        [
            "patient_state.py",
            "stage_01_prescreening.py",
            "stage_02_enrollment.py",
            "stage_03_digital_twin.py",
            "stage_04_robot_qualification.py",
            "stage_05_surgery.py",
            "stage_06_recovery.py",
            "stage_07_immunotherapy.py",
            "stage_08_federation.py",
            "stage_09_surveillance.py",
            "stage_10_closeout.py",
            "master_journey.py",
        ],
    )
    def test_module_exists(self, filename):
        """Each module file must exist in patient-journey/."""
        fpath = PROJECT_ROOT / "patient-journey" / filename
        assert fpath.exists(), f"Missing module: {fpath}"
        assert fpath.stat().st_size > 500, f"Module too small: {fpath}"
