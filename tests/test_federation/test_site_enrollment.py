"""Tests for federation.site_enrollment module.

Covers EnrollmentSynchronizer, StratifiedRandomizer, ConflictResolver,
patient screening/enrollment/withdrawal, and balance monitoring.

License: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from federation.site_enrollment import (
    ConflictResolver,
    ConflictType,
    EnrollmentStatus,
    EnrollmentSynchronizer,
    PatientEnrollment,
    SiteEnrollmentState,
    StratificationConfig,
    StratifiedRandomizer,
    TreatmentArm,
)


# ---------------------------------------------------------------------------
# StratifiedRandomizer tests
# ---------------------------------------------------------------------------


class TestStratifiedRandomizer:
    def test_assigns_arm(self):
        config = StratificationConfig(factors={"stage": ["II", "III"]}, block_size=4)
        randomizer = StratifiedRandomizer(config)
        patient = PatientEnrollment(patient_id="P1", stratification_factors={"stage": "II"})
        arm = randomizer.randomize(patient)
        assert arm in (TreatmentArm.EXPERIMENTAL, TreatmentArm.CONTROL)
        assert patient.status == EnrollmentStatus.RANDOMIZED

    def test_balanced_assignment(self):
        config = StratificationConfig(factors={"stage": ["II"]}, block_size=4)
        randomizer = StratifiedRandomizer(config)
        arms = []
        for i in range(20):
            patient = PatientEnrollment(patient_id=f"P{i}", stratification_factors={"stage": "II"})
            arm = randomizer.randomize(patient)
            arms.append(arm)
        exp_count = sum(1 for a in arms if a == TreatmentArm.EXPERIMENTAL)
        ctrl_count = sum(1 for a in arms if a == TreatmentArm.CONTROL)
        assert abs(exp_count - ctrl_count) <= 2


# ---------------------------------------------------------------------------
# ConflictResolver tests
# ---------------------------------------------------------------------------


class TestConflictResolver:
    def test_detect_no_duplicate(self):
        resolver = ConflictResolver()
        enrollments = {
            "s1": [PatientEnrollment(patient_id="P1", status=EnrollmentStatus.ENROLLED)],
            "s2": [PatientEnrollment(patient_id="P2", status=EnrollmentStatus.ENROLLED)],
        }
        conflict = resolver.detect_duplicate("P1", enrollments)
        assert conflict is None

    def test_detect_duplicate(self):
        resolver = ConflictResolver()
        enrollments = {
            "s1": [PatientEnrollment(patient_id="P1", status=EnrollmentStatus.ENROLLED)],
            "s2": [PatientEnrollment(patient_id="P1", status=EnrollmentStatus.ENROLLED)],
        }
        conflict = resolver.detect_duplicate("P1", enrollments)
        assert conflict is not None
        assert conflict.conflict_type == ConflictType.DUPLICATE_ENROLLMENT

    def test_resolve_duplicate_first_come(self):
        resolver = ConflictResolver()
        enrollments = {
            "s1": [PatientEnrollment(patient_id="P1", status=EnrollmentStatus.ENROLLED)],
            "s2": [PatientEnrollment(patient_id="P1", status=EnrollmentStatus.ENROLLED)],
        }
        conflict = resolver.detect_duplicate("P1", enrollments)
        winner = resolver.resolve_duplicate(conflict)
        assert winner == "s1"
        assert conflict.resolved

    def test_check_arm_balance_no_imbalance(self):
        resolver = ConflictResolver()
        states = {
            "s1": SiteEnrollmentState(site_id="s1", arm_counts={"experimental": 10, "control": 10}),
        }
        conflicts = resolver.check_arm_balance(states)
        assert len(conflicts) == 0


# ---------------------------------------------------------------------------
# EnrollmentSynchronizer tests
# ---------------------------------------------------------------------------


class TestEnrollmentSynchronizer:
    def test_register_site(self):
        sync = EnrollmentSynchronizer(trial_id="TEST")
        state = sync.register_site("s1", "Hospital A", target=30)
        assert state.site_id == "s1"
        assert state.target_enrollment == 30
        assert "s1" in sync.sites

    def test_screen_patient(self):
        sync = EnrollmentSynchronizer()
        sync.register_site("s1")
        patient = sync.screen_patient("s1", eligibility={"consent": True})
        assert patient.status == EnrollmentStatus.SCREENING
        assert patient.site_id == "s1"

    def test_screen_unregistered_site(self):
        sync = EnrollmentSynchronizer()
        with pytest.raises(ValueError, match="not registered"):
            sync.screen_patient("nonexistent")

    def test_enroll_patient(self):
        sync = EnrollmentSynchronizer()
        sync.register_site("s1")
        patient = sync.screen_patient(
            "s1", eligibility={"consent": True}, stratification={"cancer_type": "breast", "stage": "III"}
        )
        result = sync.enroll_patient("s1", patient.patient_id)
        assert result is not None
        assert result.treatment_arm is not None
        assert result.status == EnrollmentStatus.RANDOMIZED

    def test_enroll_screen_fail(self):
        sync = EnrollmentSynchronizer()
        sync.register_site("s1")
        patient = sync.screen_patient("s1", eligibility={"consent": False})
        result = sync.enroll_patient("s1", patient.patient_id)
        assert result is not None
        assert result.status == EnrollmentStatus.SCREEN_FAIL

    def test_withdraw_patient(self):
        sync = EnrollmentSynchronizer()
        sync.register_site("s1")
        patient = sync.screen_patient(
            "s1", eligibility={"consent": True}, stratification={"cancer_type": "lung", "stage": "IV"}
        )
        sync.enroll_patient("s1", patient.patient_id)
        result = sync.withdraw_patient("s1", patient.patient_id, reason="AE")
        assert result is not None
        assert result.status == EnrollmentStatus.WITHDRAWN
        assert sync.sites["s1"].withdrawal_count == 1

    def test_enrollment_summary(self):
        sync = EnrollmentSynchronizer(trial_id="SUM-TEST")
        sync.register_site("s1", target=10)
        sync.register_site("s2", target=10)

        for site_id in ["s1", "s2"]:
            for _ in range(3):
                p = sync.screen_patient(
                    site_id, eligibility={"ok": True}, stratification={"cancer_type": "breast", "stage": "II"}
                )
                sync.enroll_patient(site_id, p.patient_id)

        summary = sync.get_enrollment_summary()
        assert summary["total_enrolled"] == 6
        assert summary["total_target"] == 20
        assert summary["total_sites"] == 2

    def test_enroll_nonexistent_patient(self):
        sync = EnrollmentSynchronizer()
        sync.register_site("s1")
        result = sync.enroll_patient("s1", "FAKE_ID")
        assert result is None

    def test_check_balance(self):
        sync = EnrollmentSynchronizer()
        sync.register_site("s1")
        for _ in range(8):
            p = sync.screen_patient(
                "s1", eligibility={"ok": True}, stratification={"cancer_type": "lung", "stage": "III"}
            )
            sync.enroll_patient("s1", p.patient_id)
        imbalances = sync.check_balance()
        assert isinstance(imbalances, list)
