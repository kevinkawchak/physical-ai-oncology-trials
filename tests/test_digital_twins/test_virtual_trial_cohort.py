"""Tests for digital-twins/examples-twins/05_virtual_trial_cohort_dt.py."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module(
        "virtual_trial_cohort",
        "digital-twins/examples-twins/05_virtual_trial_cohort_dt.py",
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestTumorSite:
    def test_nsclc(self, mod):
        assert mod.TumorSite.NSCLC.value == "non_small_cell_lung_cancer"

    def test_breast(self, mod):
        assert mod.TumorSite.BREAST.value == "breast_cancer"

    def test_colorectal(self, mod):
        assert mod.TumorSite.COLORECTAL.value == "colorectal_cancer"

    def test_melanoma(self, mod):
        assert mod.TumorSite.MELANOMA.value == "melanoma"

    def test_pancreatic(self, mod):
        assert mod.TumorSite.PANCREATIC.value == "pancreatic_cancer"


class TestTrialEndpoint:
    def test_pfs(self, mod):
        assert mod.TrialEndpoint.PFS.value == "progression_free_survival"

    def test_os(self, mod):
        assert mod.TrialEndpoint.OS.value == "overall_survival"

    def test_orr(self, mod):
        assert mod.TrialEndpoint.ORR.value == "objective_response_rate"

    def test_dor(self, mod):
        assert mod.TrialEndpoint.DOR.value == "duration_of_response"


# ---------------------------------------------------------------------------
# VirtualPatient tests
# ---------------------------------------------------------------------------


class TestVirtualPatient:
    def test_creation(self, mod):
        vp = mod.VirtualPatient(
            patient_id="VP-0001",
            age=65.0,
            sex="M",
            ecog=1,
            stage="IVA",
            tumor_volume_cm3=25.0,
            growth_rate=0.015,
            treatment_sensitivity=0.3,
            pdl1_tps=50.0,
            tmb=10.0,
            prior_lines=1,
            comorbidity_index=2,
        )
        assert vp.patient_id == "VP-0001"
        assert vp.age == pytest.approx(65.0)
        assert vp.ecog == 1


# ---------------------------------------------------------------------------
# VirtualCohortGenerator tests
# ---------------------------------------------------------------------------


class TestVirtualCohortGenerator:
    def test_init_nsclc(self, mod):
        gen = mod.VirtualCohortGenerator(mod.TumorSite.NSCLC, seed=42)
        assert gen.params.tumor_site == mod.TumorSite.NSCLC

    def test_generate_cohort_size(self, mod):
        gen = mod.VirtualCohortGenerator(mod.TumorSite.NSCLC, seed=42)
        cohort = gen.generate(n_patients=50)
        assert len(cohort) == 50

    def test_generate_unique_ids(self, mod):
        gen = mod.VirtualCohortGenerator(mod.TumorSite.NSCLC, seed=42)
        cohort = gen.generate(n_patients=20)
        ids = [p.patient_id for p in cohort]
        assert len(ids) == len(set(ids))

    def test_generate_age_in_range(self, mod):
        gen = mod.VirtualCohortGenerator(mod.TumorSite.NSCLC, seed=42)
        cohort = gen.generate(n_patients=100)
        ages = [p.age for p in cohort]
        assert all(18 <= a <= 90 for a in ages)

    def test_generate_ecog_valid(self, mod):
        gen = mod.VirtualCohortGenerator(mod.TumorSite.NSCLC, seed=42)
        cohort = gen.generate(n_patients=100)
        ecog_vals = {p.ecog for p in cohort}
        assert ecog_vals.issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# OutcomeSimulator tests
# ---------------------------------------------------------------------------


class TestOutcomeSimulator:
    def test_simulate_outcomes(self, mod):
        gen = mod.VirtualCohortGenerator(mod.TumorSite.NSCLC, seed=42)
        cohort = gen.generate(n_patients=10)
        sim = mod.OutcomeSimulator(baseline_median_pfs_days=180)
        rng = np.random.default_rng(42)
        outcome = sim.simulate_patient(cohort[0], arm="control", treatment_hr=1.0, rng=rng)
        assert isinstance(outcome, mod.PatientOutcome)
        assert outcome.pfs_days > 0
        assert outcome.os_days >= outcome.pfs_days

    def test_treatment_hr_effect(self, mod):
        """Lower HR should yield longer PFS on average."""
        gen = mod.VirtualCohortGenerator(mod.TumorSite.NSCLC, seed=42)
        cohort = gen.generate(n_patients=200)
        sim = mod.OutcomeSimulator(baseline_median_pfs_days=180)
        rng = np.random.default_rng(42)
        pfs_control = [sim.simulate_patient(p, "ctrl", 1.0, rng=rng).pfs_days for p in cohort]
        rng2 = np.random.default_rng(42)
        pfs_exp = [sim.simulate_patient(p, "exp", 0.5, rng=rng2).pfs_days for p in cohort]
        assert np.median(pfs_exp) > np.median(pfs_control)


# ---------------------------------------------------------------------------
# TrialDesign tests
# ---------------------------------------------------------------------------


class TestTrialDesign:
    def test_creation(self, mod):
        design = mod.TrialDesign(trial_id="TEST-001")
        assert design.trial_id == "TEST-001"

    def test_defaults(self, mod):
        design = mod.TrialDesign(trial_id="TEST-002")
        assert design.n_patients_per_arm == 150
        assert design.alpha == pytest.approx(0.025)
        assert design.power == pytest.approx(0.80)
        assert "control" in design.arms
        assert "experimental" in design.arms


# ---------------------------------------------------------------------------
# VirtualTrialSimulator tests
# ---------------------------------------------------------------------------


class TestVirtualTrialSimulator:
    def test_init(self, mod):
        design = mod.TrialDesign(trial_id="SIM-001", n_patients_per_arm=20)
        sim = mod.VirtualTrialSimulator(mod.TumorSite.NSCLC, design)
        assert sim.tumor_site == mod.TumorSite.NSCLC

    def test_run_trial(self, mod):
        design = mod.TrialDesign(
            trial_id="SIM-002",
            n_patients_per_arm=30,
            arms={"control": 1.0, "experimental": 0.65},
        )
        sim = mod.VirtualTrialSimulator(mod.TumorSite.NSCLC, design)
        results = sim.run_trial(seed=42)
        assert isinstance(results, mod.TrialResults)
        assert results.trial_id == "SIM-002"
        assert results.hazard_ratio > 0

    def test_hazard_ratio_direction(self, mod):
        """With HR=0.65, experimental should have longer PFS (HR<1)."""
        design = mod.TrialDesign(
            trial_id="SIM-003",
            n_patients_per_arm=100,
            arms={"control": 1.0, "experimental": 0.65},
        )
        sim = mod.VirtualTrialSimulator(mod.TumorSite.NSCLC, design)
        results = sim.run_trial(seed=42)
        # Median PFS should be higher in experimental
        assert results.median_pfs_by_arm["experimental"] > results.median_pfs_by_arm["control"]

    def test_results_have_outcomes(self, mod):
        design = mod.TrialDesign(trial_id="SIM-004", n_patients_per_arm=20)
        sim = mod.VirtualTrialSimulator(mod.TumorSite.NSCLC, design)
        results = sim.run_trial(seed=42)
        assert len(results.outcomes) > 0


# ---------------------------------------------------------------------------
# VirtualControlArmBuilder tests
# ---------------------------------------------------------------------------


class TestVirtualControlArmBuilder:
    def test_init(self, mod):
        outcome = mod.PatientOutcome(
            patient_id="H-001",
            arm="control",
            pfs_days=120.0,
            os_days=300.0,
            best_response="SD",
            progressed=True,
        )
        builder = mod.VirtualControlArmBuilder([outcome])
        assert len(builder.historical) == 1

    def test_build(self, mod):
        outcomes = [
            mod.PatientOutcome(
                patient_id=f"H-{i:03d}",
                arm="control",
                pfs_days=float(np.random.uniform(60, 300)),
                os_days=float(np.random.uniform(200, 600)),
                best_response="SD",
                progressed=True,
            )
            for i in range(20)
        ]
        builder = mod.VirtualControlArmBuilder(outcomes)
        gen = mod.VirtualCohortGenerator(mod.TumorSite.NSCLC, seed=42)
        target = gen.generate(5)
        matched = builder.build_control(target, match_ratio=2)
        assert len(matched) == 10  # 5 patients x 2 matches
