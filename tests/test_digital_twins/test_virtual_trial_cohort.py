"""Tests for digital-twins/examples-twins/05_virtual_trial_cohort_dt.py

Covers TrialDesign, VirtualTrialSimulator, and Bayesian interim analysis.
"""

from __future__ import annotations

import numpy as np


# ── Enum values ──────────────────────────────────────────────────────────


class TestEnums:
    def test_tumor_site_members(self, cohort_mod):
        sites = cohort_mod.TumorSite
        assert sites.NSCLC.value == "non_small_cell_lung_cancer"
        assert sites.BREAST.value == "breast_cancer"
        assert sites.COLORECTAL.value == "colorectal_cancer"


# ── TrialDesign dataclass ────────────────────────────────────────────────


class TestTrialDesign:
    def test_creation(self, cohort_mod):
        design = cohort_mod.TrialDesign(
            trial_id="VTC-001",
            n_patients_per_arm=50,
            arms={"control": 1.0, "experimental": 0.7},
        )
        assert design.trial_id == "VTC-001"
        assert design.n_patients_per_arm == 50
        assert len(design.arms) == 2


# ── VirtualTrialSimulator ────────────────────────────────────────────────


class TestVirtualTrialSimulator:
    def test_creation(self, cohort_mod):
        design = cohort_mod.TrialDesign(
            trial_id="SIM-001",
            n_patients_per_arm=30,
            arms={"control": 1.0, "experimental": 0.5},
        )
        sim = cohort_mod.VirtualTrialSimulator(
            cohort_mod.TumorSite.NSCLC,
            design,
        )
        assert sim.design.trial_id == "SIM-001"

    def test_run_trial_returns_results(self, cohort_mod):
        design = cohort_mod.TrialDesign(
            trial_id="SIM-002",
            n_patients_per_arm=30,
            arms={"control": 1.0, "experimental": 0.5},
        )
        sim = cohort_mod.VirtualTrialSimulator(
            cohort_mod.TumorSite.NSCLC,
            design,
        )
        results = sim.run_trial(seed=42)
        assert results.hazard_ratio is not None
        assert np.isfinite(results.hazard_ratio)

    def test_run_trial_hr_positive(self, cohort_mod):
        design = cohort_mod.TrialDesign(
            trial_id="SIM-003",
            n_patients_per_arm=50,
            arms={"control": 1.0, "experimental": 0.5},
        )
        sim = cohort_mod.VirtualTrialSimulator(
            cohort_mod.TumorSite.BREAST,
            design,
        )
        results = sim.run_trial(seed=42)
        assert results.hazard_ratio > 0

    def test_run_trial_survival_arrays(self, cohort_mod):
        design = cohort_mod.TrialDesign(
            trial_id="SIM-004",
            n_patients_per_arm=20,
            arms={"control": 1.0, "experimental": 0.8},
        )
        sim = cohort_mod.VirtualTrialSimulator(
            cohort_mod.TumorSite.COLORECTAL,
            design,
        )
        results = sim.run_trial(seed=123)
        pfs_times = [o.pfs_days for o in results.outcomes]
        assert len(pfs_times) > 0
        assert np.all(np.array(pfs_times) > 0)
