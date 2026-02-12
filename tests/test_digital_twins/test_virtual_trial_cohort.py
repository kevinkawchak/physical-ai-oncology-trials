"""Unit tests for digital-twins/examples-twins/05_virtual_trial_cohort_dt.py.

Tests cover TumorSite, TrialEndpoint enums, VirtualPatient, PopulationParameters
dataclasses, population library, VirtualTrialSimulator, and trial results.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "digital-twins/examples-twins/05_virtual_trial_cohort_dt.py"


@pytest.fixture()
def mod():
    return load_module("virtual_trial_cohort", MOD_PATH)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_tumor_site_members(self, mod):
        names = [e.name for e in mod.TumorSite]
        for site in ("NSCLC", "BREAST", "COLORECTAL", "MELANOMA", "PANCREATIC"):
            assert site in names

    def test_trial_endpoint_members(self, mod):
        names = [e.name for e in mod.TrialEndpoint]
        for ep in ("PFS", "OS", "ORR", "DOR"):
            assert ep in names


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestVirtualPatient:
    def test_creation(self, mod):
        vp = mod.VirtualPatient(
            patient_id="VP-001",
            age=65,
            sex="M",
            ecog=1,
            stage="IIIA",
            tumor_volume_cm3=15.0,
            growth_rate=0.05,
            treatment_sensitivity=0.5,
            pdl1_tps=50.0,
            tmb=10.0,
            prior_lines=1,
            comorbidity_index=2,
        )
        assert vp.patient_id == "VP-001"
        assert vp.age == 65

    def test_default_optional_fields(self, mod):
        vp = mod.VirtualPatient(
            patient_id="VP-002",
            age=55,
            sex="F",
            ecog=0,
            stage="II",
            tumor_volume_cm3=8.0,
            growth_rate=0.03,
            treatment_sensitivity=0.7,
            pdl1_tps=10.0,
            tmb=5.0,
            prior_lines=0,
            comorbidity_index=1,
        )
        assert hasattr(vp, "pdl1_tps")
        assert hasattr(vp, "tmb")


class TestPopulationParameters:
    def test_nsclc_in_library(self, mod):
        assert mod.TumorSite.NSCLC in mod.POPULATION_LIBRARY

    def test_all_sites_have_params(self, mod):
        # POPULATION_LIBRARY covers NSCLC, COLORECTAL, and MELANOMA
        for site in (mod.TumorSite.NSCLC, mod.TumorSite.COLORECTAL, mod.TumorSite.MELANOMA):
            assert site in mod.POPULATION_LIBRARY, f"{site.name} missing from library"

    def test_parameters_valid_ranges(self, mod):
        for site, params in mod.POPULATION_LIBRARY.items():
            assert params.age_mean > 0, f"{site.name} invalid age_mean"
            assert params.age_std > 0, f"{site.name} invalid age_std"
            assert 0.0 <= params.male_fraction <= 1.0, f"{site.name} invalid male_fraction"


# ---------------------------------------------------------------------------
# Virtual patient generation tests
# ---------------------------------------------------------------------------


class TestVirtualPatientGeneration:
    def test_generate_cohort(self, mod):
        generator = mod.VirtualCohortGenerator(
            tumor_site=mod.TumorSite.NSCLC,
            seed=42,
        )
        cohort = generator.generate(n_patients=20)
        assert len(cohort) == 20
        for p in cohort:
            assert isinstance(p, mod.VirtualPatient)
            assert p.age > 0
            assert p.tumor_volume_cm3 > 0

    def test_deterministic_with_seed(self, mod):
        gen_a = mod.VirtualCohortGenerator(tumor_site=mod.TumorSite.NSCLC, seed=123)
        cohort_a = gen_a.generate(n_patients=10)
        gen_b = mod.VirtualCohortGenerator(tumor_site=mod.TumorSite.NSCLC, seed=123)
        cohort_b = gen_b.generate(n_patients=10)
        for a, b in zip(cohort_a, cohort_b):
            assert a.age == b.age
            assert a.tumor_volume_cm3 == pytest.approx(b.tumor_volume_cm3)


# ---------------------------------------------------------------------------
# Trial simulator tests
# ---------------------------------------------------------------------------


class TestVirtualTrialSimulator:
    def test_run_trial(self, mod):
        design = mod.TrialDesign(
            trial_id="TEST-VT-001",
            n_patients_per_arm=30,
            arms={"control": 1.0, "experimental": 0.5},
        )
        sim = mod.VirtualTrialSimulator(mod.TumorSite.NSCLC, design)
        results = sim.run_trial(seed=42)
        assert results is not None
        assert hasattr(results, "hazard_ratio")
        assert np.isfinite(results.hazard_ratio)

    def test_hazard_ratio_direction(self, mod):
        """Experimental arm with lower hazard -> HR > 1 (longer survival)."""
        design = mod.TrialDesign(
            trial_id="TEST-HR-DIR",
            n_patients_per_arm=100,
            arms={"control": 1.0, "experimental": 0.5},
        )
        sim = mod.VirtualTrialSimulator(mod.TumorSite.NSCLC, design)
        results = sim.run_trial(seed=42)
        assert results.hazard_ratio > 1.0

    def test_interim_analysis(self, mod):
        design = mod.TrialDesign(
            trial_id="TEST-INTERIM",
            n_patients_per_arm=50,
            arms={"control": 1.0, "experimental": 0.6},
        )
        sim = mod.VirtualTrialSimulator(mod.TumorSite.MELANOMA, design)
        results = sim.run_trial(seed=42)
        assert hasattr(results, "p_value")
