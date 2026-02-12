"""Tests for digital-twins/examples-twins/02_multi_organ_toxicity_twin.py."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module(
        "multi_organ_toxicity",
        "digital-twins/examples-twins/02_multi_organ_toxicity_twin.py",
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestChemoDrug:
    def test_cisplatin(self, mod):
        assert mod.ChemoDrug.CISPLATIN.value == "cisplatin"

    def test_doxorubicin(self, mod):
        assert mod.ChemoDrug.DOXORUBICIN.value == "doxorubicin"

    def test_paclitaxel(self, mod):
        assert mod.ChemoDrug.PACLITAXEL.value == "paclitaxel"

    def test_all_unique(self, mod):
        vals = [d.value for d in mod.ChemoDrug]
        assert len(vals) == len(set(vals))


class TestOrganSystem:
    def test_cardiac(self, mod):
        assert mod.OrganSystem.CARDIAC.value == "cardiac"

    def test_renal(self, mod):
        assert mod.OrganSystem.RENAL.value == "renal"

    def test_hepatic(self, mod):
        assert mod.OrganSystem.HEPATIC.value == "hepatic"

    def test_neurological(self, mod):
        assert mod.OrganSystem.NEUROLOGICAL.value == "neurological"

    def test_hematologic(self, mod):
        assert mod.OrganSystem.HEMATOLOGIC.value == "hematologic"


class TestCTCAEGrade:
    def test_ordering_0_to_5(self, mod):
        grades = [
            mod.CTCAEGrade.GRADE_0,
            mod.CTCAEGrade.GRADE_1,
            mod.CTCAEGrade.GRADE_2,
            mod.CTCAEGrade.GRADE_3,
            mod.CTCAEGrade.GRADE_4,
            mod.CTCAEGrade.GRADE_5,
        ]
        for i, g in enumerate(grades):
            assert g.value == i

    def test_count(self, mod):
        assert len(mod.CTCAEGrade) == 6


# ---------------------------------------------------------------------------
# PBPKParameters tests
# ---------------------------------------------------------------------------


class TestPBPKParameters:
    def test_defaults_for_cisplatin(self, mod):
        params = mod.DRUG_PBPK_LIBRARY[mod.ChemoDrug.CISPLATIN]
        assert params.drug == mod.ChemoDrug.CISPLATIN
        assert params.cl_renal == pytest.approx(4.5)
        assert params.kp_kidney == pytest.approx(5.0)

    def test_default_vd_plasma(self, mod):
        params = mod.PBPKParameters(drug=mod.ChemoDrug.GEMCITABINE)
        assert params.vd_plasma == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# PBPKModel tests
# ---------------------------------------------------------------------------


class TestPBPKModel:
    def test_init(self, mod):
        pbpk = mod.PBPKModel(mod.ChemoDrug.CISPLATIN)
        assert pbpk.drug == mod.ChemoDrug.CISPLATIN
        assert pbpk.N_COMPARTMENTS == 6

    def test_simulate_returns_expected_keys(self, mod):
        pbpk = mod.PBPKModel(mod.ChemoDrug.CISPLATIN)
        result = pbpk.simulate(dose_mg=150.0, duration_hours=24, dt_hours=0.5)
        assert "time_hours" in result
        assert "concentration_plasma" in result
        assert "cmax_kidney" in result
        assert "auc_heart" in result

    def test_simulate_state_shape(self, mod):
        pbpk = mod.PBPKModel(mod.ChemoDrug.CISPLATIN)
        result = pbpk.simulate(dose_mg=100.0, duration_hours=24, dt_hours=0.5)
        n_time = len(result["time_hours"])
        assert len(result["concentration_plasma"]) == n_time
        assert len(result["concentration_kidney"]) == n_time

    def test_simulate_concentrations_nonnegative(self, mod):
        pbpk = mod.PBPKModel(mod.ChemoDrug.DOXORUBICIN)
        result = pbpk.simulate(dose_mg=100.0, duration_hours=48)
        for name in ["plasma", "heart", "kidney", "liver", "nerve", "marrow"]:
            conc = result[f"concentration_{name}"]
            assert np.all(conc >= 0), f"Negative concentration in {name}"


# ---------------------------------------------------------------------------
# Organ toxicity model tests
# ---------------------------------------------------------------------------


class TestOrganModels:
    def test_cardiac_toxicity_assess(self, mod):
        cardiac = mod.CardiacToxicityModel(baseline_lvef=0.60)
        state = cardiac.update(mod.ChemoDrug.DOXORUBICIN, auc_heart=50.0)
        assert state.organ == mod.OrganSystem.CARDIAC
        assert state.current_biomarker <= 0.60
        assert state.ctcae_grade.value >= 0

    def test_renal_toxicity_assess(self, mod):
        renal = mod.RenalToxicityModel(baseline_gfr=100.0)
        state = renal.update(mod.ChemoDrug.CISPLATIN, auc_kidney=30.0, hydration=True)
        assert state.organ == mod.OrganSystem.RENAL
        assert state.current_biomarker <= 100.0

    def test_grade_bounds(self, mod):
        """All CTCAE grades should stay in 0-5 range."""
        cardiac = mod.CardiacToxicityModel()
        for _ in range(20):
            state = cardiac.update(mod.ChemoDrug.DOXORUBICIN, auc_heart=100.0)
        assert 0 <= state.ctcae_grade.value <= 5

    def test_hepatic_model(self, mod):
        hepatic = mod.HepaticToxicityModel(baseline_bilirubin=0.8)
        state = hepatic.update(mod.ChemoDrug.OXALIPLATIN, auc_liver=20.0)
        assert state.organ == mod.OrganSystem.HEPATIC
        assert state.current_biomarker >= 0.8  # bilirubin rises

    def test_neurological_model(self, mod):
        neuro = mod.NeurologicalToxicityModel()
        state = neuro.update(mod.ChemoDrug.OXALIPLATIN, auc_nerve=15.0)
        assert state.organ == mod.OrganSystem.NEUROLOGICAL
        assert state.cumulative_damage >= 0

    def test_hematologic_model(self, mod):
        hemato = mod.HematologicToxicityModel(baseline_anc=4.5)
        state = hemato.update(mod.ChemoDrug.CISPLATIN, auc_marrow=25.0)
        assert state.organ == mod.OrganSystem.HEMATOLOGIC
        assert state.cycles_exposed == 1


# ---------------------------------------------------------------------------
# MultiOrganToxicityTwin tests
# ---------------------------------------------------------------------------


class TestMultiOrganToxicityTwin:
    def test_init(self, mod):
        twin = mod.MultiOrganToxicityTwin(patient_id="TEST-001")
        assert twin.patient_id == "TEST-001"
        assert len(twin.cycle_history) == 0

    def test_simulate_cycle(self, mod):
        twin = mod.MultiOrganToxicityTwin(patient_id="TEST-002")
        twin.set_baseline(gfr=95.0, lvef=0.62, anc=5.0)
        profile = twin.simulate_cycle(
            drug=mod.ChemoDrug.CISPLATIN,
            dose_mg_m2=75.0,
            bsa=1.85,
            cycle=1,
        )
        assert profile.cycle == 1
        assert profile.max_grade.value >= 0
        assert 0.0 <= profile.dlt_probability <= 1.0
        assert isinstance(profile.dose_recommendation, str)
        assert len(twin.cycle_history) == 1

    def test_simulate_multiple_cycles(self, mod):
        twin = mod.MultiOrganToxicityTwin(patient_id="TEST-003")
        twin.set_baseline(gfr=100.0, lvef=0.60, anc=4.5)
        for c in range(1, 4):
            twin.simulate_cycle(
                drug=mod.ChemoDrug.OXALIPLATIN,
                dose_mg_m2=85.0,
                bsa=1.85,
                cycle=c,
            )
        assert len(twin.cycle_history) == 3

    def test_get_summary_empty(self, mod):
        twin = mod.MultiOrganToxicityTwin(patient_id="TEST-004")
        summary = twin.get_summary()
        assert summary["status"] == "no_cycles_simulated"
