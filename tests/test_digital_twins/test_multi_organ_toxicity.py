"""Tests for digital-twins/examples-twins/02_multi_organ_toxicity_twin.py

Covers PBPK model, organ toxicity models, CTCAEGrade, and
MultiOrganToxicityTwin end-to-end.
"""

from __future__ import annotations

import numpy as np


# ── Enum values ──────────────────────────────────────────────────────────


class TestEnums:
    def test_chemo_drug_members(self, toxicity_mod):
        drugs = toxicity_mod.ChemoDrug
        assert drugs.CISPLATIN.value == "cisplatin"
        assert drugs.DOXORUBICIN.value == "doxorubicin"
        assert drugs.PACLITAXEL.value == "paclitaxel"

    def test_organ_system_members(self, toxicity_mod):
        organs = toxicity_mod.OrganSystem
        assert organs.CARDIAC.value == "cardiac"
        assert organs.RENAL.value == "renal"
        assert organs.HEPATIC.value == "hepatic"
        assert organs.NEUROLOGICAL.value == "neurological"
        assert organs.HEMATOLOGIC.value == "hematologic"

    def test_ctcae_grade_ordering(self, toxicity_mod):
        g = toxicity_mod.CTCAEGrade
        assert g.GRADE_0.value < g.GRADE_1.value
        assert g.GRADE_1.value < g.GRADE_2.value
        assert g.GRADE_4.value < g.GRADE_5.value


# ── PBPKParameters dataclass ─────────────────────────────────────────────


class TestPBPKParameters:
    def test_default_values(self, toxicity_mod):
        params = toxicity_mod.PBPKParameters(drug=toxicity_mod.ChemoDrug.CISPLATIN)
        assert params.vd_plasma == 3.0
        assert params.cl_renal == 5.0
        assert params.cl_hepatic == 10.0

    def test_drug_assignment(self, toxicity_mod):
        params = toxicity_mod.PBPKParameters(drug=toxicity_mod.ChemoDrug.DOXORUBICIN)
        assert params.drug == toxicity_mod.ChemoDrug.DOXORUBICIN


# ── OrganState dataclass ─────────────────────────────────────────────────


class TestOrganState:
    def test_default_state(self, toxicity_mod):
        state = toxicity_mod.OrganState(organ=toxicity_mod.OrganSystem.CARDIAC)
        assert state.functional_reserve == 1.0
        assert state.cumulative_damage == 0.0
        assert state.ctcae_grade == toxicity_mod.CTCAEGrade.GRADE_0


# ── PBPKModel ────────────────────────────────────────────────────────────


class TestPBPKModel:
    def test_creation(self, toxicity_mod):
        model = toxicity_mod.PBPKModel(drug=toxicity_mod.ChemoDrug.CISPLATIN)
        assert model.params.drug == toxicity_mod.ChemoDrug.CISPLATIN

    def test_simulate_returns_dict(self, toxicity_mod):
        model = toxicity_mod.PBPKModel(drug=toxicity_mod.ChemoDrug.CISPLATIN)
        result = model.simulate(dose_mg=100.0, duration_hours=24)
        assert isinstance(result, dict)
        assert "time_hours" in result
        assert "auc_heart" in result
        assert "auc_kidney" in result

    def test_simulate_concentrations_finite(self, toxicity_mod):
        model = toxicity_mod.PBPKModel(drug=toxicity_mod.ChemoDrug.CISPLATIN)
        result = model.simulate(dose_mg=100.0, duration_hours=48)
        assert np.all(np.isfinite(result["concentration_plasma"]))

    def test_simulate_plasma_decays(self, toxicity_mod):
        model = toxicity_mod.PBPKModel(drug=toxicity_mod.ChemoDrug.CISPLATIN)
        result = model.simulate(dose_mg=100.0, duration_hours=72)
        plasma = result["concentration_plasma"]
        # After infusion ends, plasma should eventually decrease
        peak_idx = np.argmax(plasma)
        if peak_idx < len(plasma) - 1:
            assert plasma[-1] < plasma[peak_idx]


# ── Organ Toxicity Models ────────────────────────────────────────────────


class TestCardiacToxicityModel:
    def test_creation(self, toxicity_mod):
        model = toxicity_mod.CardiacToxicityModel(baseline_lvef=0.60)
        assert model.baseline_lvef == 0.60

    def test_update_returns_organ_state(self, toxicity_mod):
        model = toxicity_mod.CardiacToxicityModel()
        state = model.update(
            drug=toxicity_mod.ChemoDrug.DOXORUBICIN,
            auc_heart=50.0,
        )
        assert isinstance(state, toxicity_mod.OrganState)
        assert state.organ == toxicity_mod.OrganSystem.CARDIAC


class TestRenalToxicityModel:
    def test_creation(self, toxicity_mod):
        model = toxicity_mod.RenalToxicityModel(baseline_gfr=100.0)
        assert model.baseline_gfr == 100.0

    def test_update_returns_organ_state(self, toxicity_mod):
        model = toxicity_mod.RenalToxicityModel()
        state = model.update(
            drug=toxicity_mod.ChemoDrug.CISPLATIN,
            auc_kidney=30.0,
        )
        assert isinstance(state, toxicity_mod.OrganState)
        assert state.organ == toxicity_mod.OrganSystem.RENAL


class TestHematologicToxicityModel:
    def test_creation(self, toxicity_mod):
        model = toxicity_mod.HematologicToxicityModel(baseline_anc=4.5)
        assert model.baseline_anc == 4.5

    def test_update_returns_organ_state(self, toxicity_mod):
        model = toxicity_mod.HematologicToxicityModel()
        state = model.update(
            drug=toxicity_mod.ChemoDrug.PACLITAXEL,
            auc_marrow=40.0,
        )
        assert isinstance(state, toxicity_mod.OrganState)
        assert state.organ == toxicity_mod.OrganSystem.HEMATOLOGIC


# ── MultiOrganToxicityTwin ───────────────────────────────────────────────


class TestMultiOrganToxicityTwin:
    def test_creation(self, toxicity_mod):
        twin = toxicity_mod.MultiOrganToxicityTwin(patient_id="TOX-001")
        assert twin.patient_id == "TOX-001"

    def test_set_baseline(self, toxicity_mod):
        twin = toxicity_mod.MultiOrganToxicityTwin(patient_id="TOX-002")
        twin.set_baseline(gfr=90.0, lvef=0.55, anc=3.5)
        # No crash means baseline was set correctly

    def test_simulate_cycle_returns_profile(self, toxicity_mod):
        twin = toxicity_mod.MultiOrganToxicityTwin(patient_id="TOX-003")
        twin.set_baseline()
        profile = twin.simulate_cycle(
            drug=toxicity_mod.ChemoDrug.CISPLATIN,
            dose_mg_m2=75,
            cycle=1,
        )
        assert isinstance(profile, toxicity_mod.ToxicityProfile)
        assert profile.cycle == 1
        assert isinstance(profile.dose_recommendation, str)

    def test_get_summary(self, toxicity_mod):
        twin = toxicity_mod.MultiOrganToxicityTwin(patient_id="TOX-004")
        twin.set_baseline()
        summary = twin.get_summary()
        assert "patient_id" in summary
        assert summary["patient_id"] == "TOX-004"
