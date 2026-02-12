"""Tests for digital-twins/examples-twins/04_tumor_microenvironment_immunotherapy_dt.py

Covers TME dynamics model, immune phenotypes, checkpoint agents,
and immunotherapy response prediction.
"""

from __future__ import annotations

import numpy as np


# ── Enum values ──────────────────────────────────────────────────────────


class TestEnums:
    def test_immune_phenotype_members(self, tme_mod):
        pheno = tme_mod.ImmunePheno
        assert pheno.INFLAMED.value == "inflamed"
        assert pheno.IMMUNE_EXCLUDED.value == "immune_excluded"
        assert pheno.IMMUNE_DESERT.value == "immune_desert"

    def test_checkpoint_agents(self, tme_mod):
        agents = tme_mod.CheckpointAgent
        assert agents.PEMBROLIZUMAB.value == "pembrolizumab"
        assert agents.NIVOLUMAB.value == "nivolumab"
        assert agents.IPILIMUMAB.value == "ipilimumab"

    def test_irecist_responses(self, tme_mod):
        resp = tme_mod.iRECISTResponse
        assert resp.iCR.value == "immune_complete_response"
        assert resp.iPR.value == "immune_partial_response"
        assert resp.iUPD.value == "immune_unconfirmed_progressive_disease"
        assert resp.iCPD.value == "immune_confirmed_progressive_disease"


# ── PatientTMEProfile dataclass ──────────────────────────────────────────


class TestPatientTMEProfile:
    def test_default_values(self, tme_mod):
        profile = tme_mod.PatientTMEProfile(patient_id="TME-001")
        assert profile.patient_id == "TME-001"
        assert profile.tumor_cells_per_mm2 == 5000.0
        assert profile.cd8_til_per_mm2 == 200.0
        assert profile.pdl1_tps == 50.0
        assert profile.immune_phenotype == tme_mod.ImmunePheno.INFLAMED


# ── TMEParameters dataclass ──────────────────────────────────────────────


class TestTMEParameters:
    def test_default_values(self, tme_mod):
        params = tme_mod.TMEParameters()
        assert params.tumor_growth_rate == 0.02
        assert params.cd8_killing_rate == 0.05
        assert params.pd1_blockade_efficacy == 0.0


# ── TMEDynamicsModel ─────────────────────────────────────────────────────


class TestTMEDynamicsModel:
    def test_creation(self, tme_mod):
        params = tme_mod.TMEParameters()
        model = tme_mod.TMEDynamicsModel(params=params)
        assert model.params is params

    def test_simulate_returns_dict(self, tme_mod):
        params = tme_mod.TMEParameters()
        model = tme_mod.TMEDynamicsModel(params=params)
        initial_state = np.array(
            [
                500.0,
                100.0,
                20.0,
                50.0,
                10.0,
                0.5,
                20.0,
                0.3,
                5.0,
            ]
        )
        result = model.simulate(initial_state, duration_days=30, dt_days=1.0)
        assert isinstance(result, dict)
        assert "time_days" in result

    def test_simulate_values_finite(self, tme_mod):
        params = tme_mod.TMEParameters()
        model = tme_mod.TMEDynamicsModel(params=params)
        initial_state = np.array(
            [
                500.0,
                100.0,
                20.0,
                50.0,
                10.0,
                0.5,
                20.0,
                0.3,
                5.0,
            ]
        )
        result = model.simulate(initial_state, duration_days=30, dt_days=1.0)
        for key, values in result.items():
            if isinstance(values, np.ndarray) and values.dtype.kind == "f":
                assert np.all(np.isfinite(values)), "non-finite values in " + key


# ── CheckpointPKModel ────────────────────────────────────────────────────


class TestCheckpointPKModel:
    def test_creation(self, tme_mod):
        model = tme_mod.CheckpointPKModel(
            agent=tme_mod.CheckpointAgent.PEMBROLIZUMAB,
        )
        assert model.params.agent == tme_mod.CheckpointAgent.PEMBROLIZUMAB

    def test_efficacy_bounded(self, tme_mod):
        model = tme_mod.CheckpointPKModel(
            agent=tme_mod.CheckpointAgent.NIVOLUMAB,
        )
        efficacy = model.get_blockade_efficacy(day=10.0, dose_days=[0.0])
        assert 0.0 <= efficacy <= 1.0


# ── ImmunotherapyResponsePredictor ───────────────────────────────────────


class TestImmunotherapyResponsePredictor:
    def test_creation(self, tme_mod):
        profile = tme_mod.PatientTMEProfile(patient_id="TME-PRED-001")
        predictor = tme_mod.ImmunotherapyResponsePredictor(profile)
        assert predictor.profile.patient_id == "TME-PRED-001"

    def test_predict_response_returns_dict(self, tme_mod):
        profile = tme_mod.PatientTMEProfile(patient_id="TME-PRED-002")
        predictor = tme_mod.ImmunotherapyResponsePredictor(profile)
        result = predictor.predict_response(
            agent=tme_mod.CheckpointAgent.PEMBROLIZUMAB,
            duration_days=60,
            n_doses=3,
        )
        assert isinstance(result, dict)
        assert "irecist_response" in result
        assert "response_probability" in result

    def test_response_probability_bounded(self, tme_mod):
        profile = tme_mod.PatientTMEProfile(patient_id="TME-PRED-003")
        predictor = tme_mod.ImmunotherapyResponsePredictor(profile)
        result = predictor.predict_response(duration_days=60, n_doses=3)
        assert 0.0 <= result["response_probability"] <= 1.0
