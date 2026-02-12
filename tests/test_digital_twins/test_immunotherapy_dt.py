"""Tests for digital-twins/examples-twins/04_tumor_microenvironment_immunotherapy_dt.py."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module(
        "immunotherapy_dt",
        "digital-twins/examples-twins/04_tumor_microenvironment_immunotherapy_dt.py",
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestImmunePheno:
    def test_inflamed(self, mod):
        assert mod.ImmunePheno.INFLAMED.value == "inflamed"

    def test_excluded(self, mod):
        assert mod.ImmunePheno.IMMUNE_EXCLUDED.value == "immune_excluded"

    def test_desert(self, mod):
        assert mod.ImmunePheno.IMMUNE_DESERT.value == "immune_desert"

    def test_count(self, mod):
        assert len(mod.ImmunePheno) == 3


class TestCheckpointAgent:
    def test_pembrolizumab(self, mod):
        assert mod.CheckpointAgent.PEMBROLIZUMAB.value == "pembrolizumab"

    def test_nivolumab(self, mod):
        assert mod.CheckpointAgent.NIVOLUMAB.value == "nivolumab"

    def test_atezolizumab(self, mod):
        assert mod.CheckpointAgent.ATEZOLIZUMAB.value == "atezolizumab"

    def test_ipilimumab(self, mod):
        assert mod.CheckpointAgent.IPILIMUMAB.value == "ipilimumab"

    def test_durvalumab(self, mod):
        assert mod.CheckpointAgent.DURVALUMAB.value == "durvalumab"


class TestiRECISTResponse:
    def test_icr(self, mod):
        assert mod.iRECISTResponse.iCR.value == "immune_complete_response"

    def test_ipr(self, mod):
        assert mod.iRECISTResponse.iPR.value == "immune_partial_response"

    def test_isd(self, mod):
        assert mod.iRECISTResponse.iSD.value == "immune_stable_disease"

    def test_iupd(self, mod):
        assert mod.iRECISTResponse.iUPD.value == "immune_unconfirmed_progressive_disease"

    def test_icpd(self, mod):
        assert mod.iRECISTResponse.iCPD.value == "immune_confirmed_progressive_disease"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestPatientTMEProfile:
    def test_creation(self, mod):
        profile = mod.PatientTMEProfile(patient_id="IO-001")
        assert profile.patient_id == "IO-001"

    def test_defaults(self, mod):
        profile = mod.PatientTMEProfile(patient_id="IO-002")
        assert profile.tumor_cells_per_mm2 == pytest.approx(5000.0)
        assert profile.cd8_til_per_mm2 == pytest.approx(200.0)
        assert profile.pdl1_tps == pytest.approx(50.0)
        assert profile.msi_status == "MSS"
        assert profile.immune_phenotype == mod.ImmunePheno.INFLAMED

    def test_custom_values(self, mod):
        profile = mod.PatientTMEProfile(
            patient_id="IO-003",
            tmb_mut_per_mb=20.0,
            pdl1_tps=80.0,
            immune_phenotype=mod.ImmunePheno.IMMUNE_DESERT,
        )
        assert profile.tmb_mut_per_mb == pytest.approx(20.0)
        assert profile.immune_phenotype == mod.ImmunePheno.IMMUNE_DESERT


class TestTMEParameters:
    def test_defaults(self, mod):
        params = mod.TMEParameters()
        assert params.tumor_growth_rate == pytest.approx(0.02)
        assert params.tumor_carrying_capacity == pytest.approx(1000.0)
        assert params.immunogenicity == pytest.approx(0.5)
        assert params.pd1_blockade_efficacy == pytest.approx(0.0)

    def test_custom_values(self, mod):
        params = mod.TMEParameters(
            tumor_growth_rate=0.05,
            pd1_blockade_efficacy=0.8,
        )
        assert params.tumor_growth_rate == pytest.approx(0.05)
        assert params.pd1_blockade_efficacy == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# TMEDynamicsModel tests
# ---------------------------------------------------------------------------


class TestTMEDynamicsModel:
    def _make_model_and_state(self, mod):
        params = mod.TMEParameters(
            tumor_growth_rate=0.02,
            immunogenicity=0.5,
            cd8_killing_rate=0.05,
        )
        model = mod.TMEDynamicsModel(params)
        # [T, E, R, M, D, P, IFNg, TGFb, IL10]
        initial_state = np.array([50.0, 2.0, 0.5, 1.0, 1.0, 0.5, 0.5, 0.4, 0.3])
        return model, initial_state

    def test_init(self, mod):
        params = mod.TMEParameters()
        model = mod.TMEDynamicsModel(params)
        assert model.params.tumor_growth_rate == pytest.approx(0.02)

    def test_simulate_dynamics_shape(self, mod):
        model, state = self._make_model_and_state(mod)
        result = model.simulate(state, duration_days=30, dt_days=1.0)
        assert "time_days" in result
        assert "tumor_trajectory" in result
        assert len(result["tumor_trajectory"]) == len(result["time_days"])

    def test_state_finite(self, mod):
        model, state = self._make_model_and_state(mod)
        result = model.simulate(state, duration_days=30, dt_days=1.0)
        assert np.all(np.isfinite(result["tumor_trajectory"]))
        assert np.all(np.isfinite(result["cd8_trajectory"]))

    def test_state_nonnegative(self, mod):
        model, state = self._make_model_and_state(mod)
        result = model.simulate(state, duration_days=30, dt_days=1.0)
        # Non-negativity enforced
        assert np.all(result["tumor_trajectory"] >= 0)
        assert np.all(result["cd8_trajectory"] >= 0)


# ---------------------------------------------------------------------------
# CheckpointPKModel tests
# ---------------------------------------------------------------------------


class TestCheckpointPKModel:
    def test_concentration_bounded(self, mod):
        pk = mod.CheckpointPKModel(mod.CheckpointAgent.PEMBROLIZUMAB)
        dose_days = [0.0, 21.0]
        eff = pk.get_blockade_efficacy(day=10.0, dose_days=dose_days)
        assert 0.0 <= eff <= 1.0

    def test_efficacy_at_zero(self, mod):
        pk = mod.CheckpointPKModel(mod.CheckpointAgent.NIVOLUMAB)
        eff = pk.get_blockade_efficacy(day=0.0, dose_days=[0.0])
        assert 0.0 <= eff <= 1.0

    def test_efficacy_increases_with_doses(self, mod):
        pk = mod.CheckpointPKModel(mod.CheckpointAgent.PEMBROLIZUMAB)
        eff_one = pk.get_blockade_efficacy(day=10.0, dose_days=[0.0])
        eff_two = pk.get_blockade_efficacy(day=10.0, dose_days=[0.0, 5.0])
        assert eff_two >= eff_one

    def test_efficacy_does_not_exceed_peak(self, mod):
        pk = mod.CheckpointPKModel(mod.CheckpointAgent.PEMBROLIZUMAB)
        dose_days = list(range(0, 200, 21))  # many doses
        eff = pk.get_blockade_efficacy(day=190.0, dose_days=dose_days)
        assert eff <= pk.params.peak_efficacy


# ---------------------------------------------------------------------------
# ImmunotherapyResponsePredictor tests
# ---------------------------------------------------------------------------


class TestImmunotherapyResponsePredictor:
    def test_predict_returns_valid_response(self, mod):
        profile = mod.PatientTMEProfile(
            patient_id="IO-PRED-001",
            cd8_til_per_mm2=300.0,
            pdl1_tps=60.0,
            tmb_mut_per_mb=12.0,
            immune_phenotype=mod.ImmunePheno.INFLAMED,
        )
        predictor = mod.ImmunotherapyResponsePredictor(profile)
        result = predictor.predict_response(
            agent=mod.CheckpointAgent.PEMBROLIZUMAB,
            duration_days=10,
            n_doses=1,
        )
        assert "irecist_response" in result
        assert isinstance(result["irecist_response"], mod.iRECISTResponse)
        assert 0.0 <= result["response_probability"] <= 1.0

    def test_immunogenicity_score_bounded(self, mod):
        profile = mod.PatientTMEProfile(patient_id="IO-PRED-002")
        predictor = mod.ImmunotherapyResponsePredictor(profile)
        assert 0.0 <= predictor.immunogenicity <= 1.0

    def test_desert_lower_immunogenicity(self, mod):
        """Desert phenotype should have lower immunogenicity than inflamed."""
        inflamed = mod.PatientTMEProfile(
            patient_id="IO-HOT",
            cd8_til_per_mm2=400.0,
            pdl1_tps=70.0,
            tmb_mut_per_mb=15.0,
            immune_phenotype=mod.ImmunePheno.INFLAMED,
        )
        desert = mod.PatientTMEProfile(
            patient_id="IO-COLD",
            cd8_til_per_mm2=20.0,
            pdl1_tps=2.0,
            tmb_mut_per_mb=2.0,
            immune_phenotype=mod.ImmunePheno.IMMUNE_DESERT,
        )
        p_hot = mod.ImmunotherapyResponsePredictor(inflamed)
        p_cold = mod.ImmunotherapyResponsePredictor(desert)
        assert p_hot.immunogenicity > p_cold.immunogenicity
