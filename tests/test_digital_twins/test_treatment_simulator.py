"""Tests for digital-twins/treatment-simulation/treatment_simulator.py."""

from __future__ import annotations


import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("treatment_simulator", "digital-twins/treatment-simulation/treatment_simulator.py")


@pytest.fixture()
def mock_twin(mod):
    """Lightweight mock that satisfies TreatmentSimulator.__init__."""

    class _ClinicalData:
        tumor_grade = "III"

    class _Parameters:
        proliferation_rate = 0.05

    class _Model:
        parameters = _Parameters()

    class _MockTwin:
        patient_id = "MOCK-001"
        current_volume_cm3 = 10.0
        proliferation_rate = 0.05
        clinical_data = _ClinicalData()
        model = _Model()

    return _MockTwin()


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestTreatmentType:
    def test_chemotherapy(self, mod):
        assert mod.TreatmentType.CHEMOTHERAPY.value == "chemotherapy"

    def test_radiation(self, mod):
        assert mod.TreatmentType.RADIATION.value == "radiation"

    def test_immunotherapy(self, mod):
        assert mod.TreatmentType.IMMUNOTHERAPY.value == "immunotherapy"

    def test_surgery(self, mod):
        assert mod.TreatmentType.SURGERY.value == "surgery"

    def test_combined(self, mod):
        assert mod.TreatmentType.COMBINED.value == "combined"


class TestResponseType:
    def test_cr(self, mod):
        assert mod.ResponseType.COMPLETE_RESPONSE.value == "CR"

    def test_pr(self, mod):
        assert mod.ResponseType.PARTIAL_RESPONSE.value == "PR"

    def test_sd(self, mod):
        assert mod.ResponseType.STABLE_DISEASE.value == "SD"

    def test_pd(self, mod):
        assert mod.ResponseType.PROGRESSIVE_DISEASE.value == "PD"

    def test_all_values(self, mod):
        vals = {m.value for m in mod.ResponseType}
        assert vals == {"CR", "PR", "SD", "PD"}


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestTreatmentProtocol:
    def test_creation(self, mod):
        tp = mod.TreatmentProtocol(type=mod.TreatmentType.RADIATION)
        assert tp.type == mod.TreatmentType.RADIATION

    def test_defaults(self, mod):
        tp = mod.TreatmentProtocol(type=mod.TreatmentType.CHEMOTHERAPY)
        assert tp.name == ""
        assert tp.parameters == {}
        assert tp.schedule == {}
        assert tp.duration_days == 0


class TestTreatmentResponse:
    def test_creation(self, mod):
        tr = mod.TreatmentResponse(
            timepoints=np.arange(10),
            volumes=np.ones(10),
            response_category=mod.ResponseType.STABLE_DISEASE,
            volume_change_percent=0.0,
            control_probability=0.5,
        )
        assert tr.response_category == mod.ResponseType.STABLE_DISEASE
        assert tr.volume_change_percent == 0.0

    def test_fields_default_dicts(self, mod):
        tr = mod.TreatmentResponse(
            timepoints=np.arange(5),
            volumes=np.ones(5),
            response_category=mod.ResponseType.PARTIAL_RESPONSE,
            volume_change_percent=-40.0,
            control_probability=0.8,
        )
        assert tr.toxicity_estimate == {}
        assert tr.confidence_interval == {}
        assert tr.metrics == {}


# ---------------------------------------------------------------------------
# TreatmentSimulator tests
# ---------------------------------------------------------------------------


class TestTreatmentSimulator:
    def test_init_with_mock(self, mod, mock_twin):
        sim = mod.TreatmentSimulator(mock_twin)
        assert sim.baseline_volume == pytest.approx(10.0)

    def test_simulate_radiation(self, mod, mock_twin):
        sim = mod.TreatmentSimulator(mock_twin)
        response = sim.predict_response({"type": "radiation", "total_dose_gy": 60, "fractions": 30}, horizon_days=90)
        assert isinstance(response, mod.TreatmentResponse)
        assert len(response.timepoints) > 0
        assert len(response.volumes) == len(response.timepoints)

    def test_simulate_chemotherapy(self, mod, mock_twin):
        sim = mod.TreatmentSimulator(mock_twin)
        response = sim.predict_response(
            {"type": "chemotherapy", "drug": "cisplatin", "dose_mg_m2": 75, "cycles": 4},
            horizon_days=90,
        )
        assert isinstance(response, mod.TreatmentResponse)
        assert response.volume_change_percent is not None

    def test_classify_response_cr(self, mod, mock_twin):
        sim = mod.TreatmentSimulator(mock_twin)
        assert sim._classify_response(-100) == mod.ResponseType.COMPLETE_RESPONSE

    def test_classify_response_pr(self, mod, mock_twin):
        sim = mod.TreatmentSimulator(mock_twin)
        assert sim._classify_response(-50) == mod.ResponseType.PARTIAL_RESPONSE

    def test_classify_response_sd(self, mod, mock_twin):
        sim = mod.TreatmentSimulator(mock_twin)
        assert sim._classify_response(10) == mod.ResponseType.STABLE_DISEASE

    def test_classify_response_pd(self, mod, mock_twin):
        sim = mod.TreatmentSimulator(mock_twin)
        assert sim._classify_response(25) == mod.ResponseType.PROGRESSIVE_DISEASE


# ---------------------------------------------------------------------------
# LinearQuadraticModel tests
# ---------------------------------------------------------------------------


class TestLinearQuadraticModel:
    def test_surviving_fraction_zero_dose(self, mod):
        lq = mod.LinearQuadraticModel()
        sf = lq.surviving_fraction(0.0, 0.3, 0.03)
        assert sf == pytest.approx(1.0)

    def test_surviving_fraction_positive_dose(self, mod):
        lq = mod.LinearQuadraticModel()
        sf = lq.surviving_fraction(2.0, 0.3, 0.03)
        assert 0 < sf < 1

    def test_surviving_fraction_decreases_with_dose(self, mod):
        lq = mod.LinearQuadraticModel()
        sf1 = lq.surviving_fraction(1.0, 0.3, 0.03)
        sf2 = lq.surviving_fraction(3.0, 0.3, 0.03)
        assert sf2 < sf1

    def test_bed_computation(self, mod):
        lq = mod.LinearQuadraticModel()
        bed = lq.biologically_effective_dose(60.0, 2.0, 10.0)
        # BED = 60 * (1 + 2/10) = 60 * 1.2 = 72
        assert bed == pytest.approx(72.0)

    def test_bed_hypofractionation(self, mod):
        lq = mod.LinearQuadraticModel()
        bed_standard = lq.biologically_effective_dose(60.0, 2.0, 10.0)
        bed_hypo = lq.biologically_effective_dose(42.56, 2.66, 10.0)
        # Hypofractionated BED is lower total dose but higher dose/fx
        assert bed_hypo != bed_standard


# ---------------------------------------------------------------------------
# PharmacokineticModel tests
# ---------------------------------------------------------------------------


class TestPharmacokineticModel:
    def test_concentration_at_time_zero(self, mod):
        pk = mod.PharmacokineticModel()
        params = pk.get_drug_parameters("cisplatin")
        conc = pk.concentration_profile(100.0, 0.0, params)
        assert conc == pytest.approx(100.0 / params["vd"])

    def test_concentration_decreases(self, mod):
        pk = mod.PharmacokineticModel()
        params = pk.get_drug_parameters("generic")
        c0 = pk.concentration_profile(100.0, 0.0, params)
        c1 = pk.concentration_profile(100.0, 10.0, params)
        assert c1 < c0

    def test_elimination_exponential(self, mod):
        pk = mod.PharmacokineticModel()
        params = pk.get_drug_parameters("cisplatin")
        c_half = pk.concentration_profile(100.0, params["t_half"], params)
        c0 = pk.concentration_profile(100.0, 0.0, params)
        assert c_half == pytest.approx(c0 / 2.0, rel=0.01)


# ---------------------------------------------------------------------------
# ImmunotherapyModel tests
# ---------------------------------------------------------------------------


class TestImmunotherapyModel:
    def test_default_rates(self, mod):
        im = mod.ImmunotherapyModel()
        assert im.activation_rate == pytest.approx(0.1)
        assert im.kill_rate == pytest.approx(0.05)
        assert im.exhaustion_rate == pytest.approx(0.01)

    def test_predict_response_via_simulator(self, mod, mock_twin):
        sim = mod.TreatmentSimulator(mock_twin)
        response = sim.predict_response({"type": "immunotherapy", "agent": "pembrolizumab"}, horizon_days=60)
        assert isinstance(response, mod.TreatmentResponse)
