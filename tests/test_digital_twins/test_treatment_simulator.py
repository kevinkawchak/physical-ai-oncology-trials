"""Unit tests for digital-twins/treatment-simulation/treatment_simulator.py.

Tests cover enums, dataclasses, LinearQuadraticModel, PharmacokineticModel,
ImmunotherapyModel, and TreatmentSimulator.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "digital-twins/treatment-simulation/treatment_simulator.py"


@pytest.fixture()
def mod():
    return load_module("treatment_simulator", MOD_PATH)


@pytest.fixture()
def mock_twin(mod):
    """Mock patient twin for TreatmentSimulator."""

    class MockTwin:
        patient_id = "TEST-SIM-001"
        current_volume_cm3 = 15.0
        proliferation_rate = 0.05

        class clinical_data:
            tumor_grade = "III"

        class model:
            parameters = type(
                "P",
                (),
                {
                    "proliferation_rate": 0.05,
                    "treatment_sensitivity": 0.5,
                },
            )()

    return MockTwin()


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_treatment_type_members(self, mod):
        names = [e.name for e in mod.TreatmentType]
        for expected in ("CHEMOTHERAPY", "RADIATION", "IMMUNOTHERAPY", "SURGERY"):
            assert expected in names

    def test_response_type_recist(self, mod):
        names = [e.name for e in mod.ResponseType]
        for expected in (
            "COMPLETE_RESPONSE",
            "PARTIAL_RESPONSE",
            "STABLE_DISEASE",
            "PROGRESSIVE_DISEASE",
        ):
            assert expected in names


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_treatment_protocol_creation(self, mod):
        protocol = mod.TreatmentProtocol(
            type=mod.TreatmentType.RADIATION,
            name="Standard RT",
            parameters={"total_dose": 60.0, "fractions": 30},
            schedule="daily",
            duration_days=42,
        )
        assert protocol.type == mod.TreatmentType.RADIATION
        assert protocol.duration_days == 42

    def test_treatment_response_fields(self, mod):
        resp = mod.TreatmentResponse(
            timepoints=np.arange(10),
            volumes=np.linspace(15.0, 10.0, 10),
            response_category=mod.ResponseType.PARTIAL_RESPONSE,
            volume_change_percent=-33.3,
            control_probability=0.5,
        )
        assert resp.response_category == mod.ResponseType.PARTIAL_RESPONSE
        assert resp.volume_change_percent < 0


# ---------------------------------------------------------------------------
# LinearQuadraticModel tests
# ---------------------------------------------------------------------------


class TestLinearQuadraticModel:
    def test_surviving_fraction_range(self, mod):
        lqm = mod.LinearQuadraticModel()
        sf = lqm.surviving_fraction(dose=2.0, alpha=0.3, beta=0.03)
        assert 0.0 < sf < 1.0

    def test_surviving_fraction_zero_dose(self, mod):
        lqm = mod.LinearQuadraticModel()
        sf = lqm.surviving_fraction(dose=0.0, alpha=0.3, beta=0.03)
        assert sf == pytest.approx(1.0)

    def test_bed_calculation(self, mod):
        lqm = mod.LinearQuadraticModel()
        bed = lqm.biologically_effective_dose(total_dose=60.0, dose_per_fraction=2.0, alpha_beta=10.0)
        # BED = 60 * (1 + 2/10) = 72 Gy
        assert bed == pytest.approx(72.0, rel=0.01)

    def test_tcp_bounded(self, mod):
        lqm = mod.LinearQuadraticModel()
        tcp = lqm.tumor_control_probability(initial_cells=1e8, total_sf=1e-12)
        assert 0.0 <= tcp <= 1.0

    def test_tissue_params_available(self, mod):
        lqm = mod.LinearQuadraticModel()
        params = lqm.get_tissue_parameters("III")
        # Returns a tuple (alpha, beta)
        assert isinstance(params, tuple)
        assert len(params) == 2
        assert params[0] > 0 and params[1] > 0


# ---------------------------------------------------------------------------
# PharmacokineticModel tests
# ---------------------------------------------------------------------------


class TestPharmacokineticModel:
    def test_concentration_decays(self, mod):
        pk = mod.PharmacokineticModel()
        params = pk.get_drug_parameters("cisplatin")
        c_early = pk.concentration_profile(dose_mg_m2=75.0, time_hours=1.0, params=params)
        c_late = pk.concentration_profile(dose_mg_m2=75.0, time_hours=24.0, params=params)
        assert c_early > c_late

    def test_zero_dose_zero_concentration(self, mod):
        pk = mod.PharmacokineticModel()
        params = pk.get_drug_parameters("cisplatin")
        c = pk.concentration_profile(dose_mg_m2=0.0, time_hours=1.0, params=params)
        assert c == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TreatmentSimulator tests
# ---------------------------------------------------------------------------


class TestTreatmentSimulator:
    def test_creation(self, mod, mock_twin):
        sim = mod.TreatmentSimulator(patient_twin=mock_twin)
        assert sim.baseline_volume == 15.0

    def test_predict_response_returns_response(self, mod, mock_twin):
        sim = mod.TreatmentSimulator(patient_twin=mock_twin)
        protocol = {
            "type": "radiation",
            "total_dose_gy": 60.0,
            "fractions": 30,
        }
        response = sim.predict_response(protocol, horizon_days=90)
        assert isinstance(response, mod.TreatmentResponse)
        assert len(response.timepoints) > 0
        assert response.response_category in list(mod.ResponseType)

    def test_create_simulator_factory(self, mod, mock_twin):
        sim = mod.create_simulator(patient_twin=mock_twin)
        assert isinstance(sim, mod.TreatmentSimulator)
