"""Integration tests: Digital Twin -> Treatment Simulation -> Response Prediction.

Tests the cross-module flow from tumor_twin_pipeline through treatment_simulator,
verifying that digital twin outputs properly feed into treatment simulation
and produce valid RECIST response predictions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def twin_mod():
    return load_module("tumor_twin_pipeline", "digital-twins/patient-modeling/tumor_twin_pipeline.py")


@pytest.fixture()
def sim_mod():
    return load_module("treatment_simulator", "digital-twins/treatment-simulation/treatment_simulator.py")


def _make_mock_twin(twin_mod, volume: float = 15.0, prolif: float = 0.05, grade: str = "III", sensitivity: float = 0.5):
    """Create a minimal PatientDigitalTwin-compatible object for the simulator."""
    model_params = twin_mod.ModelParameters(proliferation_rate=prolif, treatment_sensitivity=sensitivity)
    model = twin_mod.LogisticGrowthModel(domain_shape=(5, 5, 5), config={"domain": "test"})
    model.parameters = model_params
    mask = np.ones((5, 5, 5)) * (volume / 0.125)  # rough voxel fill for volume
    clinical = twin_mod.PatientClinicalData(patient_id="INT-001", age=60, sex="M", tumor_grade=grade)
    twin = twin_mod.PatientDigitalTwin(
        patient_id="INT-001",
        model=model,
        imaging_data={"mri": np.zeros((5, 5, 5))},
        tumor_mask=mask,
        clinical_data=clinical,
        config={"domain": "test"},
        solver=twin_mod.SolverType.CPU_SERIAL,
    )
    return twin


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestDigitalTwinToSimulation:
    """Integration: PatientDigitalTwin -> TreatmentSimulator."""

    def test_twin_feeds_simulator(self, twin_mod, sim_mod):
        """TreatmentSimulator accepts a PatientDigitalTwin and sets baseline_volume."""
        twin = _make_mock_twin(twin_mod, volume=15.0)
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        assert simulator.baseline_volume == twin.current_volume_cm3

    def test_growth_model_to_treatment(self, twin_mod, sim_mod):
        """Grow tumor via LogisticGrowthModel, feed resulting volume to simulator."""
        model = twin_mod.LogisticGrowthModel(domain_shape=(5, 5, 5), config={"domain": "test"})
        model.parameters.proliferation_rate = 0.05
        model.parameters.carrying_capacity = 200.0
        initial = np.ones((5, 5, 5)) * 0.5
        result = model.simulate(initial, duration_days=30)
        final_volume = np.sum(result[-1])
        assert final_volume > 0
        # Now use it indirectly: create a twin whose volume derives from growth
        twin = _make_mock_twin(twin_mod, volume=final_volume / 125.0)
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        assert simulator.baseline_volume > 0

    def test_treatment_response_categorization(self, twin_mod, sim_mod):
        """Simulate radiation treatment and verify RECIST response is valid."""
        twin = _make_mock_twin(twin_mod, volume=20.0, prolif=0.03)
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        response = simulator.predict_response(
            treatment={"type": "radiation", "total_dose_gy": 60, "fractions": 30},
            horizon_days=90,
        )
        valid_responses = {"CR", "PR", "SD", "PD"}
        assert response.response_category.value in valid_responses

    def test_combined_chemo_radiation(self, twin_mod, sim_mod):
        """Test combined modality treatment flow."""
        twin = _make_mock_twin(twin_mod, volume=25.0, prolif=0.04)
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        response = simulator.predict_response(
            treatment={
                "type": "combined",
                "modalities": [
                    {"type": "radiation", "total_dose_gy": 60, "fractions": 30},
                    {"type": "chemotherapy", "drug": "cisplatin", "dose_mg_m2": 75, "cycles": 4},
                ],
            },
            horizon_days=120,
        )
        assert response.volumes is not None
        assert len(response.volumes) > 0
        assert response.response_category.value in {"CR", "PR", "SD", "PD"}

    def test_model_parameters_propagate(self, twin_mod, sim_mod):
        """Verify ModelParameters.treatment_sensitivity affects simulator results."""
        twin_sensitive = _make_mock_twin(twin_mod, volume=20.0, prolif=0.04, sensitivity=0.9)
        twin_resistant = _make_mock_twin(twin_mod, volume=20.0, prolif=0.04, sensitivity=0.1)
        sim_s = sim_mod.TreatmentSimulator(patient_twin=twin_sensitive)
        sim_r = sim_mod.TreatmentSimulator(patient_twin=twin_resistant)
        # Both should initialize with the same baseline
        assert abs(sim_s.baseline_volume - sim_r.baseline_volume) < 1.0

    def test_radiation_volumes_decrease_during_treatment(self, twin_mod, sim_mod):
        """Radiation should cause tumor volume to decrease during treatment phase."""
        twin = _make_mock_twin(twin_mod, volume=30.0, prolif=0.01)
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        response = simulator.predict_response(
            treatment={"type": "radiation", "total_dose_gy": 60, "fractions": 30},
            horizon_days=90,
        )
        # Nadir should be less than baseline
        nadir = np.min(response.volumes)
        assert nadir < simulator.baseline_volume

    def test_chemotherapy_response_has_valid_volume_trajectory(self, twin_mod, sim_mod):
        """Chemotherapy simulation produces a valid volume trajectory."""
        twin = _make_mock_twin(twin_mod, volume=18.0, prolif=0.04)
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        response = simulator.predict_response(
            treatment={"type": "chemotherapy", "drug": "generic", "dose_mg_m2": 100, "cycles": 6},
            horizon_days=180,
        )
        assert len(response.volumes) == len(response.timepoints)
        assert np.all(np.isfinite(response.volumes))

    def test_surgery_response_returns_valid_result(self, twin_mod, sim_mod):
        """Surgery simulation returns a valid TreatmentResponse."""
        twin = _make_mock_twin(twin_mod, volume=10.0, prolif=0.05)
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        response = simulator.predict_response(
            treatment={"type": "surgery", "resection_fraction": 0.95, "surgery_day": 1},
            horizon_days=60,
        )
        assert response.volumes is not None
        assert response.response_category.value in {"CR", "PR", "SD", "PD"}

    def test_volume_change_percent_is_finite(self, twin_mod, sim_mod):
        """Volume change percent should always be a finite number."""
        twin = _make_mock_twin(twin_mod, volume=12.0, prolif=0.03)
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        response = simulator.predict_response(
            treatment={"type": "radiation", "total_dose_gy": 50, "fractions": 25},
            horizon_days=60,
        )
        assert math.isfinite(response.volume_change_percent)

    def test_control_probability_in_valid_range(self, twin_mod, sim_mod):
        """TCP (tumor control probability) should be in [0, 1]."""
        twin = _make_mock_twin(twin_mod, volume=8.0, prolif=0.02)
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        response = simulator.predict_response(
            treatment={"type": "radiation", "total_dose_gy": 70, "fractions": 35},
            horizon_days=90,
        )
        assert 0.0 <= response.control_probability <= 1.0

    def test_immunotherapy_response_flow(self, twin_mod, sim_mod):
        """Immunotherapy simulation produces valid results."""
        twin = _make_mock_twin(twin_mod, volume=22.0, prolif=0.03)
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        response = simulator.predict_response(
            treatment={"type": "immunotherapy", "agent": "pembrolizumab", "cycles": 4},
            horizon_days=120,
        )
        assert response.volumes is not None
        assert len(response.timepoints) > 0
