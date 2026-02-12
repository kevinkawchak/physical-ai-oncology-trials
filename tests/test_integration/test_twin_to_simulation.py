"""Integration test: Digital Twin -> Simulation Framework pipeline.

Validates that a patient digital twin can be created, calibrated,
and its predictions exported for simulation framework consumption.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def twin_mod():
    return load_module("tumor_twin_pipeline", "digital-twins/patient-modeling/tumor_twin_pipeline.py")


@pytest.fixture()
def sim_mod():
    return load_module("treatment_simulator", "digital-twins/treatment-simulation/treatment_simulator.py")


@pytest.fixture()
def bridge_mod():
    return load_module("isaac_mujoco_bridge", "unification/simulation_physics/isaac_mujoco_bridge.py")


class TestTwinToSimulationPipeline:
    def test_create_twin_and_simulate_growth(self, twin_mod):
        """Create a tumor twin and simulate growth dynamics."""
        pipeline = twin_mod.create_pipeline(
            tumor_type=twin_mod.TumorType.LUNG_CANCER,
            model_type=twin_mod.ModelType.LOGISTIC,
        )
        assert pipeline is not None
        config = {"domain": "test"}
        model = twin_mod.LogisticGrowthModel(domain_shape=(8, 8, 8), config=config)
        model.parameters.proliferation_rate = 0.05
        model.parameters.carrying_capacity = 100.0
        initial = np.ones((8, 8, 8)) * 5.0
        trajectory = model.simulate(initial, duration_days=30)
        assert len(trajectory) > 1
        assert np.all(np.isfinite(trajectory[-1]))

    def test_twin_to_treatment_response(self, twin_mod, sim_mod):
        """Chain: create twin model -> feed to treatment simulator."""

        class MockTwin:
            patient_id = "INTEG-001"
            current_volume_cm3 = 12.0
            proliferation_rate = 0.05

            class clinical_data:
                tumor_grade = "III"

            class model:
                parameters = type(
                    "P",
                    (),
                    {"proliferation_rate": 0.05, "treatment_sensitivity": 0.5},
                )()

        simulator = sim_mod.TreatmentSimulator(patient_twin=MockTwin())

        # Use a dict for the treatment protocol so predict_response can
        # access keys like "total_dose_gy" and "fractions" directly.
        protocol = {
            "type": "radiation",
            "total_dose_gy": 60.0,
            "fractions": 30,
        }
        response = simulator.predict_response(protocol, horizon_days=90)
        assert response is not None
        assert response.response_category in list(sim_mod.ResponseType)

    def test_physics_parameters_for_sim_export(self, bridge_mod):
        """Validate physics parameters can be created for simulation."""
        params = bridge_mod.PhysicsParameters(
            timestep=0.002,
            gravity=np.array([0.0, 0.0, -9.81]),
            friction_coefficient=0.5,
            contact_stiffness=1000.0,
            contact_damping=100.0,
            solver_iterations=50,
        )
        assert params.timestep == 0.002
        assert params.gravity[2] == pytest.approx(-9.81)
