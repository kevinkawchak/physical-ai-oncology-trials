"""Unit tests for digital-twins/patient-modeling/tumor_twin_pipeline.py.

Tests cover enums, dataclasses, growth models (Logistic, Gompertz),
PatientDigitalTwin, and the create_pipeline factory.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "digital-twins/patient-modeling/tumor_twin_pipeline.py"


@pytest.fixture()
def mod():
    return load_module("tumor_twin_pipeline", MOD_PATH)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_tumor_type_members(self, mod):
        names = [e.name for e in mod.TumorType]
        for expected in ("GLIOBLASTOMA", "BREAST_CANCER", "LUNG_CANCER"):
            assert expected in names

    def test_model_type_members(self, mod):
        names = [e.name for e in mod.ModelType]
        assert "LOGISTIC" in names
        assert "GOMPERTZ" in names
        assert "REACTION_DIFFUSION" in names

    def test_solver_type_members(self, mod):
        names = [e.name for e in mod.SolverType]
        assert "CPU_SERIAL" in names
        assert "GPU_PARALLEL" in names


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_patient_clinical_data_fields(self, mod):
        pcd = mod.PatientClinicalData(
            patient_id="PT-001",
            age=62,
            sex="M",
            tumor_grade="III",
            tumor_stage="IIIA",
        )
        assert pcd.patient_id == "PT-001"
        assert pcd.age == 62

    def test_model_parameters_defaults(self, mod):
        mp = mod.ModelParameters()
        assert mp.proliferation_rate >= 0
        assert mp.carrying_capacity > 0

    def test_twin_prediction_fields(self, mod):
        tp = mod.TwinPrediction(
            timepoints=np.arange(5),
            volumes=np.ones(5),
            spatial_distributions=[],
            uncertainty_bounds={},
        )
        assert len(tp.timepoints) == 5
        assert len(tp.volumes) == 5


# ---------------------------------------------------------------------------
# Growth model tests
# ---------------------------------------------------------------------------


class TestLogisticGrowthModel:
    def test_instantiation(self, mod):
        config = {"domain": "test"}
        model = mod.LogisticGrowthModel(domain_shape=(10, 10, 10), config=config)
        assert model is not None

    def test_simulate_returns_nonempty(self, mod):
        config = {"domain": "test"}
        model = mod.LogisticGrowthModel(domain_shape=(10, 10, 10), config=config)
        model.parameters.proliferation_rate = 0.05
        model.parameters.carrying_capacity = 100.0
        initial = np.ones((10, 10, 10)) * 5.0
        result = model.simulate(initial, duration_days=10)
        assert len(result) >= 1
        assert np.all(np.isfinite(result[-1]))

    def test_simulate_zero_initial_no_crash(self, mod):
        """Post-resection scenario: zero initial volume must not crash."""
        config = {"domain": "test"}
        model = mod.LogisticGrowthModel(domain_shape=(10, 10, 10), config=config)
        model.parameters.proliferation_rate = 0.05
        model.parameters.carrying_capacity = 100.0
        initial = np.zeros((10, 10, 10))
        result = model.simulate(initial, duration_days=30)
        assert len(result) >= 1
        assert np.all(np.isfinite(result[-1]))

    def test_growth_is_bounded(self, mod):
        """Logistic growth should not exceed carrying capacity."""
        config = {"domain": "test"}
        model = mod.LogisticGrowthModel(domain_shape=(5, 5, 5), config=config)
        model.parameters.proliferation_rate = 0.1
        model.parameters.carrying_capacity = 50.0
        initial = np.ones((5, 5, 5)) * 2.0
        result = model.simulate(initial, duration_days=100)
        assert np.max(result[-1]) <= model.parameters.carrying_capacity * 1.1


class TestGompertzGrowthModel:
    def test_instantiation(self, mod):
        config = {"domain": "test"}
        model = mod.GompertzGrowthModel(domain_shape=(8, 8, 8), config=config)
        assert model is not None

    def test_simulate_produces_finite(self, mod):
        config = {"domain": "test"}
        model = mod.GompertzGrowthModel(domain_shape=(8, 8, 8), config=config)
        model.parameters.proliferation_rate = 0.03
        model.parameters.carrying_capacity = 80.0
        initial = np.ones((8, 8, 8)) * 3.0
        result = model.simulate(initial, duration_days=20)
        assert np.all(np.isfinite(result[-1]))


# ---------------------------------------------------------------------------
# Pipeline factory tests
# ---------------------------------------------------------------------------


class TestCreatePipeline:
    def test_factory_returns_pipeline(self, mod):
        pipeline = mod.create_pipeline(
            tumor_type="lung_cancer",
            model_type="logistic",
        )
        assert isinstance(pipeline, mod.TumorTwinPipeline)

    def test_pipeline_attributes(self, mod):
        pipeline = mod.create_pipeline(
            tumor_type="breast_cancer",
            model_type="gompertz",
        )
        assert pipeline.tumor_type == mod.TumorType.BREAST_CANCER
        assert pipeline.model_type == mod.ModelType.GOMPERTZ
