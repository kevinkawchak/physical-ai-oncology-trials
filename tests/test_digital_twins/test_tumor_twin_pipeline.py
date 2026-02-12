"""Tests for digital-twins/patient-modeling/tumor_twin_pipeline.py

Covers TumorType, ModelType, growth models, and PatientDigitalTwin.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── Enum values ──────────────────────────────────────────────────────────


class TestEnums:
    def test_tumor_types(self, twin_pipeline_mod):
        tt = twin_pipeline_mod.TumorType
        assert tt.GLIOBLASTOMA.value == "glioblastoma"
        assert tt.BREAST_CANCER.value == "breast_cancer"
        assert tt.LUNG_CANCER.value == "lung_cancer"

    def test_model_types(self, twin_pipeline_mod):
        mt = twin_pipeline_mod.ModelType
        assert mt.LOGISTIC.value == "logistic"
        assert mt.GOMPERTZ.value == "gompertz"
        assert mt.REACTION_DIFFUSION.value == "reaction_diffusion"

    def test_solver_types(self, twin_pipeline_mod):
        st = twin_pipeline_mod.SolverType
        assert st.CPU_SERIAL.value == "cpu_serial"
        assert st.GPU_PARALLEL.value == "gpu_parallel"


# ── PatientClinicalData ──────────────────────────────────────────────────


class TestPatientClinicalData:
    def test_creation(self, twin_pipeline_mod):
        data = twin_pipeline_mod.PatientClinicalData(
            patient_id="P-001",
            age=55,
            sex="M",
            tumor_grade="III",
        )
        assert data.patient_id == "P-001"
        assert data.age == 55
        assert data.tumor_grade == "III"

    def test_defaults(self, twin_pipeline_mod):
        data = twin_pipeline_mod.PatientClinicalData(
            patient_id="P-002",
            age=60,
            sex="F",
        )
        assert data.tumor_grade == "unknown"
        assert data.molecular_markers == {}
        assert data.treatment_history == []


# ── ModelParameters ──────────────────────────────────────────────────────


class TestModelParameters:
    def test_defaults(self, twin_pipeline_mod):
        params = twin_pipeline_mod.ModelParameters()
        assert params.diffusion_coefficient == 0.1
        assert params.proliferation_rate == 0.05
        assert params.carrying_capacity == 1.0


# ── LogisticGrowthModel ──────────────────────────────────────────────────


class TestLogisticGrowthModel:
    def test_simulate_zero_volume_no_crash(self, twin_pipeline_mod):
        """Regression: v0.9.2 fixed division by zero with zero initial volume."""
        config = {"domain": "test"}
        model = twin_pipeline_mod.LogisticGrowthModel(
            domain_shape=(10, 10, 10),
            config=config,
        )
        model.parameters.proliferation_rate = 0.05
        model.parameters.carrying_capacity = 100.0
        initial = np.zeros((10, 10, 10))
        result = model.simulate(initial, duration_days=30)
        assert len(result) >= 1
        assert np.all(np.isfinite(result[-1]))

    def test_simulate_nonzero_initial_grows(self, twin_pipeline_mod):
        config = {"domain": "test"}
        model = twin_pipeline_mod.LogisticGrowthModel(
            domain_shape=(10, 10, 10),
            config=config,
        )
        model.parameters.proliferation_rate = 0.1
        model.parameters.carrying_capacity = 100.0
        initial = np.zeros((10, 10, 10))
        initial[4:6, 4:6, 4:6] = 0.5
        result = model.simulate(initial, duration_days=30)
        initial_sum = initial.sum()
        final_sum = result[-1].sum()
        assert final_sum >= initial_sum


# ── TumorTwinPipeline ────────────────────────────────────────────────────


class TestTumorTwinPipeline:
    def test_creation_defaults(self, twin_pipeline_mod):
        pipeline = twin_pipeline_mod.TumorTwinPipeline()
        assert pipeline.tumor_type == twin_pipeline_mod.TumorType.GLIOBLASTOMA

    def test_creation_custom(self, twin_pipeline_mod):
        pipeline = twin_pipeline_mod.TumorTwinPipeline(
            model_type=twin_pipeline_mod.ModelType.LOGISTIC,
            tumor_type=twin_pipeline_mod.TumorType.BREAST_CANCER,
        )
        assert pipeline.tumor_type == twin_pipeline_mod.TumorType.BREAST_CANCER

    def test_create_twin(self, twin_pipeline_mod, synthetic_planning_ct, synthetic_tumor_mask):
        pipeline = twin_pipeline_mod.TumorTwinPipeline(
            model_type=twin_pipeline_mod.ModelType.LOGISTIC,
            solver=twin_pipeline_mod.SolverType.CPU_SERIAL,
        )
        clinical = twin_pipeline_mod.PatientClinicalData(
            patient_id="TWIN-001",
            age=55,
            sex="M",
            tumor_grade="III",
        )
        twin = pipeline.create_twin(
            patient_id="TWIN-001",
            imaging_data={"t1": synthetic_planning_ct},
            tumor_segmentation=synthetic_tumor_mask,
            clinical_data=clinical,
        )
        assert twin.patient_id == "TWIN-001"
        assert twin.current_volume_cm3 > 0


# ── create_pipeline helper ───────────────────────────────────────────────


class TestCreatePipeline:
    def test_creates_pipeline(self, twin_pipeline_mod):
        pipeline = twin_pipeline_mod.create_pipeline(
            tumor_type="glioblastoma",
            use_gpu=False,
        )
        assert isinstance(pipeline, twin_pipeline_mod.TumorTwinPipeline)

    def test_invalid_tumor_type(self, twin_pipeline_mod):
        with pytest.raises((ValueError, KeyError)):
            twin_pipeline_mod.create_pipeline(tumor_type="nonexistent")
