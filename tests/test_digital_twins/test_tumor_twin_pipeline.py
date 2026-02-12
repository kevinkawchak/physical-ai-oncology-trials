"""Tests for digital-twins/patient-modeling/tumor_twin_pipeline.py."""

from __future__ import annotations


import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("tumor_twin_pipeline", "digital-twins/patient-modeling/tumor_twin_pipeline.py")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestTumorType:
    def test_glioblastoma(self, mod):
        assert mod.TumorType.GLIOBLASTOMA.value == "glioblastoma"

    def test_breast_cancer(self, mod):
        assert mod.TumorType.BREAST_CANCER.value == "breast_cancer"

    def test_lung_cancer(self, mod):
        assert mod.TumorType.LUNG_CANCER.value == "lung_cancer"

    def test_prostate_cancer(self, mod):
        assert mod.TumorType.PROSTATE_CANCER.value == "prostate_cancer"

    def test_pancreatic_cancer(self, mod):
        assert mod.TumorType.PANCREATIC_CANCER.value == "pancreatic_cancer"

    def test_all_values_unique(self, mod):
        vals = [m.value for m in mod.TumorType]
        assert len(vals) == len(set(vals))


class TestModelType:
    def test_logistic(self, mod):
        assert mod.ModelType.LOGISTIC.value == "logistic"

    def test_gompertz(self, mod):
        assert mod.ModelType.GOMPERTZ.value == "gompertz"

    def test_reaction_diffusion(self, mod):
        assert mod.ModelType.REACTION_DIFFUSION.value == "reaction_diffusion"

    def test_mechanistic(self, mod):
        assert mod.ModelType.MECHANISTIC.value == "mechanistic"


class TestSolverType:
    def test_cpu_serial(self, mod):
        assert mod.SolverType.CPU_SERIAL.value == "cpu_serial"

    def test_gpu_parallel(self, mod):
        assert mod.SolverType.GPU_PARALLEL.value == "gpu_parallel"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestPatientClinicalData:
    def test_creation_required_fields(self, mod):
        pcd = mod.PatientClinicalData(patient_id="P1", age=55, sex="M")
        assert pcd.patient_id == "P1"
        assert pcd.age == 55
        assert pcd.sex == "M"

    def test_defaults(self, mod):
        pcd = mod.PatientClinicalData(patient_id="P2", age=40, sex="F")
        assert pcd.tumor_grade == "unknown"
        assert pcd.tumor_stage == "unknown"
        assert pcd.molecular_markers == {}
        assert pcd.treatment_history == []
        assert pcd.comorbidities == []

    def test_custom_markers(self, mod):
        pcd = mod.PatientClinicalData(patient_id="P3", age=60, sex="M", molecular_markers={"EGFR": "positive"})
        assert pcd.molecular_markers["EGFR"] == "positive"


class TestModelParameters:
    def test_default_diffusion(self, mod):
        mp = mod.ModelParameters()
        assert mp.diffusion_coefficient == pytest.approx(0.1)

    def test_default_proliferation(self, mod):
        mp = mod.ModelParameters()
        assert mp.proliferation_rate == pytest.approx(0.05)

    def test_default_carrying_capacity(self, mod):
        mp = mod.ModelParameters()
        assert mp.carrying_capacity == pytest.approx(1.0)

    def test_custom_values(self, mod):
        mp = mod.ModelParameters(diffusion_coefficient=0.5, proliferation_rate=0.1)
        assert mp.diffusion_coefficient == pytest.approx(0.5)
        assert mp.proliferation_rate == pytest.approx(0.1)


class TestTwinPrediction:
    def test_creation(self, mod):
        tp = mod.TwinPrediction(
            timepoints=np.array([0, 1, 2]),
            volumes=np.array([10.0, 11.0, 12.0]),
            spatial_distributions=[],
            uncertainty_bounds={"lower": np.array([9.0]), "upper": np.array([13.0])},
        )
        assert len(tp.timepoints) == 3
        assert len(tp.volumes) == 3


# ---------------------------------------------------------------------------
# Growth model tests
# ---------------------------------------------------------------------------


class TestLogisticGrowthModel:
    def test_simulate_with_zeros_no_div_by_zero(self, mod):
        """Regression: simulating with all-zero initial condition must not raise."""
        model = mod.LogisticGrowthModel(domain_shape=(4, 4, 4), config={})
        ic = np.zeros((4, 4, 4))
        trajectory = model.simulate(ic, duration_days=10, dt=1.0)
        # Should return at least two entries (initial + final)
        assert len(trajectory) >= 2
        # Final should also be all zeros since initial was zero
        assert np.allclose(trajectory[-1], 0.0)

    def test_simulate_normal_growth(self, mod):
        model = mod.LogisticGrowthModel(domain_shape=(4, 4, 4), config={})
        ic = np.ones((4, 4, 4)) * 0.01
        trajectory = model.simulate(ic, duration_days=50, dt=0.5)
        assert len(trajectory) >= 2
        # Volume should grow
        initial_sum = np.sum(trajectory[0])
        final_sum = np.sum(trajectory[-1])
        assert final_sum >= initial_sum

    def test_result_shape(self, mod):
        model = mod.LogisticGrowthModel(domain_shape=(4, 4, 4), config={})
        ic = np.ones((4, 4, 4)) * 0.1
        trajectory = model.simulate(ic, duration_days=5, dt=0.5)
        assert trajectory[-1].shape == (4, 4, 4)


class TestGompertzGrowthModel:
    def test_growth_decelerates_toward_capacity(self, mod):
        """Gompertz growth decelerates as V approaches K."""
        model = mod.GompertzGrowthModel(domain_shape=(4, 4, 4), config={})
        model.parameters.carrying_capacity = 100.0
        model.parameters.proliferation_rate = 0.05
        ic = np.ones((4, 4, 4)) * 0.1  # total = 6.4, well below K=100
        traj = model.simulate(ic, duration_days=50, dt=0.5)
        total_final = np.sum(traj[-1])
        # Should grow but remain below carrying capacity
        assert total_final > np.sum(ic)
        assert total_final <= model.parameters.carrying_capacity * 1.1

    def test_result_is_list(self, mod):
        model = mod.GompertzGrowthModel(domain_shape=(4, 4, 4), config={})
        ic = np.ones((4, 4, 4)) * 0.1
        traj = model.simulate(ic, duration_days=5, dt=0.5)
        assert isinstance(traj, list)


# ---------------------------------------------------------------------------
# PatientDigitalTwin tests
# ---------------------------------------------------------------------------


class TestPatientDigitalTwin:
    def _make_twin(self, mod):
        mask = np.zeros((8, 8, 8))
        mask[3:5, 3:5, 3:5] = 1.0
        model = mod.LogisticGrowthModel(domain_shape=(8, 8, 8), config={})
        clinical = mod.PatientClinicalData(patient_id="T1", age=50, sex="F")
        twin = mod.PatientDigitalTwin(
            patient_id="T1",
            model=model,
            imaging_data={"mri": np.zeros((8, 8, 8))},
            tumor_mask=mask,
            clinical_data=clinical,
            config={},
            solver=mod.SolverType.CPU_SERIAL,
        )
        return twin

    def test_creation(self, mod):
        twin = self._make_twin(mod)
        assert twin.patient_id == "T1"
        assert twin.is_calibrated is False

    def test_current_volume(self, mod):
        twin = self._make_twin(mod)
        # 8 voxels at 1mm voxel size = 8mm^3 = 0.008 cm^3
        assert twin.current_volume_cm3 == pytest.approx(0.008, abs=0.001)

    def test_predict_returns_prediction(self, mod):
        twin = self._make_twin(mod)
        prediction = twin.predict(horizon_days=10)
        assert isinstance(prediction, mod.TwinPrediction)
        assert len(prediction.timepoints) > 0
        assert len(prediction.volumes) > 0

    def test_predict_has_metrics(self, mod):
        twin = self._make_twin(mod)
        prediction = twin.predict(horizon_days=10)
        assert "initial_volume_cm3" in prediction.metrics
        assert "final_volume_cm3" in prediction.metrics
        assert "volume_change_percent" in prediction.metrics
