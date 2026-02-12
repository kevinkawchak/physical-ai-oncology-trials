"""Tests for digital-twins/treatment-simulation/treatment_simulator.py

Covers TreatmentType, ResponseType, LinearQuadraticModel,
PharmacokineticModel, and TreatmentSimulator.
"""

from __future__ import annotations

import math

import numpy as np

from tests.conftest import load_module


# ── Enum values ──────────────────────────────────────────────────────────


class TestEnums:
    def test_treatment_types(self, treatment_sim_mod):
        tt = treatment_sim_mod.TreatmentType
        assert tt.CHEMOTHERAPY.value == "chemotherapy"
        assert tt.RADIATION.value == "radiation"
        assert tt.IMMUNOTHERAPY.value == "immunotherapy"
        assert tt.SURGERY.value == "surgery"

    def test_response_types(self, treatment_sim_mod):
        rt = treatment_sim_mod.ResponseType
        assert rt.COMPLETE_RESPONSE.value == "CR"
        assert rt.PROGRESSIVE_DISEASE.value == "PD"


# ── TreatmentProtocol dataclass ──────────────────────────────────────────


class TestTreatmentProtocol:
    def test_creation(self, treatment_sim_mod):
        proto = treatment_sim_mod.TreatmentProtocol(
            type=treatment_sim_mod.TreatmentType.RADIATION,
            name="Standard RT",
        )
        assert proto.name == "Standard RT"
        assert proto.duration_days == 0
        assert proto.parameters == {}


# ── LinearQuadraticModel ─────────────────────────────────────────────────


class TestLinearQuadraticModel:
    def test_surviving_fraction_decreases_with_dose(self, treatment_sim_mod):
        lq = treatment_sim_mod.LinearQuadraticModel()
        sf_low = lq.surviving_fraction(1.0, alpha=0.3, beta=0.03)
        sf_high = lq.surviving_fraction(3.0, alpha=0.3, beta=0.03)
        assert sf_high < sf_low

    def test_surviving_fraction_bounded(self, treatment_sim_mod):
        lq = treatment_sim_mod.LinearQuadraticModel()
        sf = lq.surviving_fraction(2.0, alpha=0.3, beta=0.03)
        assert 0.0 < sf <= 1.0

    def test_bed_standard_fractionation(self, treatment_sim_mod):
        lq = treatment_sim_mod.LinearQuadraticModel()
        bed = lq.biologically_effective_dose(60.0, 2.0, 10.0)
        # BED = 60 * (1 + 2/10) = 72
        assert math.isclose(bed, 72.0, rel_tol=1e-9)

    def test_tissue_parameters(self, treatment_sim_mod):
        lq = treatment_sim_mod.LinearQuadraticModel()
        alpha, beta = lq.get_tissue_parameters("III")
        assert alpha > 0
        assert beta > 0


# ── PharmacokineticModel ─────────────────────────────────────────────────


class TestPharmacokineticModel:
    def test_drug_parameters(self, treatment_sim_mod):
        pk = treatment_sim_mod.PharmacokineticModel()
        params = pk.get_drug_parameters("cisplatin")
        assert isinstance(params, dict)
        assert "t_half" in params

    def test_concentration_profile(self, treatment_sim_mod):
        pk = treatment_sim_mod.PharmacokineticModel()
        params = pk.get_drug_parameters("cisplatin")
        conc = pk.concentration_profile(75.0, 6.0, params)
        assert conc >= 0


# ── TreatmentSimulator ──────────────────────────────────────────────────


class TestTreatmentSimulator:
    def _make_mock_twin(self, twin_pipeline_mod):
        config = {"domain": "test"}
        model = twin_pipeline_mod.LogisticGrowthModel(
            domain_shape=(10, 10, 10),
            config=config,
        )
        model.parameters.proliferation_rate = 0.05
        model.parameters.carrying_capacity = 100.0
        model.parameters.treatment_sensitivity = 0.5

        initial = np.zeros((10, 10, 10))
        initial[3:7, 3:7, 3:7] = 1.0

        clinical = twin_pipeline_mod.PatientClinicalData(
            patient_id="SIM-001",
            age=55,
            sex="M",
            tumor_grade="III",
        )

        pipeline = twin_pipeline_mod.TumorTwinPipeline(
            model_type=twin_pipeline_mod.ModelType.LOGISTIC,
            solver=twin_pipeline_mod.SolverType.CPU_SERIAL,
        )
        twin = pipeline.create_twin(
            patient_id="SIM-001",
            imaging_data={"t1": np.random.rand(10, 10, 10).astype(np.float32)},
            tumor_segmentation=initial,
            clinical_data=clinical,
        )
        return twin

    def test_creation(self, treatment_sim_mod):
        twin_mod = load_module(
            "tumor_twin_pipeline",
            "digital-twins/patient-modeling/tumor_twin_pipeline.py",
        )
        twin = self._make_mock_twin(twin_mod)
        sim = treatment_sim_mod.TreatmentSimulator(patient_twin=twin)
        assert sim.baseline_volume > 0

    def test_predict_radiation_response(self, treatment_sim_mod):
        twin_mod = load_module(
            "tumor_twin_pipeline",
            "digital-twins/patient-modeling/tumor_twin_pipeline.py",
        )
        twin = self._make_mock_twin(twin_mod)
        sim = treatment_sim_mod.TreatmentSimulator(patient_twin=twin)
        treatment = {
            "type": "radiation",
            "total_dose_gy": 60.0,
            "fractions": 30,
        }
        response = sim.predict_response(treatment, horizon_days=90)
        assert response.volumes is not None
        assert len(response.volumes) > 0
        assert response.response_category is not None
