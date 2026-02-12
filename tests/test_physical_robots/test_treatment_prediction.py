from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("treatment_response_prediction", "examples/05_treatment_response_prediction.py")


@pytest.fixture()
def sample_patient(mod):
    return mod.PatientProfile(
        patient_id="TEST-001",
        age=60,
        sex="F",
        tumor_type=mod.TumorType.NSCLC,
        stage="IIIA",
        tumor_volume_cm3=12.0,
        biomarkers={"PD-L1": 50},
        performance_status=1,
    )


@pytest.fixture()
def sample_tumor(mod):
    return mod.TumorCharacteristics(
        proliferation_rate=0.04,
        death_rate=0.01,
        drug_sensitivity=0.5,
        radiation_sensitivity=0.3,
        immune_visibility=0.6,
    )


# ---------------------------------------------------------------------------
# TestTumorTypeEnum
# ---------------------------------------------------------------------------


class TestTumorTypeEnum:
    """Tests for TumorType enum."""

    def test_nsclc(self, mod):
        assert mod.TumorType.NSCLC.value == "non_small_cell_lung_cancer"

    def test_sclc(self, mod):
        assert mod.TumorType.SCLC.value == "small_cell_lung_cancer"

    def test_breast_tnbc(self, mod):
        assert mod.TumorType.BREAST_TNBC.value == "triple_negative_breast_cancer"

    def test_colorectal(self, mod):
        assert mod.TumorType.COLORECTAL.value == "colorectal_cancer"

    def test_glioblastoma(self, mod):
        assert mod.TumorType.GLIOBLASTOMA.value == "glioblastoma"


# ---------------------------------------------------------------------------
# TestTreatmentModalityEnum
# ---------------------------------------------------------------------------


class TestTreatmentModalityEnum:
    """Tests for TreatmentModality enum."""

    def test_chemotherapy(self, mod):
        assert mod.TreatmentModality.CHEMOTHERAPY.value == "chemotherapy"

    def test_radiation(self, mod):
        assert mod.TreatmentModality.RADIATION.value == "radiation"

    def test_immunotherapy(self, mod):
        assert mod.TreatmentModality.IMMUNOTHERAPY.value == "immunotherapy"

    def test_surgery(self, mod):
        assert mod.TreatmentModality.SURGERY.value == "surgery"


# ---------------------------------------------------------------------------
# TestPatientProfile
# ---------------------------------------------------------------------------


class TestPatientProfile:
    """Tests for PatientProfile dataclass."""

    def test_creation(self, mod, sample_patient):
        assert sample_patient.patient_id == "TEST-001"
        assert sample_patient.age == 60

    def test_default_biomarkers(self, mod):
        p = mod.PatientProfile(
            patient_id="X",
            age=50,
            sex="M",
            tumor_type=mod.TumorType.NSCLC,
            stage="II",
            tumor_volume_cm3=5.0,
        )
        assert p.biomarkers == {}
        assert p.comorbidities == []


# ---------------------------------------------------------------------------
# TestTumorCharacteristics
# ---------------------------------------------------------------------------


class TestTumorCharacteristics:
    """Tests for TumorCharacteristics dataclass."""

    def test_creation(self, mod, sample_tumor):
        assert sample_tumor.proliferation_rate == 0.04

    def test_defaults(self, mod):
        tc = mod.TumorCharacteristics()
        assert tc.proliferation_rate == 0.05
        assert tc.death_rate == 0.01


# ---------------------------------------------------------------------------
# TestExponentialGrowthModel
# ---------------------------------------------------------------------------


class TestExponentialGrowthModel:
    """Tests for ExponentialGrowthModel class."""

    def test_simulate_increases(self, mod, sample_tumor):
        model = mod.ExponentialGrowthModel(sample_tumor)
        volumes = model.simulate(initial_volume=10.0, duration_days=30)
        assert volumes[-1] > volumes[0]

    def test_net_growth_rate_positive(self, mod, sample_tumor):
        net = sample_tumor.proliferation_rate - sample_tumor.death_rate
        assert net > 0

    def test_simulate_length(self, mod, sample_tumor):
        model = mod.ExponentialGrowthModel(sample_tumor)
        volumes = model.simulate(initial_volume=10.0, duration_days=10)
        assert len(volumes) == 11  # 0..10 inclusive


# ---------------------------------------------------------------------------
# TestGompertzGrowthModel
# ---------------------------------------------------------------------------


class TestGompertzGrowthModel:
    """Tests for GompertzGrowthModel class."""

    def test_simulate_approaches_capacity(self, mod, sample_tumor):
        model = mod.GompertzGrowthModel(sample_tumor, carrying_capacity=100.0)
        volumes = model.simulate(initial_volume=10.0, duration_days=500)
        # Should grow toward but not exceed carrying capacity
        assert volumes[-1] <= 100.0 * 1.05  # Allow slight numerical overshoot

    def test_simulate_grows(self, mod, sample_tumor):
        model = mod.GompertzGrowthModel(sample_tumor, carrying_capacity=1000.0)
        volumes = model.simulate(initial_volume=10.0, duration_days=30)
        assert volumes[-1] > volumes[0]


# ---------------------------------------------------------------------------
# TestTreatmentResponseModel
# ---------------------------------------------------------------------------


class TestTreatmentResponseModel:
    """Tests for TreatmentResponseModel class."""

    def test_init(self, mod, sample_patient, sample_tumor):
        model = mod.TreatmentResponseModel(sample_patient, sample_tumor)
        assert model.patient is sample_patient

    def test_predict_chemotherapy(self, mod, sample_patient, sample_tumor):
        model = mod.TreatmentResponseModel(sample_patient, sample_tumor)
        protocol = mod.TreatmentProtocol(
            modality=mod.TreatmentModality.CHEMOTHERAPY,
            name="TestChemo",
            parameters={"drug": "carboplatin", "cycle_days": 21},
            cycles=4,
        )
        outcome = model.predict_response(protocol, baseline_volume=12.0, horizon_days=180)
        assert isinstance(outcome, mod.TreatmentOutcome)
        assert outcome.response_category in ("CR", "PR", "SD", "PD")

    def test_classify_response_categories(self, mod, sample_patient, sample_tumor):
        model = mod.TreatmentResponseModel(sample_patient, sample_tumor)
        assert model._classify_response(-100) == "CR"
        assert model._classify_response(-50) == "PR"
        assert model._classify_response(0) == "SD"
        assert model._classify_response(25) == "PD"


# ---------------------------------------------------------------------------
# TestTreatmentOptimizer
# ---------------------------------------------------------------------------


class TestTreatmentOptimizer:
    """Tests for TreatmentOptimizer class."""

    def test_init(self, mod, sample_patient, sample_tumor):
        opt = mod.TreatmentOptimizer(sample_patient, sample_tumor)
        assert opt.patient is sample_patient

    def test_compare_returns_list(self, mod, sample_patient, sample_tumor):
        opt = mod.TreatmentOptimizer(sample_patient, sample_tumor)
        protocols = [
            mod.TreatmentProtocol(
                modality=mod.TreatmentModality.CHEMOTHERAPY,
                name="Chemo",
                parameters={"drug": "generic", "cycle_days": 21},
                cycles=4,
            ),
            mod.TreatmentProtocol(
                modality=mod.TreatmentModality.RADIATION,
                name="Radiation",
                parameters={"total_dose_gy": 60, "fractions": 30},
            ),
        ]
        outcomes = opt.compare_treatments(protocols, baseline_volume=12.0)
        assert isinstance(outcomes, list)
        assert len(outcomes) == 2

    def test_generate_recommendation(self, mod, sample_patient, sample_tumor):
        opt = mod.TreatmentOptimizer(sample_patient, sample_tumor)
        protocols = [
            mod.TreatmentProtocol(
                modality=mod.TreatmentModality.CHEMOTHERAPY,
                name="Chemo",
                parameters={"drug": "generic", "cycle_days": 21},
                cycles=4,
            ),
        ]
        outcomes = opt.compare_treatments(protocols, baseline_volume=12.0)
        rec = opt.generate_recommendation(outcomes, optimization_target="efficacy")
        assert isinstance(rec, dict)
        assert "recommended_treatment" in rec
