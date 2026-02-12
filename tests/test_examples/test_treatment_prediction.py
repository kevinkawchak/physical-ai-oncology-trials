"""Tests for examples/05_treatment_response_prediction.py

Covers growth models, treatment response, and clinical decision support.
"""

from __future__ import annotations

import numpy as np


class TestEnums:
    def test_tumor_types(self, treatment_prediction_mod):
        tt = treatment_prediction_mod.TumorType
        assert tt.NSCLC.value == "non_small_cell_lung_cancer"
        assert tt.BREAST_TNBC.value == "triple_negative_breast_cancer"

    def test_treatment_modalities(self, treatment_prediction_mod):
        tm = treatment_prediction_mod.TreatmentModality
        assert tm.CHEMOTHERAPY.value == "chemotherapy"
        assert tm.RADIATION.value == "radiation"


class TestPatientProfile:
    def test_creation(self, treatment_prediction_mod):
        profile = treatment_prediction_mod.PatientProfile(
            patient_id="TP-001",
            age=60,
            sex="F",
            tumor_type=treatment_prediction_mod.TumorType.NSCLC,
            stage="IIIB",
            tumor_volume_cm3=15.0,
            performance_status=1,
        )
        assert profile.patient_id == "TP-001"


class TestGrowthModels:
    def test_exponential_growth(self, treatment_prediction_mod):
        characteristics = treatment_prediction_mod.TumorCharacteristics(
            proliferation_rate=0.05,
            death_rate=0.01,
        )
        model = treatment_prediction_mod.ExponentialGrowthModel(characteristics)
        trajectory = model.simulate(initial_volume=10.0, duration_days=30)
        assert len(trajectory) > 0
        assert trajectory[-1] > 10.0

    def test_gompertz_growth(self, treatment_prediction_mod):
        characteristics = treatment_prediction_mod.TumorCharacteristics(
            proliferation_rate=0.05,
            death_rate=0.01,
        )
        model = treatment_prediction_mod.GompertzGrowthModel(
            characteristics=characteristics,
            carrying_capacity=100.0,
        )
        trajectory = model.simulate(initial_volume=10.0, duration_days=30)
        assert len(trajectory) > 0
        assert np.all(np.isfinite(trajectory))


class TestTreatmentResponseModel:
    def test_creation(self, treatment_prediction_mod):
        patient = treatment_prediction_mod.PatientProfile(
            patient_id="TP-002",
            age=55,
            sex="M",
            tumor_type=treatment_prediction_mod.TumorType.NSCLC,
            stage="IIIB",
            tumor_volume_cm3=15.0,
            performance_status=0,
        )
        tumor = treatment_prediction_mod.TumorCharacteristics()
        model = treatment_prediction_mod.TreatmentResponseModel(patient=patient, tumor=tumor)
        assert model is not None

    def test_predict_response(self, treatment_prediction_mod):
        patient = treatment_prediction_mod.PatientProfile(
            patient_id="TP-002",
            age=55,
            sex="M",
            tumor_type=treatment_prediction_mod.TumorType.NSCLC,
            stage="IIIB",
            tumor_volume_cm3=15.0,
            performance_status=0,
        )
        tumor = treatment_prediction_mod.TumorCharacteristics(
            radiation_sensitivity=0.35,
        )
        model = treatment_prediction_mod.TreatmentResponseModel(patient=patient, tumor=tumor)
        treatment = treatment_prediction_mod.TreatmentProtocol(
            modality=treatment_prediction_mod.TreatmentModality.RADIATION,
            name="Test Radiation",
            parameters={"total_dose_gy": 60.0, "fractions": 30},
        )
        outcome = model.predict_response(
            protocol=treatment,
            baseline_volume=15.0,
        )
        assert isinstance(outcome, treatment_prediction_mod.TreatmentOutcome)
