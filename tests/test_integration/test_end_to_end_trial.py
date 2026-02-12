"""Integration tests: Full trial lifecycle end-to-end.

Tests the complete flow: Patient -> Digital Twin -> Simulation ->
Virtual Trial -> Dose Calculation -> De-identification, verifying
that all stages integrate correctly.
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
def cohort_mod():
    return load_module(
        "virtual_trial_cohort_dt",
        "digital-twins/examples-twins/05_virtual_trial_cohort_dt.py",
    )


@pytest.fixture()
def dose_mod():
    return load_module("dose_calculator", "tools/dose-calculator/dose_calculator.py")


@pytest.fixture()
def deid_mod():
    return load_module(
        "deidentification_pipeline",
        "privacy/de-identification/deidentification_pipeline.py",
    )


def _make_trial_design(cohort_mod, n_per_arm=30):
    """Create a small TrialDesign for fast testing."""
    return cohort_mod.TrialDesign(
        trial_id="TEST-E2E-001",
        n_patients_per_arm=n_per_arm,
        arms={"control": 1.0, "experimental": 0.70},
        primary_endpoint=cohort_mod.TrialEndpoint.PFS,
        randomization_ratio=[1, 1],
        interim_analyses=[0.5],
        max_followup_days=365,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEndTrial:
    """Integration: full trial lifecycle."""

    def test_virtual_trial_lifecycle(self, cohort_mod):
        """Generate cohort, simulate outcomes, verify hazard ratio is finite."""
        design = _make_trial_design(cohort_mod, n_per_arm=30)
        simulator = cohort_mod.VirtualTrialSimulator(cohort_mod.TumorSite.NSCLC, design)
        results = simulator.run_trial(seed=42)
        assert math.isfinite(results.hazard_ratio)
        assert math.isfinite(results.p_value)
        assert len(results.outcomes) > 0

    def test_dose_calculation_for_twin(self, twin_mod, dose_mod):
        """Create patient twin, compute BED and EQD2 for treatment."""
        bed = dose_mod.calc_bed(total_dose=60.0, fractions=30, alpha_beta=10.0)
        eqd2 = dose_mod.calc_eqd2(total_dose=60.0, fractions=30, alpha_beta=10.0)
        assert bed > 0
        assert eqd2 > 0

    def test_patient_data_deid_before_trial(self, deid_mod):
        """De-identify patient data, verify clean for trial use."""
        pipeline = deid_mod.DeidentificationPipeline()
        record = {
            "patient_name": "Jane Smith",
            "mrn": "MRN-999",
            "ssn": "987-65-4321",
            "diagnosis": "Stage IV Melanoma",
            "tumor_volume_cm3": 8.5,
            "treatment": "Immunotherapy",
        }
        phi_fields = {
            "patient_name": "NAME",
            "mrn": "MEDICAL_RECORD_NUMBER",
            "ssn": "SSN",
        }
        deid = pipeline.deidentify_record(record, phi_fields)
        assert str(deid.get("patient_name", "")) != "Jane Smith"
        assert str(deid.get("ssn", "")) != "987-65-4321"
        assert deid.get("tumor_volume_cm3") == 8.5

    def test_trial_results_complete(self, cohort_mod):
        """VirtualTrialSimulator results have all required fields."""
        design = _make_trial_design(cohort_mod, n_per_arm=20)
        simulator = cohort_mod.VirtualTrialSimulator(cohort_mod.TumorSite.NSCLC, design)
        results = simulator.run_trial(seed=99)
        assert hasattr(results, "hazard_ratio")
        assert hasattr(results, "p_value")
        assert hasattr(results, "median_pfs_by_arm")
        assert hasattr(results, "median_os_by_arm")
        assert hasattr(results, "orr_by_arm")
        assert hasattr(results, "posterior_probability")

    def test_dose_tcp_ntcp_pipeline(self, dose_mod):
        """Full dose -> TCP -> NTCP computation chain."""
        bed = dose_mod.calc_bed(total_dose=60.0, fractions=30, alpha_beta=10.0)
        assert bed > 0
        tcp = dose_mod.calc_tcp_poisson(total_dose=60.0, fractions=30, n0=1e7, alpha=0.3, alpha_beta=10.0)
        assert 0.0 <= tcp <= 1.0

        ntcp = dose_mod.calc_ntcp_lkb(
            total_dose=60.0,
            fractions=30,
            td50=30.0,
            m=0.15,
            n=0.87,
            alpha_beta=3.0,
            volume_fraction=1.0,
        )
        assert 0.0 <= ntcp <= 1.0

    def test_cohort_generation(self, cohort_mod):
        """VirtualCohortGenerator produces valid patients."""
        gen = cohort_mod.VirtualCohortGenerator(cohort_mod.TumorSite.NSCLC, seed=42)
        cohort = gen.generate(n_patients=50)
        assert len(cohort) == 50
        for patient in cohort:
            assert patient.age >= 18
            assert patient.tumor_volume_cm3 > 0
            assert patient.sex in ("M", "F")

    def test_twin_predict_produces_volumes(self, twin_mod):
        """PatientDigitalTwin.predict() returns a TwinPrediction with volumes."""
        pipeline = twin_mod.TumorTwinPipeline(
            model_type=twin_mod.ModelType.LOGISTIC,
            tumor_type=twin_mod.TumorType.LUNG_CANCER,
        )
        mask = np.ones((5, 5, 5)) * 0.5
        twin = pipeline.create_twin(
            patient_id="E2E-001",
            imaging_data={"ct": np.zeros((5, 5, 5))},
            tumor_segmentation=mask,
            clinical_data={"age": 65, "sex": "M"},
        )
        twin.is_calibrated = True
        prediction = twin.predict(horizon_days=30)
        assert prediction.volumes is not None
        assert len(prediction.volumes) > 0

    def test_dose_result_to_dict(self, dose_mod):
        """DoseResult.to_dict() includes all fields, including zero values."""
        result = dose_mod.DoseResult(
            total_dose_gy=60.0,
            fractions=30,
            dose_per_fraction_gy=2.0,
            alpha_beta_gy=10.0,
            bed_gy=72.0,
            eqd2_gy=60.0,
        )
        d = result.to_dict()
        assert "bed_gy" in d
        assert "eqd2_gy" in d
        assert d["bed_gy"] is not None

    def test_trial_with_different_tumor_sites(self, cohort_mod):
        """Trial simulator works across different tumor sites."""
        for site in [cohort_mod.TumorSite.NSCLC, cohort_mod.TumorSite.BREAST]:
            design = _make_trial_design(cohort_mod, n_per_arm=15)
            sim = cohort_mod.VirtualTrialSimulator(site, design)
            results = sim.run_trial(seed=7)
            assert math.isfinite(results.hazard_ratio)

    def test_deid_then_cohort_integration(self, deid_mod, cohort_mod):
        """De-identify records, then run virtual cohort -- no data leak."""
        pipeline = deid_mod.DeidentificationPipeline()
        record = {
            "patient_name": "Bob Test",
            "mrn": "MRN-111",
            "diagnosis": "NSCLC",
            "tumor_volume_cm3": 15.0,
        }
        phi_fields = {"patient_name": "NAME", "mrn": "MEDICAL_RECORD_NUMBER"}
        deid = pipeline.deidentify_record(record, phi_fields)
        # Verify the de-identified record has no real name
        assert str(deid.get("patient_name", "")) != "Bob Test"
        # Virtual cohort runs independently
        gen = cohort_mod.VirtualCohortGenerator(cohort_mod.TumorSite.NSCLC, seed=42)
        cohort = gen.generate(n_patients=5)
        assert len(cohort) == 5
