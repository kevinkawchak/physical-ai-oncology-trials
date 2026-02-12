"""Integration test: End-to-end simulated clinical trial workflow.

Validates the complete flow from patient creation through treatment
simulation, safety monitoring, and regulatory compliance.
"""

from __future__ import annotations

import numpy as np

from tests.conftest import load_module


class TestEndToEndTrialWorkflow:
    def test_full_trial_simulation(self):
        # 1. Create patient digital twin
        twin_mod = load_module(
            "tumor_twin_pipeline",
            "digital-twins/patient-modeling/tumor_twin_pipeline.py",
        )
        pipeline = twin_mod.TumorTwinPipeline(
            model_type=twin_mod.ModelType.LOGISTIC,
            solver=twin_mod.SolverType.CPU_SERIAL,
        )
        clinical = twin_mod.PatientClinicalData(
            patient_id="E2E-001",
            age=58,
            sex="M",
            tumor_grade="III",
        )
        imaging = np.random.rand(10, 10, 10).astype(np.float32)
        tumor_mask = np.zeros((10, 10, 10), dtype=np.float32)
        tumor_mask[3:7, 3:7, 3:7] = 1.0
        twin = pipeline.create_twin(
            patient_id="E2E-001",
            imaging_data={"t1": imaging},
            tumor_segmentation=tumor_mask,
            clinical_data=clinical,
        )
        assert twin.current_volume_cm3 > 0

        # 2. Simulate treatment
        sim_mod = load_module(
            "treatment_simulator",
            "digital-twins/treatment-simulation/treatment_simulator.py",
        )
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        treatment = {"type": "radiation", "total_dose_gy": 60.0, "fractions": 30}
        response = simulator.predict_response(treatment, horizon_days=90)
        assert response.response_category is not None

        # 3. Calculate dose metrics
        dose_mod = load_module(
            "dose_calculator",
            "tools/dose-calculator/dose_calculator.py",
        )
        bed = dose_mod.calc_bed(60.0, 30, 10.0)
        assert bed > 0

        # 4. De-identify for reporting
        deid_mod = load_module(
            "deidentification_pipeline",
            "privacy/de-identification/deidentification_pipeline.py",
        )
        config = deid_mod.DeidentificationConfig()
        transformer = deid_mod.SafeHarborTransformer(config)
        text = "Patient E2E-001 completed radiation therapy."
        transformed, _ = transformer.transform_text(text)
        assert isinstance(transformed, str)

    def test_multi_arm_virtual_trial(self):
        cohort_mod = load_module(
            "virtual_trial_cohort",
            "digital-twins/examples-twins/05_virtual_trial_cohort_dt.py",
        )
        design = cohort_mod.TrialDesign(
            trial_id="E2E-VTC",
            n_patients_per_arm=50,
            arms={
                "control": 1.0,
                "experimental_low": 0.7,
                "experimental_high": 0.5,
            },
            randomization_ratio=[1, 1, 1],
        )
        sim = cohort_mod.VirtualTrialSimulator(
            cohort_mod.TumorSite.BREAST,
            design,
        )
        results = sim.run_trial(seed=42)
        assert results.hazard_ratio > 0
        assert np.isfinite(results.hazard_ratio)
