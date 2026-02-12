"""Integration test: Digital twin -> treatment simulation workflow.

Validates the end-to-end flow from patient digital twin creation
through treatment simulation and response prediction.
"""

from __future__ import annotations

import numpy as np

from tests.conftest import load_module


class TestTwinToSimulationWorkflow:
    def test_create_twin_and_simulate_treatment(self):
        twin_mod = load_module(
            "tumor_twin_pipeline",
            "digital-twins/patient-modeling/tumor_twin_pipeline.py",
        )
        sim_mod = load_module(
            "treatment_simulator",
            "digital-twins/treatment-simulation/treatment_simulator.py",
        )

        # Create patient digital twin
        pipeline = twin_mod.TumorTwinPipeline(
            model_type=twin_mod.ModelType.LOGISTIC,
            solver=twin_mod.SolverType.CPU_SERIAL,
        )
        clinical = twin_mod.PatientClinicalData(
            patient_id="INT-001",
            age=55,
            sex="M",
            tumor_grade="III",
        )
        imaging = np.random.rand(10, 10, 10).astype(np.float32)
        tumor_mask = np.zeros((10, 10, 10), dtype=np.float32)
        tumor_mask[3:7, 3:7, 3:7] = 1.0

        twin = pipeline.create_twin(
            patient_id="INT-001",
            imaging_data={"t1": imaging},
            tumor_segmentation=tumor_mask,
            clinical_data=clinical,
        )
        assert twin.current_volume_cm3 > 0

        # Simulate treatment response
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        treatment = {"type": "radiation", "total_dose_gy": 60.0, "fractions": 30}
        response = simulator.predict_response(treatment, horizon_days=90)
        assert response.volumes is not None
        assert len(response.volumes) > 0

    def test_twin_predict_without_treatment(self):
        twin_mod = load_module(
            "tumor_twin_pipeline",
            "digital-twins/patient-modeling/tumor_twin_pipeline.py",
        )

        pipeline = twin_mod.TumorTwinPipeline(
            model_type=twin_mod.ModelType.LOGISTIC,
            solver=twin_mod.SolverType.CPU_SERIAL,
        )
        clinical = twin_mod.PatientClinicalData(
            patient_id="INT-002",
            age=60,
            sex="F",
            tumor_grade="II",
        )
        imaging = np.random.rand(10, 10, 10).astype(np.float32)
        tumor_mask = np.zeros((10, 10, 10), dtype=np.float32)
        tumor_mask[4:6, 4:6, 4:6] = 0.3

        twin = pipeline.create_twin(
            patient_id="INT-002",
            imaging_data={"t1": imaging},
            tumor_segmentation=tumor_mask,
            clinical_data=clinical,
        )
        prediction = twin.predict(horizon_days=30)
        assert prediction.volumes is not None
