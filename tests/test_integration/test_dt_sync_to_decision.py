"""Integration test: Digital twin sync -> agentic decision workflow.

Validates the flow from DT synchronization through compliance checking.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from tests.conftest import load_module


class TestDTSyncToDecisionWorkflow:
    def test_ekf_sync_to_compliance_check(self):
        dt_mod = load_module(
            "dt_synchronization",
            "digital-twins/examples-twins/01_realtime_dt_synchronization.py",
        )
        rag_mod = load_module(
            "protocol_rag",
            "agentic-ai/examples-agentic-ai/06_protocol_rag_compliance_agent.py",
        )

        # Run EKF synchronization
        x0 = np.array([15.0, 0.01, 0.0, 4.5, 1.0, 13.0, 70.0, 1.0])
        P0 = np.diag([9.0, 0.0001, 0.01, 0.25, 0.01, 0.25, 4.0, 0.09])
        t0 = datetime(2026, 3, 1)

        ekf = dt_mod.ExtendedKalmanFilterSync(x0, P0, start_time=t0)
        ekf.predict(dt_days=1.0)

        obs = dt_mod.ClinicalObservation(
            obs_type=dt_mod.ObservationType.LAB_NEUTROPHILS,
            value=3.5,
            timestamp=t0 + timedelta(days=1),
        )
        ekf.update(obs)
        assert np.all(np.isfinite(ekf.x))

        # Check compliance of treatment decision
        agent = rag_mod.ProtocolRAGComplianceAgent()
        report = agent.check_compliance(
            action_description="Adjust chemotherapy dose based on ANC level",
            context={
                "anc_level": float(ekf.x[3]),
                "patient_id": "INT-DT-001",
            },
        )
        assert report.overall_status is not None

    def test_virtual_trial_to_regulatory(self):
        cohort_mod = load_module(
            "virtual_trial_cohort",
            "digital-twins/examples-twins/05_virtual_trial_cohort_dt.py",
        )

        design = cohort_mod.TrialDesign(
            trial_id="INT-VTC-001",
            n_patients_per_arm=30,
            arms={"control": 1.0, "experimental": 0.6},
        )
        sim = cohort_mod.VirtualTrialSimulator(
            cohort_mod.TumorSite.NSCLC,
            design,
        )
        results = sim.run_trial(seed=42)
        assert results.hazard_ratio > 0
        assert np.isfinite(results.hazard_ratio)
