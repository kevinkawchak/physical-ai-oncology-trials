"""Regression tests for critical bugs fixed in v0.9.1 and v0.9.2.

Each test targets a specific bug from CHANGELOG.md to prevent recurrence.
Tests are named after the component and bug type.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from tests.conftest import load_module


# ---------------------------------------------------------------------------
# v0.9.2 CRITICAL: EKF Jacobian sign error
# Fixed: 1.0 + rate*dt -> 1.0 - rate*dt in creatinine state
# ---------------------------------------------------------------------------


class TestEKFJacobianSign:
    """Regression for EKF Jacobian sign error in dt_synchronization.py:295."""

    def test_creatinine_jacobian_negative_diagonal(self):
        mod = load_module(
            "dt_synchronization",
            "digital-twins/examples-twins/01_realtime_dt_synchronization.py",
        )
        dynamics = mod.TumorPatientDynamics()
        x = np.array([15.0, 0.01, 0.0, 4.5, 1.0, 13.0, 70.0, 1.0])
        F = dynamics.jacobian(x, dt_days=1.0)
        # Creatinine clearance rate should reduce state -> diagonal <= 1
        creatinine_idx = 4
        assert F[creatinine_idx, creatinine_idx] <= 1.0

    def test_ekf_creatinine_does_not_diverge(self):
        """EKF should not diverge when assimilating creatinine observations."""
        mod = load_module(
            "dt_synchronization",
            "digital-twins/examples-twins/01_realtime_dt_synchronization.py",
        )
        x0 = np.array([15.0, 0.01, 0.0, 4.5, 2.0, 13.0, 70.0, 1.0])
        P0 = np.diag([9.0, 0.0001, 0.01, 0.25, 0.5, 0.25, 4.0, 0.09])
        t0 = datetime(2026, 3, 1)

        ekf = mod.ExtendedKalmanFilterSync(x0, P0, start_time=t0)

        # Assimilate repeated creatinine = 1.0 observations
        for day in range(1, 10):
            ekf.predict(dt_days=1.0)
            obs = mod.ClinicalObservation(
                obs_type=mod.ObservationType.LAB_CREATININE,
                value=1.0,
                timestamp=t0 + timedelta(days=day),
            )
            ekf.update(obs)

        # Should remain finite (not diverge)
        assert np.all(np.isfinite(ekf.x))
        # Creatinine should move toward 1.0, not away
        assert abs(ekf.x[4] - 1.0) < abs(x0[4] - 1.0)


# ---------------------------------------------------------------------------
# v0.9.2 CRITICAL: Inverted hazard ratio
# Fixed: control/experimental -> experimental/control
# ---------------------------------------------------------------------------


class TestHazardRatioDirection:
    """Regression for inverted HR in virtual_trial_cohort_dt.py:743."""

    def test_hr_convention(self):
        mod = load_module(
            "virtual_trial_cohort",
            "digital-twins/examples-twins/05_virtual_trial_cohort_dt.py",
        )
        # Experimental arm has lower hazard (0.5) → longer survival
        design = mod.TrialDesign(
            trial_id="REGRESSION-HR",
            n_patients_per_arm=100,
            arms={"control": 1.0, "experimental": 0.5},
        )
        simulator = mod.VirtualTrialSimulator(mod.TumorSite.NSCLC, design)
        results = simulator.run_trial(seed=42)

        # v0.9.2 fix: HR = median_experimental / median_control
        # When experimental has lower hazard → longer survival → HR > 1
        assert results.hazard_ratio > 1.0
        assert np.isfinite(results.hazard_ratio)


# ---------------------------------------------------------------------------
# v0.9.2 CRITICAL: DoseResult truthiness dropping zero values
# Fixed: if self.bed_gy -> if self.bed_gy is not None
# ---------------------------------------------------------------------------


class TestDoseResultZeroValues:
    """Regression for DoseResult.to_dict() dropping zero values."""

    def test_zero_bed_included(self):
        mod = load_module(
            "dose_calculator",
            "tools/dose-calculator/dose_calculator.py",
        )
        result = mod.DoseResult(
            total_dose_gy=0.0,
            fractions=1,
            dose_per_fraction_gy=0.0,
            alpha_beta_gy=10.0,
            bed_gy=0.0,
            eqd2_gy=0.0,
            tcp=0.0,
            ntcp=0.0,
        )
        d = result.to_dict()
        assert "bed_gy" in d
        assert d["bed_gy"] == 0.0
        assert "eqd2_gy" in d
        assert d["eqd2_gy"] == 0.0
        assert "tcp" in d
        assert d["tcp"] == 0.0
        assert "ntcp" in d
        assert d["ntcp"] == 0.0


# ---------------------------------------------------------------------------
# v0.9.2: Division by zero in tumor_twin_pipeline
# Fixed: guard for initial_condition summing to zero
# ---------------------------------------------------------------------------


class TestTumorTwinDivisionByZero:
    """Regression for division by zero in tumor_twin_pipeline.py."""

    def test_zero_initial_volume_no_crash(self):
        mod = load_module(
            "tumor_twin_pipeline",
            "digital-twins/patient-modeling/tumor_twin_pipeline.py",
        )
        config = {"domain": "test"}
        model = mod.LogisticGrowthModel(domain_shape=(10, 10, 10), config=config)
        model.parameters.proliferation_rate = 0.05
        model.parameters.carrying_capacity = 100.0
        # Post-resection: volume near zero
        initial = np.zeros((10, 10, 10))
        result = model.simulate(initial, duration_days=30)
        assert len(result) >= 1
        # All values should be finite
        assert np.all(np.isfinite(result[-1]))


# ---------------------------------------------------------------------------
# v0.9.2: Floating-point equality in treatment_simulator surgery check
# ---------------------------------------------------------------------------


class TestTreatmentSimulatorInit:
    """Regression: TreatmentSimulator requires a patient_twin argument."""

    def test_simulator_with_mock_twin(self):
        mod = load_module(
            "treatment_simulator",
            "digital-twins/treatment-simulation/treatment_simulator.py",
        )

        class MockTwin:
            patient_id = "TEST-REG-001"
            current_volume_cm3 = 10.0
            proliferation_rate = 0.05

            class clinical_data:
                tumor_grade = "III"

            class model:
                parameters = type(
                    "P",
                    (),
                    {
                        "proliferation_rate": 0.05,
                        "treatment_sensitivity": 0.5,
                    },
                )()

        sim = mod.TreatmentSimulator(patient_twin=MockTwin())
        assert sim.baseline_volume == 10.0


# ---------------------------------------------------------------------------
# v0.9.1: Weak default salt in deidentification
# Fixed: random salt via os.urandom
# ---------------------------------------------------------------------------


class TestDeidentificationSalt:
    """Regression for weak default salt in deidentification_pipeline.py."""

    def test_default_salt_not_hardcoded(self):
        mod = load_module(
            "deidentification_pipeline",
            "privacy/de-identification/deidentification_pipeline.py",
        )
        config = mod.DeidentificationConfig()
        # After v0.9.1 fix, default salt should not be "default_salt"
        assert config.hash_salt != "default_salt"
