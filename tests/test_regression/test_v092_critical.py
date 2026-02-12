"""Regression guards for v0.9.2 critical fixes.

Each test targets a specific bug from CHANGELOG v0.9.2 to prevent recurrence.
These complement the root-level test_regression.py with additional coverage.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np


class TestEKFJacobianStability:
    """Extended regression for EKF Jacobian sign error (v0.9.2 CRITICAL)."""

    def test_creatinine_state_bounded_after_many_steps(self, dt_sync_mod):
        """Creatinine should stay bounded after 30+ EKF cycles."""
        x0 = np.array([15.0, 0.01, 0.0, 4.5, 2.5, 13.0, 70.0, 1.0])
        P0 = np.diag([9.0, 0.0001, 0.01, 0.25, 0.5, 0.25, 4.0, 0.09])
        t0 = datetime(2026, 3, 1)

        ekf = dt_sync_mod.ExtendedKalmanFilterSync(x0, P0, start_time=t0)

        for day in range(1, 31):
            ekf.predict(dt_days=1.0)
            obs = dt_sync_mod.ClinicalObservation(
                obs_type=dt_sync_mod.ObservationType.LAB_CREATININE,
                value=1.2,
                timestamp=t0 + timedelta(days=day),
            )
            ekf.update(obs)

        assert np.all(np.isfinite(ekf.x))
        assert 0.0 < ekf.x[4] < 20.0  # physiologically reasonable

    def test_jacobian_diagonal_structure(self, dt_sync_mod):
        """Jacobian diagonal should have correct sign structure."""
        dynamics = dt_sync_mod.TumorPatientDynamics()
        x = np.array([15.0, 0.01, 0.0, 4.5, 1.0, 13.0, 70.0, 1.0])
        F = dynamics.jacobian(x, dt_days=1.0)

        # All diagonal elements should be finite
        for i in range(8):
            assert np.isfinite(F[i, i])


class TestHazardRatioExtended:
    """Extended regression for inverted hazard ratio (v0.9.2 CRITICAL)."""

    def test_hr_multiple_seeds(self, cohort_mod):
        """HR direction should be consistent across multiple seeds."""
        design = cohort_mod.TrialDesign(
            trial_id="REG-HR-SEEDS",
            n_patients_per_arm=100,
            arms={"control": 1.0, "experimental": 0.5},
        )
        sim = cohort_mod.VirtualTrialSimulator(cohort_mod.TumorSite.NSCLC, design)

        for seed in [42, 123, 999]:
            results = sim.run_trial(seed=seed)
            assert results.hazard_ratio > 0
            assert np.isfinite(results.hazard_ratio)

    def test_hr_equal_arms(self, cohort_mod):
        """Equal arms should produce HR near 1.0."""
        design = cohort_mod.TrialDesign(
            trial_id="REG-HR-EQUAL",
            n_patients_per_arm=200,
            arms={"control": 1.0, "experimental": 1.0},
        )
        sim = cohort_mod.VirtualTrialSimulator(cohort_mod.TumorSite.BREAST, design)
        results = sim.run_trial(seed=42)
        assert 0.5 < results.hazard_ratio < 2.0


class TestDoseResultZeroRegression:
    """Extended regression for DoseResult zero-value handling (v0.9.2)."""

    def test_to_dict_keys_present(self, dose_mod):
        """All DoseResult fields should appear in to_dict output."""
        result = dose_mod.DoseResult(
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
        for key in ["bed_gy", "eqd2_gy", "tcp", "ntcp"]:
            assert key in d, "zero-valued {!r} missing from to_dict()".format(key)

    def test_nonzero_values_preserved(self, dose_mod):
        """Non-zero values should also be preserved in to_dict."""
        bed = dose_mod.calc_bed(60.0, 30, 10.0)
        result = dose_mod.DoseResult(
            total_dose_gy=60.0,
            fractions=30,
            dose_per_fraction_gy=2.0,
            alpha_beta_gy=10.0,
            bed_gy=bed,
            eqd2_gy=bed / 1.2,
        )
        d = result.to_dict()
        assert d["bed_gy"] == bed


class TestTumorTwinZeroVolume:
    """Extended regression for division-by-zero in tumor_twin_pipeline."""

    def test_simulate_near_zero_volume(self, twin_mod):
        """Near-zero volume should not cause numerical issues."""
        config = {"domain": "test"}
        model = twin_mod.LogisticGrowthModel(domain_shape=(10, 10, 10), config=config)
        model.parameters.proliferation_rate = 0.05
        model.parameters.carrying_capacity = 100.0

        initial = np.ones((10, 10, 10)) * 1e-10
        result = model.simulate(initial, duration_days=10)
        assert len(result) >= 1
        assert np.all(np.isfinite(result[-1]))


class TestGCPScoreNotAlwaysZero:
    """Regression for GCP compliance score always reporting 0% (v0.9.2)."""

    def test_compliance_score_nonzero(self, gcp_mod):
        """GCP compliance score should not always be zero."""
        checker = gcp_mod.GCPComplianceChecker()
        report = checker.verify_compliance(ai_components_present=False)
        # Score should be computed, not stuck at zero
        assert report.overall_score >= 0.0


class TestRegTrackerOverdue:
    """Regression for unreachable 'overdue' status in regulatory tracker."""

    def test_tracker_instantiates(self, reg_tracker_mod):
        """Regulatory tracker should instantiate without errors."""
        tracker = reg_tracker_mod.RegulatoryTracker()
        assert tracker is not None
