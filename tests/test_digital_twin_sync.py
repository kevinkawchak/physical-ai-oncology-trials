"""Tests for digital-twins/examples-twins/01_realtime_dt_synchronization.py

Covers TumorPatientDynamics, ObservationModel, ExtendedKalmanFilterSync,
ParticleFilterSync, ClinicalAnomalyDetector, and DigitalTwinSynchronizer.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest


# ── TumorPatientDynamics ──────────────────────────────────────────────────


class TestTumorPatientDynamics:
    def test_state_transition_preserves_dimension(self, dt_sync_mod, baseline_patient_state):
        dynamics = dt_sync_mod.TumorPatientDynamics()
        x_next = dynamics.state_transition(baseline_patient_state, dt_days=1.0)
        assert x_next.shape == (dt_sync_mod.STATE_DIM,)

    def test_tumor_grows_without_treatment(self, dt_sync_mod, baseline_patient_state):
        dynamics = dt_sync_mod.TumorPatientDynamics()
        x0 = baseline_patient_state.copy()
        x_next = dynamics.state_transition(x0, dt_days=7.0, treatment_active=False)
        assert x_next[0] > x0[0]  # volume should increase

    def test_drug_effect_decays(self, dt_sync_mod, baseline_patient_state):
        dynamics = dt_sync_mod.TumorPatientDynamics()
        x0 = baseline_patient_state.copy()
        x0[2] = 1.0  # peak drug effect
        x_next = dynamics.state_transition(x0, dt_days=7.0, treatment_active=False)
        assert x_next[2] < 1.0  # drug should decay

    def test_jacobian_shape(self, dt_sync_mod, baseline_patient_state):
        dynamics = dt_sync_mod.TumorPatientDynamics()
        F = dynamics.jacobian(baseline_patient_state, dt_days=1.0)
        assert F.shape == (dt_sync_mod.STATE_DIM, dt_sync_mod.STATE_DIM)

    def test_jacobian_diagonal_dominated(self, dt_sync_mod, baseline_patient_state):
        dynamics = dt_sync_mod.TumorPatientDynamics()
        F = dynamics.jacobian(baseline_patient_state, dt_days=1.0)
        for i in range(dt_sync_mod.STATE_DIM):
            assert F[i, i] != 0.0

    def test_volume_floor(self, dt_sync_mod):
        dynamics = dt_sync_mod.TumorPatientDynamics()
        x = np.array([0.001, -0.5, 0.0, 4.5, 1.0, 13.0, 70.0, 1.0])
        x_next = dynamics.state_transition(x, dt_days=30.0)
        assert x_next[0] >= 0.01  # minimum detectable volume


# ── ObservationModel ──────────────────────────────────────────────────────


class TestObservationModel:
    def test_tumor_marker_proportional_to_volume(self, dt_sync_mod, baseline_patient_state):
        obs = dt_sync_mod.ObservationModel()
        marker = obs.observe(baseline_patient_state, dt_sync_mod.ObservationType.TUMOR_MARKER)
        expected = obs.MARKER_SENSITIVITY * baseline_patient_state[0]
        assert np.isclose(marker, expected)

    def test_imaging_volume_direct(self, dt_sync_mod, baseline_patient_state):
        obs = dt_sync_mod.ObservationModel()
        vol = obs.observe(baseline_patient_state, dt_sync_mod.ObservationType.IMAGING_VOLUME)
        assert np.isclose(vol, baseline_patient_state[0])

    def test_anc_direct_read(self, dt_sync_mod, baseline_patient_state):
        obs = dt_sync_mod.ObservationModel()
        anc = obs.observe(baseline_patient_state, dt_sync_mod.ObservationType.LAB_NEUTROPHILS)
        assert np.isclose(anc, baseline_patient_state[3])

    def test_observation_jacobian_shape(self, dt_sync_mod, baseline_patient_state):
        obs = dt_sync_mod.ObservationModel()
        H = obs.observation_jacobian(baseline_patient_state, dt_sync_mod.ObservationType.TUMOR_MARKER)
        assert H.shape == (dt_sync_mod.STATE_DIM,)

    def test_noise_variance_positive(self, dt_sync_mod):
        obs = dt_sync_mod.ObservationModel()
        for obs_type in dt_sync_mod.ObservationType:
            var = obs.noise_variance(obs_type)
            assert var > 0


# ── ExtendedKalmanFilterSync ──────────────────────────────────────────────


class TestEKFSync:
    def test_predict_advances_time(self, dt_sync_mod, baseline_patient_state, baseline_covariance, reference_timestamp):
        ekf = dt_sync_mod.ExtendedKalmanFilterSync(
            baseline_patient_state,
            baseline_covariance,
            start_time=reference_timestamp,
        )
        state = ekf.predict(dt_days=1.0)
        assert state.timestamp == reference_timestamp + timedelta(days=1.0)

    def test_predict_increases_uncertainty(self, dt_sync_mod, baseline_patient_state, baseline_covariance):
        ekf = dt_sync_mod.ExtendedKalmanFilterSync(baseline_patient_state, baseline_covariance)
        P_before = ekf.P.copy()
        ekf.predict(dt_days=1.0)
        # At least some diagonal entries should increase
        assert np.any(np.diag(ekf.P) >= np.diag(P_before))

    def test_update_reduces_uncertainty(
        self, dt_sync_mod, baseline_patient_state, baseline_covariance, reference_timestamp
    ):
        ekf = dt_sync_mod.ExtendedKalmanFilterSync(
            baseline_patient_state,
            baseline_covariance,
            start_time=reference_timestamp,
        )
        P_before = ekf.P.copy()

        obs = dt_sync_mod.ClinicalObservation(
            obs_type=dt_sync_mod.ObservationType.LAB_NEUTROPHILS,
            value=4.5,
            timestamp=reference_timestamp + timedelta(hours=1),
        )
        ekf.update(obs)
        # ANC variance (index 3) should decrease after direct observation
        assert ekf.P[3, 3] <= P_before[3, 3]

    def test_audit_trail_populated(self, dt_sync_mod, baseline_patient_state, baseline_covariance, reference_timestamp):
        ekf = dt_sync_mod.ExtendedKalmanFilterSync(
            baseline_patient_state,
            baseline_covariance,
            start_time=reference_timestamp,
        )
        obs = dt_sync_mod.ClinicalObservation(
            obs_type=dt_sync_mod.ObservationType.LAB_CREATININE,
            value=1.0,
            timestamp=reference_timestamp + timedelta(hours=1),
        )
        ekf.update(obs)
        trail = ekf.get_audit_trail()
        assert len(trail) >= 1

    def test_treatment_registration(
        self, dt_sync_mod, baseline_patient_state, baseline_covariance, reference_timestamp
    ):
        ekf = dt_sync_mod.ExtendedKalmanFilterSync(
            baseline_patient_state,
            baseline_covariance,
            start_time=reference_timestamp,
        )
        event = dt_sync_mod.TreatmentEvent(
            timestamp=reference_timestamp,
            drug="FOLFOX",
            dose_mg_m2=85,
            cycle=1,
        )
        ekf.register_treatment(event)
        assert ekf.x[2] == 1.0  # drug effect at peak
        assert ekf.treatment_active is True


# ── ParticleFilterSync ────────────────────────────────────────────────────


class TestParticleFilterSync:
    def test_particle_count(self, dt_sync_mod, baseline_patient_state):
        pf = dt_sync_mod.ParticleFilterSync(n_particles=100, initial_state=baseline_patient_state)
        assert pf.particles.shape == (100, dt_sync_mod.STATE_DIM)

    def test_predict_does_not_crash(self, dt_sync_mod, baseline_patient_state):
        pf = dt_sync_mod.ParticleFilterSync(n_particles=50, initial_state=baseline_patient_state)
        pf.predict(dt_days=1.0)
        state = pf.get_state()
        assert state.mean.shape == (dt_sync_mod.STATE_DIM,)

    def test_ess_starts_at_n(self, dt_sync_mod, baseline_patient_state):
        n = 100
        pf = dt_sync_mod.ParticleFilterSync(n_particles=n, initial_state=baseline_patient_state)
        ess = pf.effective_sample_size()
        assert np.isclose(ess, n, rtol=0.01)

    def test_response_probability_bounded(self, dt_sync_mod, baseline_patient_state):
        pf = dt_sync_mod.ParticleFilterSync(n_particles=50, initial_state=baseline_patient_state)
        prob = pf.get_response_probability()
        assert 0.0 <= prob <= 1.0

    def test_map_estimate_shape(self, dt_sync_mod, baseline_patient_state):
        pf = dt_sync_mod.ParticleFilterSync(n_particles=50, initial_state=baseline_patient_state)
        m = pf.get_map_estimate()
        assert m.shape == (dt_sync_mod.STATE_DIM,)


# ── ClinicalAnomalyDetector ──────────────────────────────────────────────


class TestClinicalAnomalyDetector:
    def test_no_alert_for_normal_innovations(self, dt_sync_mod, reference_timestamp):
        detector = dt_sync_mod.ClinicalAnomalyDetector()
        for i in range(10):
            result = detector.process_innovation(
                dt_sync_mod.ObservationType.LAB_NEUTROPHILS,
                innovation=0.1,
                expected_std=1.0,
                timestamp=reference_timestamp + timedelta(days=i),
            )
        assert len(detector.alerts) == 0

    def test_persistent_elevation_triggers_alert(self, dt_sync_mod, reference_timestamp):
        detector = dt_sync_mod.ClinicalAnomalyDetector(cusum_threshold=3.0)
        for i in range(20):
            detector.process_innovation(
                dt_sync_mod.ObservationType.LAB_CREATININE,
                innovation=2.0,  # consistently high
                expected_std=1.0,
                timestamp=reference_timestamp + timedelta(days=i),
            )
        assert len(detector.alerts) >= 1
        assert detector.alerts[0]["type"] == "persistent_elevation"


# ── DigitalTwinSynchronizer ───────────────────────────────────────────────


class TestDigitalTwinSynchronizer:
    def test_from_baseline_creates_ekf(self, dt_sync_mod):
        sync = dt_sync_mod.DigitalTwinSynchronizer.from_baseline(
            patient_id="TEST-001",
            tumor_volume_cm3=12.0,
            filter_type="ekf",
        )
        assert sync.patient_id == "TEST-001"

    def test_from_baseline_creates_particle(self, dt_sync_mod):
        sync = dt_sync_mod.DigitalTwinSynchronizer.from_baseline(
            patient_id="TEST-002",
            tumor_volume_cm3=12.0,
            filter_type="particle",
        )
        assert sync.patient_id == "TEST-002"

    def test_invalid_filter_raises(self, dt_sync_mod):
        with pytest.raises(ValueError, match="Unknown filter type"):
            dt_sync_mod.DigitalTwinSynchronizer(
                patient_id="X",
                filter_type="invalid",
            )

    def test_process_observation(self, dt_sync_mod, reference_timestamp):
        sync = dt_sync_mod.DigitalTwinSynchronizer.from_baseline(
            patient_id="TEST-003",
            tumor_volume_cm3=15.0,
        )
        obs = dt_sync_mod.ClinicalObservation(
            obs_type=dt_sync_mod.ObservationType.LAB_NEUTROPHILS,
            value=3.5,
            timestamp=reference_timestamp + timedelta(days=3),
        )
        state = sync.process_observation(obs)
        assert state.mean.shape == (dt_sync_mod.STATE_DIM,)

    def test_state_summary_keys(self, dt_sync_mod):
        sync = dt_sync_mod.DigitalTwinSynchronizer.from_baseline(
            patient_id="TEST-004",
            tumor_volume_cm3=10.0,
        )
        summary = sync.get_state_summary()
        assert "patient_id" in summary
        assert "Volume" in summary
        assert "ANC" in summary
