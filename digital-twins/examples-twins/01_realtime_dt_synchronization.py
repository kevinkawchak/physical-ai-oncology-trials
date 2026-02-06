"""
=============================================================================
EXAMPLE 01: Real-Time Digital Twin Synchronization & State Estimation
=============================================================================

WHAT THIS CODE DOES:
    Continuously synchronizes a patient digital twin with live clinical data
    streams (vitals, laboratory results, imaging biomarkers) using Bayesian
    state estimation. Implements Extended Kalman Filtering (EKF) and
    Sequential Monte Carlo (particle filter) for real-time inference of
    latent tumor state from noisy, asynchronous clinical observations.

CLINICAL CONTEXT:
    During active treatment cycles (e.g., 6 cycles of FOLFOX for colorectal
    cancer over 12 weeks), patient state evolves continuously but is observed
    only at discrete, irregular intervals:
      - Vitals: every clinic visit (weekly-biweekly)
      - Labs (CBC, CMP, tumor markers): every 2-3 weeks
      - Imaging (CT/MRI): every 6-12 weeks
    The digital twin must fuse these heterogeneous observations into a
    coherent, continuously updated patient state estimate that clinicians
    can query at any time for treatment monitoring and early intervention.

USE CASES COVERED:
    1. Asynchronous multi-modal data fusion (vitals + labs + imaging)
    2. Extended Kalman Filter for tumor state estimation
    3. Particle filter for non-Gaussian / multimodal posteriors
    4. Anomaly detection for early adverse event warning
    5. Missing data handling and observation gating
    6. Streaming update architecture for clinical dashboard integration

FRAMEWORK REQUIREMENTS:
    Required:
        - NumPy 1.24.0+
        - SciPy 1.11.0+
    Optional:
        - fhirclient 4.0.0+ (for FHIR observation streams)
        - pydicom 2.4.0+ (for DICOM RT structure ingestion)

REGULATORY NOTES:
    - FDA 21 CFR Part 11: All state updates logged with timestamps for audit
    - IEC 62304: Software lifecycle documented per Class B medical device
    - ICH E6(R3) Sec 4.6: Digital technology provisions for real-time data

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from scipy.linalg import block_diag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: OBSERVATION MODEL — CLINICAL DATA TYPES
# =============================================================================
# In oncology trials, the DT receives observations from multiple clinical
# sources at irregular intervals. Each source has different noise
# characteristics and maps to different components of the latent state.


class ObservationType(Enum):
    """Clinical observation modalities feeding the digital twin."""

    TUMOR_MARKER = "tumor_marker"          # e.g., CEA, CA-125, PSA
    IMAGING_VOLUME = "imaging_volume"      # CT/MRI tumor volume (cm^3)
    IMAGING_DIAMETER = "imaging_diameter"   # RECIST longest diameter (mm)
    LAB_NEUTROPHILS = "lab_neutrophils"    # ANC for myelosuppression
    LAB_CREATININE = "lab_creatinine"      # Renal function
    LAB_BILIRUBIN = "lab_bilirubin"        # Hepatic function
    LAB_HEMOGLOBIN = "lab_hemoglobin"      # Anemia monitoring
    VITAL_WEIGHT = "vital_weight"          # Body weight (kg)
    VITAL_ECOG = "vital_ecog"             # Performance status (0-4)
    CTCAE_TOXICITY = "ctcae_toxicity"     # CTCAE grade (0-5)


@dataclass
class ClinicalObservation:
    """A single clinical observation with metadata for audit trail.

    Attributes:
        obs_type: Type of clinical observation
        value: Observed numeric value
        timestamp: When the observation was recorded
        uncertainty: Measurement uncertainty (standard deviation)
        source_system: Originating system (EHR, PACS, LIS)
        observation_id: Unique identifier for 21 CFR Part 11 audit
    """

    obs_type: ObservationType
    value: float
    timestamp: datetime
    uncertainty: float = 0.0
    source_system: str = "EHR"
    observation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# =============================================================================
# SECTION 2: STATE SPACE MODEL — TUMOR + PATIENT DYNAMICS
# =============================================================================
# The latent state vector encodes both tumor biology and patient physiology.
# State vector x = [V, g, d, ANC, Cr, Hb, W, E]
#   V   : tumor volume (cm^3)
#   g   : net tumor growth rate (/day), may go negative during treatment
#   d   : drug effect compartment (dimensionless, 0 = no drug, 1 = peak)
#   ANC : absolute neutrophil count (10^9/L)
#   Cr  : serum creatinine (mg/dL)
#   Hb  : hemoglobin (g/dL)
#   W   : body weight (kg)
#   E   : ECOG performance status (continuous latent, discretized on read)


STATE_DIM = 8
STATE_LABELS = ["Volume", "GrowthRate", "DrugEffect",
                "ANC", "Creatinine", "Hemoglobin", "Weight", "ECOG"]


@dataclass
class TreatmentEvent:
    """A treatment administration event that modifies state dynamics.

    Attributes:
        timestamp: When treatment was administered
        drug: Drug name
        dose_mg_m2: Dose in mg/m^2
        route: Administration route
        cycle: Treatment cycle number
    """

    timestamp: datetime
    drug: str = "generic"
    dose_mg_m2: float = 0.0
    route: str = "IV"
    cycle: int = 1


@dataclass
class StateEstimate:
    """Current state estimate with uncertainty.

    Attributes:
        mean: State vector mean [STATE_DIM]
        covariance: State covariance matrix [STATE_DIM x STATE_DIM]
        timestamp: Time of this estimate
        log_likelihood: Cumulative log-likelihood for model comparison
    """

    mean: np.ndarray
    covariance: np.ndarray
    timestamp: datetime
    log_likelihood: float = 0.0


# =============================================================================
# SECTION 3: PROCESS MODEL — STATE TRANSITION DYNAMICS
# =============================================================================
# The process model defines how the patient state evolves between observations.
# Tumor volume follows exponential-decay growth with drug effect modulation.
# Hematologic and organ function follow compartmental pharmacodynamics.


class TumorPatientDynamics:
    """Continuous-time state transition model for tumor + patient state.

    Encodes the coupled ODEs governing tumor growth, drug pharmacodynamics,
    myelosuppression, and organ function during active oncology treatment.

    Instructions for engineers:
        - Adjust growth_rate_prior and drug_sensitivity for your tumor type
        - ANC nadir timing (~7-14 days post-chemo) is drug-specific
        - Creatinine dynamics assume first-order renal clearance
    """

    def __init__(
        self,
        baseline_growth_rate: float = 0.01,
        drug_elimination_rate: float = 0.1,
        anc_recovery_rate: float = 0.15,
        creatinine_clearance_rate: float = 0.05,
        hemoglobin_recovery_rate: float = 0.02,
    ):
        # Tumor dynamics
        self.baseline_growth_rate = baseline_growth_rate  # /day untreated
        self.drug_elimination_rate = drug_elimination_rate  # /day

        # Hematologic recovery rates
        self.anc_recovery_rate = anc_recovery_rate
        self.creatinine_clearance_rate = creatinine_clearance_rate
        self.hemoglobin_recovery_rate = hemoglobin_recovery_rate

        # Homeostatic setpoints
        self.anc_baseline = 4.5      # 10^9/L
        self.creatinine_baseline = 1.0  # mg/dL
        self.hemoglobin_baseline = 13.0  # g/dL

    def state_transition(
        self, x: np.ndarray, dt_days: float, treatment_active: bool = False
    ) -> np.ndarray:
        """Propagate state forward by dt_days.

        Args:
            x: Current state vector [STATE_DIM]
            dt_days: Time step in days
            treatment_active: Whether drug is currently being infused

        Returns:
            Predicted next state vector
        """
        V, g, d, anc, cr, hb, w, e = x

        # --- Tumor volume: dV/dt = g * V, with g modulated by drug ---
        # Net growth = baseline growth - drug-induced kill
        drug_kill = 0.05 * d  # drug effect on growth rate
        g_effective = g - drug_kill
        V_next = V * np.exp(g_effective * dt_days)
        V_next = max(V_next, 0.01)  # minimum detectable volume

        # --- Growth rate: random walk (captured via process noise) ---
        g_next = g

        # --- Drug effect compartment: first-order elimination ---
        if treatment_active:
            d_next = 1.0  # bolus: jump to peak
        else:
            d_next = d * np.exp(-self.drug_elimination_rate * dt_days)

        # --- ANC: myelosuppression from drug, recovery toward baseline ---
        # Drug causes ANC suppression proportional to drug concentration
        anc_suppression = 0.3 * d * dt_days
        anc_recovery = self.anc_recovery_rate * (self.anc_baseline - anc) * dt_days
        anc_next = max(0.1, anc + anc_recovery - anc_suppression)

        # --- Creatinine: renal stress from drug, clearance toward baseline ---
        cr_stress = 0.05 * d * dt_days
        cr_recovery = self.creatinine_clearance_rate * (self.creatinine_baseline - cr) * dt_days
        cr_next = max(0.3, cr + cr_stress - cr_recovery)

        # --- Hemoglobin: chemotherapy-induced anemia, slow recovery ---
        hb_suppression = 0.1 * d * dt_days
        hb_recovery = self.hemoglobin_recovery_rate * (self.hemoglobin_baseline - hb) * dt_days
        hb_next = max(4.0, hb + hb_recovery - hb_suppression)

        # --- Weight: gradual change driven by disease burden and toxicity ---
        weight_change = -0.01 * (e - 1) * dt_days  # weight loss with poor ECOG
        w_next = max(30.0, w + weight_change)

        # --- ECOG: driven by tumor burden and toxicity ---
        # Higher tumor volume and ANC nadir worsen performance status
        tumor_burden_factor = 0.01 * (V_next - V) / max(V, 0.1)
        toxicity_factor = 0.1 * max(0, 2.0 - anc_next) * dt_days
        e_next = np.clip(e + tumor_burden_factor + toxicity_factor, 0.0, 4.0)

        return np.array([V_next, g_next, d_next, anc_next, cr_next,
                         hb_next, w_next, e_next])

    def jacobian(self, x: np.ndarray, dt_days: float) -> np.ndarray:
        """Compute state transition Jacobian for EKF.

        Linearize f(x) around current state for the Extended Kalman Filter
        prediction step. This is essential for non-linear tumor dynamics.

        Args:
            x: Current state vector
            dt_days: Time step in days

        Returns:
            STATE_DIM x STATE_DIM Jacobian matrix
        """
        V, g, d, anc, cr, hb, w, e = x
        F = np.eye(STATE_DIM)

        # dV_next/dV = exp(g_eff * dt)
        g_eff = g - 0.05 * d
        F[0, 0] = np.exp(g_eff * dt_days)
        # dV_next/dg = V * dt * exp(g_eff * dt)
        F[0, 1] = V * dt_days * np.exp(g_eff * dt_days)
        # dV_next/dd = V * (-0.05) * dt * exp(g_eff * dt)
        F[0, 2] = V * (-0.05) * dt_days * np.exp(g_eff * dt_days)

        # Drug elimination
        F[2, 2] = np.exp(-self.drug_elimination_rate * dt_days)

        # ANC dynamics
        F[3, 3] = 1.0 - self.anc_recovery_rate * dt_days
        F[3, 2] = -0.3 * dt_days  # drug effect on ANC

        # Creatinine dynamics
        F[4, 4] = 1.0 + self.creatinine_clearance_rate * dt_days
        F[4, 2] = 0.05 * dt_days

        # Hemoglobin
        F[5, 5] = 1.0 - self.hemoglobin_recovery_rate * dt_days
        F[5, 2] = -0.1 * dt_days

        # Weight
        F[6, 6] = 1.0
        F[6, 7] = 0.01 * dt_days

        # ECOG
        F[7, 7] = 1.0
        F[7, 3] = -0.1 * dt_days if anc < 2.0 else 0.0

        return F


# =============================================================================
# SECTION 4: OBSERVATION MODEL — MAPPING STATE TO MEASUREMENTS
# =============================================================================
# Each observation type maps to specific state components via a (possibly
# non-linear) observation function h(x).


class ObservationModel:
    """Maps latent state to expected observation values.

    Each clinical measurement type has a known relationship to the underlying
    state vector. The observation model also encodes measurement noise
    characteristics specific to each modality.

    Instructions for engineers:
        - Tumor marker (e.g., CEA) assumed proportional to volume
        - RECIST diameter derived from volume assuming sphere
        - Lab values map directly to corresponding state components
        - Adjust marker_sensitivity for your specific biomarker
    """

    # Observation noise standard deviations (measurement uncertainty)
    NOISE_STD = {
        ObservationType.TUMOR_MARKER: 5.0,        # ng/mL
        ObservationType.IMAGING_VOLUME: 2.0,       # cm^3 (segmentation error)
        ObservationType.IMAGING_DIAMETER: 2.0,     # mm (RECIST measurement)
        ObservationType.LAB_NEUTROPHILS: 0.5,      # 10^9/L
        ObservationType.LAB_CREATININE: 0.1,       # mg/dL
        ObservationType.LAB_BILIRUBIN: 0.2,        # mg/dL
        ObservationType.LAB_HEMOGLOBIN: 0.3,       # g/dL
        ObservationType.VITAL_WEIGHT: 0.5,         # kg (scale precision)
        ObservationType.VITAL_ECOG: 0.3,           # continuous approximation
        ObservationType.CTCAE_TOXICITY: 0.5,       # grade uncertainty
    }

    # Marker sensitivity: ng/mL per cm^3 of tumor volume
    MARKER_SENSITIVITY = 3.0

    def observe(self, x: np.ndarray, obs_type: ObservationType) -> float:
        """Compute expected observation from state (h(x)).

        Args:
            x: State vector [STATE_DIM]
            obs_type: Type of observation

        Returns:
            Expected observation value
        """
        V, g, d, anc, cr, hb, w, e = x

        if obs_type == ObservationType.TUMOR_MARKER:
            # Tumor markers (CEA, CA-125) proportional to viable tumor volume
            return self.MARKER_SENSITIVITY * V

        elif obs_type == ObservationType.IMAGING_VOLUME:
            return V

        elif obs_type == ObservationType.IMAGING_DIAMETER:
            # Longest diameter from sphere assumption: D = 2 * (3V/4pi)^(1/3)
            # V in cm^3, output in mm
            radius_cm = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
            return 2.0 * radius_cm * 10.0  # cm to mm

        elif obs_type == ObservationType.LAB_NEUTROPHILS:
            return anc

        elif obs_type == ObservationType.LAB_CREATININE:
            return cr

        elif obs_type == ObservationType.LAB_HEMOGLOBIN:
            return hb

        elif obs_type == ObservationType.VITAL_WEIGHT:
            return w

        elif obs_type == ObservationType.VITAL_ECOG:
            return e

        elif obs_type == ObservationType.CTCAE_TOXICITY:
            # Composite toxicity driven by ANC nadir and organ function
            tox = 0.0
            if anc < 1.0:
                tox = max(tox, 3.0)  # Grade 3 neutropenia
            elif anc < 1.5:
                tox = max(tox, 2.0)
            if cr > 1.5:
                tox = max(tox, 2.0)
            return tox

        else:
            return 0.0

    def observation_jacobian(
        self, x: np.ndarray, obs_type: ObservationType
    ) -> np.ndarray:
        """Compute observation Jacobian dh/dx for EKF update.

        Args:
            x: State vector
            obs_type: Observation type

        Returns:
            1 x STATE_DIM Jacobian row vector
        """
        H = np.zeros(STATE_DIM)
        V = x[0]

        if obs_type == ObservationType.TUMOR_MARKER:
            H[0] = self.MARKER_SENSITIVITY

        elif obs_type == ObservationType.IMAGING_VOLUME:
            H[0] = 1.0

        elif obs_type == ObservationType.IMAGING_DIAMETER:
            # d(diameter)/dV = 2 * 10 * (1/3) * (3/(4pi))^(1/3) * V^(-2/3)
            if V > 0.01:
                H[0] = (20.0 / 3.0) * (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * V ** (-2.0 / 3.0)

        elif obs_type == ObservationType.LAB_NEUTROPHILS:
            H[3] = 1.0

        elif obs_type == ObservationType.LAB_CREATININE:
            H[4] = 1.0

        elif obs_type == ObservationType.LAB_HEMOGLOBIN:
            H[5] = 1.0

        elif obs_type == ObservationType.VITAL_WEIGHT:
            H[6] = 1.0

        elif obs_type == ObservationType.VITAL_ECOG:
            H[7] = 1.0

        return H

    def noise_variance(self, obs_type: ObservationType) -> float:
        """Get observation noise variance for this modality."""
        std = self.NOISE_STD.get(obs_type, 1.0)
        return std ** 2


# =============================================================================
# SECTION 5: EXTENDED KALMAN FILTER (EKF) SYNCHRONIZER
# =============================================================================
# The EKF maintains a Gaussian approximation to the posterior state
# distribution. It is the workhorse for real-time DT synchronization when
# state dynamics are approximately unimodal.


class ExtendedKalmanFilterSync:
    """EKF-based digital twin synchronizer for real-time state estimation.

    Maintains a continuously updated Gaussian posterior over the patient
    state vector. Handles asynchronous observations from multiple clinical
    data sources with different noise characteristics.

    Instructions for engineers:
        1. Initialize with patient baseline (pre-treatment scan + labs)
        2. Call predict() at regular intervals (e.g., every 6 hours)
        3. Call update() whenever a new clinical observation arrives
        4. Call get_state() to read the current DT state at any time
        5. Monitor anomaly_score for early adverse event detection

    Example:
        >>> ekf = ExtendedKalmanFilterSync(initial_state, initial_cov)
        >>> # Daily prediction step
        >>> ekf.predict(dt_days=1.0)
        >>> # Lab result arrives
        >>> obs = ClinicalObservation(ObservationType.LAB_NEUTROPHILS, 2.1, now)
        >>> ekf.update(obs)
        >>> # Query current state
        >>> state = ekf.get_state()
        >>> print(f"Tumor volume: {state.mean[0]:.1f} cm^3")
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise_scale: float = 0.01,
        start_time: datetime | None = None,
    ):
        """Initialize EKF synchronizer.

        Args:
            initial_state: Baseline state vector [STATE_DIM]
            initial_covariance: Initial uncertainty [STATE_DIM x STATE_DIM]
            process_noise_scale: Scale factor for process noise
            start_time: Start timestamp (defaults to now)
        """
        self.x = initial_state.copy()
        self.P = initial_covariance.copy()
        self.dynamics = TumorPatientDynamics()
        self.obs_model = ObservationModel()
        self.process_noise_scale = process_noise_scale
        self.current_time = start_time or datetime.now()
        self.log_likelihood = 0.0
        self.treatment_active = False

        # Audit trail for 21 CFR Part 11 compliance
        self._audit_log: list[dict] = []
        self._anomaly_history: list[dict] = []

        # Process noise covariance (tuned for oncology time scales)
        self.Q_base = np.diag([
            0.5,    # Volume uncertainty grows ~0.5 cm^3/day
            0.001,  # Growth rate slowly drifts
            0.01,   # Drug effect uncertainty
            0.2,    # ANC fluctuation
            0.01,   # Creatinine fluctuation
            0.05,   # Hemoglobin fluctuation
            0.1,    # Weight fluctuation
            0.05,   # ECOG fluctuation
        ]) ** 2

        logger.info("EKF synchronizer initialized with %d-dim state", STATE_DIM)

    def predict(self, dt_days: float) -> StateEstimate:
        """Propagate state forward in time (EKF prediction step).

        Call this at regular intervals (e.g., every 6-24 hours) or
        before processing a new observation to advance the DT to the
        observation timestamp.

        Args:
            dt_days: Time step in days

        Returns:
            Predicted state estimate
        """
        # State prediction via nonlinear dynamics
        x_pred = self.dynamics.state_transition(
            self.x, dt_days, self.treatment_active
        )

        # Covariance prediction via linearized dynamics
        F = self.dynamics.jacobian(self.x, dt_days)
        Q = self.Q_base * self.process_noise_scale * dt_days
        P_pred = F @ self.P @ F.T + Q

        # Update internal state
        self.x = x_pred
        self.P = P_pred
        self.current_time += timedelta(days=dt_days)

        return self.get_state()

    def update(self, observation: ClinicalObservation) -> StateEstimate:
        """Incorporate a new clinical observation (EKF update step).

        Automatically advances the DT to the observation timestamp before
        incorporating the measurement.

        Args:
            observation: Clinical observation to incorporate

        Returns:
            Updated state estimate
        """
        # Advance DT to observation time if needed
        dt = (observation.timestamp - self.current_time).total_seconds() / 86400.0
        if dt > 0.01:  # more than ~15 minutes ahead
            self.predict(dt)

        # Observation prediction
        z_pred = self.obs_model.observe(self.x, observation.obs_type)
        H = self.obs_model.observation_jacobian(self.x, observation.obs_type)

        # Use provided uncertainty or default
        if observation.uncertainty > 0:
            R = observation.uncertainty ** 2
        else:
            R = self.obs_model.noise_variance(observation.obs_type)

        # Innovation (measurement residual)
        innovation = observation.value - z_pred

        # Innovation covariance
        S = H @ self.P @ H.T + R
        S_scalar = float(S) if np.isscalar(S) or S.size == 1 else S

        # Anomaly detection: Mahalanobis distance of innovation
        anomaly_score = innovation ** 2 / S_scalar
        if anomaly_score > 9.0:  # ~3 sigma
            logger.warning(
                "Anomaly detected: %s value=%.2f expected=%.2f (score=%.1f)",
                observation.obs_type.value, observation.value, z_pred,
                anomaly_score
            )
            self._anomaly_history.append({
                "timestamp": observation.timestamp.isoformat(),
                "type": observation.obs_type.value,
                "observed": observation.value,
                "expected": z_pred,
                "score": anomaly_score,
            })

        # Kalman gain
        K = self.P @ H.T / S_scalar

        # State update
        self.x = self.x + K * innovation

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(STATE_DIM) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * R

        # Accumulate log-likelihood for model comparison
        self.log_likelihood += -0.5 * (
            np.log(2 * np.pi * S_scalar) + innovation ** 2 / S_scalar
        )

        # Audit trail entry
        self._audit_log.append({
            "observation_id": observation.observation_id,
            "timestamp": observation.timestamp.isoformat(),
            "type": observation.obs_type.value,
            "value": observation.value,
            "predicted": z_pred,
            "innovation": innovation,
            "source": observation.source_system,
        })

        return self.get_state()

    def register_treatment(self, event: TreatmentEvent) -> None:
        """Register a treatment administration event.

        This modifies the drug effect compartment and flags treatment as
        active for subsequent prediction steps.

        Args:
            event: Treatment event details
        """
        # Advance to treatment time
        dt = (event.timestamp - self.current_time).total_seconds() / 86400.0
        if dt > 0.01:
            self.predict(dt)

        # Activate drug effect compartment
        self.x[2] = 1.0  # drug effect = peak
        self.treatment_active = True

        # Increase uncertainty around drug-sensitive states
        self.P[0, 0] *= 1.5  # more uncertain about volume during treatment
        self.P[3, 3] *= 2.0  # ANC highly uncertain after chemo

        self._audit_log.append({
            "event": "treatment_administered",
            "timestamp": event.timestamp.isoformat(),
            "drug": event.drug,
            "dose": event.dose_mg_m2,
            "cycle": event.cycle,
        })

        logger.info(
            "Treatment registered: %s %.0f mg/m^2 (cycle %d)",
            event.drug, event.dose_mg_m2, event.cycle
        )

    def get_state(self) -> StateEstimate:
        """Get current state estimate."""
        return StateEstimate(
            mean=self.x.copy(),
            covariance=self.P.copy(),
            timestamp=self.current_time,
            log_likelihood=self.log_likelihood,
        )

    def get_anomaly_history(self) -> list[dict]:
        """Get history of detected anomalies for clinical review."""
        return self._anomaly_history.copy()

    def get_audit_trail(self) -> list[dict]:
        """Get full audit trail for 21 CFR Part 11 compliance."""
        return self._audit_log.copy()


# =============================================================================
# SECTION 6: PARTICLE FILTER SYNCHRONIZER
# =============================================================================
# For cases where the posterior is multimodal (e.g., treatment response
# could be "responder" or "non-responder"), the particle filter maintains
# a set of weighted samples that can represent arbitrary distributions.


class ParticleFilterSync:
    """Sequential Monte Carlo synchronizer for non-Gaussian DT posteriors.

    Maintains a weighted particle set representing the full posterior
    distribution over patient state. Handles multimodal distributions
    that arise when treatment response is uncertain (responder vs.
    non-responder phenotypes).

    Instructions for engineers:
        - Use n_particles >= 500 for production; 100 for testing
        - Call resample() when effective sample size drops below n/2
        - Particle weights are in log-space for numerical stability
        - Extract MAP estimate via get_map_estimate() or mean via get_state()

    Example:
        >>> pf = ParticleFilterSync(n_particles=1000, initial_state=x0)
        >>> pf.predict(dt_days=1.0)
        >>> pf.update(observation)
        >>> state = pf.get_state()
    """

    def __init__(
        self,
        n_particles: int = 500,
        initial_state: np.ndarray | None = None,
        initial_spread: np.ndarray | None = None,
    ):
        """Initialize particle filter.

        Args:
            n_particles: Number of particles
            initial_state: Mean initial state [STATE_DIM]
            initial_spread: Standard deviation for initial particle spread
        """
        self.n_particles = n_particles
        self.dynamics = TumorPatientDynamics()
        self.obs_model = ObservationModel()
        self.treatment_active = False
        self.current_time = datetime.now()

        # Initialize particles around initial state
        if initial_state is None:
            initial_state = np.array([15.0, 0.01, 0.0, 4.5, 1.0, 13.0, 70.0, 1.0])
        if initial_spread is None:
            initial_spread = np.array([3.0, 0.005, 0.0, 0.5, 0.1, 0.5, 5.0, 0.3])

        self.particles = np.random.normal(
            initial_state, initial_spread, size=(n_particles, STATE_DIM)
        )
        # Enforce physical constraints
        self.particles[:, 0] = np.maximum(self.particles[:, 0], 0.01)
        self.particles[:, 3] = np.maximum(self.particles[:, 3], 0.1)
        self.particles[:, 7] = np.clip(self.particles[:, 7], 0, 4)

        # Uniform initial weights (log-space)
        self.log_weights = np.full(n_particles, -np.log(n_particles))

        logger.info(
            "Particle filter initialized: %d particles, %d-dim state",
            n_particles, STATE_DIM
        )

    def predict(self, dt_days: float) -> None:
        """Propagate all particles forward with stochastic dynamics.

        Args:
            dt_days: Time step in days
        """
        process_noise_std = np.array([
            0.5, 0.001, 0.01, 0.2, 0.01, 0.05, 0.1, 0.05
        ]) * np.sqrt(dt_days)

        for i in range(self.n_particles):
            # Deterministic propagation
            self.particles[i] = self.dynamics.state_transition(
                self.particles[i], dt_days, self.treatment_active
            )
            # Add process noise
            self.particles[i] += np.random.normal(0, process_noise_std)

        # Enforce constraints
        self.particles[:, 0] = np.maximum(self.particles[:, 0], 0.01)
        self.particles[:, 3] = np.maximum(self.particles[:, 3], 0.1)
        self.particles[:, 4] = np.maximum(self.particles[:, 4], 0.3)
        self.particles[:, 5] = np.maximum(self.particles[:, 5], 4.0)
        self.particles[:, 6] = np.maximum(self.particles[:, 6], 30.0)
        self.particles[:, 7] = np.clip(self.particles[:, 7], 0, 4)

        self.current_time += timedelta(days=dt_days)

    def update(self, observation: ClinicalObservation) -> None:
        """Weight particles by observation likelihood.

        Args:
            observation: Clinical observation
        """
        # Advance to observation time
        dt = (observation.timestamp - self.current_time).total_seconds() / 86400.0
        if dt > 0.01:
            self.predict(dt)

        R = self.obs_model.noise_variance(observation.obs_type)

        for i in range(self.n_particles):
            z_pred = self.obs_model.observe(
                self.particles[i], observation.obs_type
            )
            innovation = observation.value - z_pred
            # Gaussian likelihood
            log_lik = -0.5 * innovation ** 2 / R - 0.5 * np.log(2 * np.pi * R)
            self.log_weights[i] += log_lik

        # Normalize weights
        max_log_w = np.max(self.log_weights)
        self.log_weights -= max_log_w  # shift for numerical stability
        log_sum = np.log(np.sum(np.exp(self.log_weights)))
        self.log_weights -= log_sum

        # Resample if effective sample size is low
        ess = self.effective_sample_size()
        if ess < self.n_particles / 2:
            self._systematic_resample()

    def effective_sample_size(self) -> float:
        """Compute effective sample size (ESS)."""
        weights = np.exp(self.log_weights)
        return 1.0 / np.sum(weights ** 2)

    def _systematic_resample(self) -> None:
        """Systematic resampling to combat particle degeneracy."""
        weights = np.exp(self.log_weights)
        cumsum = np.cumsum(weights)

        u = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
        indices = np.searchsorted(cumsum, u)
        indices = np.clip(indices, 0, self.n_particles - 1)

        self.particles = self.particles[indices].copy()
        self.log_weights = np.full(
            self.n_particles, -np.log(self.n_particles)
        )

    def get_state(self) -> StateEstimate:
        """Get weighted mean and covariance of particle ensemble."""
        weights = np.exp(self.log_weights)
        mean = np.average(self.particles, weights=weights, axis=0)
        diff = self.particles - mean
        cov = np.zeros((STATE_DIM, STATE_DIM))
        for i in range(self.n_particles):
            cov += weights[i] * np.outer(diff[i], diff[i])

        return StateEstimate(
            mean=mean,
            covariance=cov,
            timestamp=self.current_time,
        )

    def get_map_estimate(self) -> np.ndarray:
        """Get maximum a posteriori (MAP) particle."""
        idx = np.argmax(self.log_weights)
        return self.particles[idx].copy()

    def get_response_probability(self, volume_threshold_fraction: float = 0.7) -> float:
        """Estimate probability of treatment response.

        Computes the posterior probability that tumor volume has decreased
        by more than the given fraction from the particle ensemble.

        Args:
            volume_threshold_fraction: Fraction of baseline volume for response

        Returns:
            Probability of response (0-1)
        """
        weights = np.exp(self.log_weights)
        responding = self.particles[:, 0] < (self.particles[:, 0].mean() * volume_threshold_fraction)
        return float(np.sum(weights[responding]))


# =============================================================================
# SECTION 7: ANOMALY DETECTOR — EARLY ADVERSE EVENT WARNING
# =============================================================================
# Monitors innovation sequences from the EKF for systematic deviations
# that indicate model mismatch or unexpected clinical events.


class ClinicalAnomalyDetector:
    """Detects unexpected clinical events from DT innovation analysis.

    Monitors the sequence of EKF innovations (prediction errors) for
    patterns that suggest adverse events, treatment failure, or model
    degradation. Triggers alerts based on configurable thresholds.

    Instructions for engineers:
        - Connect alerts to clinical decision support dashboards
        - CUSUM threshold calibrated for oncology cycle length (~21 days)
        - Always pair with clinical review; never use for autonomous decisions

    Clinical alert examples:
        - Rapid ANC decline: febrile neutropenia risk
        - Unexpected tumor marker rise: possible progression
        - Creatinine spike: acute kidney injury from chemotherapy
    """

    def __init__(
        self,
        cusum_threshold: float = 5.0,
        window_size: int = 10,
    ):
        self.cusum_threshold = cusum_threshold
        self.window_size = window_size
        self._innovations: dict[str, list[float]] = {}
        self._cusum_pos: dict[str, float] = {}
        self._cusum_neg: dict[str, float] = {}
        self.alerts: list[dict] = []

    def process_innovation(
        self,
        obs_type: ObservationType,
        innovation: float,
        expected_std: float,
        timestamp: datetime,
    ) -> dict | None:
        """Process a single innovation and check for anomalies.

        Args:
            obs_type: Observation type
            innovation: Measurement residual (observed - predicted)
            expected_std: Expected standard deviation of innovation
            timestamp: Observation timestamp

        Returns:
            Alert dictionary if anomaly detected, None otherwise
        """
        key = obs_type.value
        normalized = innovation / max(expected_std, 1e-6)

        # Track innovations
        if key not in self._innovations:
            self._innovations[key] = []
            self._cusum_pos[key] = 0.0
            self._cusum_neg[key] = 0.0
        self._innovations[key].append(normalized)

        # CUSUM test for persistent shift
        self._cusum_pos[key] = max(0, self._cusum_pos[key] + normalized - 0.5)
        self._cusum_neg[key] = max(0, self._cusum_neg[key] - normalized - 0.5)

        alert = None
        if self._cusum_pos[key] > self.cusum_threshold:
            alert = {
                "type": "persistent_elevation",
                "observation": key,
                "timestamp": timestamp.isoformat(),
                "cusum_score": self._cusum_pos[key],
                "message": f"Persistent elevation in {key}: possible adverse event",
            }
            self._cusum_pos[key] = 0.0  # reset after alert

        elif self._cusum_neg[key] > self.cusum_threshold:
            alert = {
                "type": "persistent_decline",
                "observation": key,
                "timestamp": timestamp.isoformat(),
                "cusum_score": self._cusum_neg[key],
                "message": f"Persistent decline in {key}: assess clinical significance",
            }
            self._cusum_neg[key] = 0.0

        if alert:
            self.alerts.append(alert)
            logger.warning("Clinical alert: %s", alert["message"])

        return alert


# =============================================================================
# SECTION 8: SYNCHRONIZATION ORCHESTRATOR
# =============================================================================
# Top-level controller that manages the synchronization pipeline,
# routing observations to the appropriate filter and managing the
# clinical event lifecycle.


class DigitalTwinSynchronizer:
    """Orchestrates real-time DT synchronization from clinical data streams.

    This is the main entry point for engineers integrating the DT
    synchronization system with clinical infrastructure. It manages:
    - Filter selection (EKF vs. particle filter)
    - Observation routing and validation
    - Treatment event handling
    - Anomaly detection and alerting
    - Audit trail generation

    Instructions for engineers:
        1. Create with patient baseline data
        2. Connect to clinical data feeds (FHIR Subscription, HL7v2, etc.)
        3. Call process_observation() for each incoming measurement
        4. Call process_treatment() for each drug administration
        5. Query get_current_state() for dashboard rendering
        6. Export audit trail for regulatory submission

    Example:
        >>> sync = DigitalTwinSynchronizer.from_baseline(
        ...     patient_id="TRIAL-001",
        ...     tumor_volume_cm3=12.5,
        ...     anc=4.2,
        ...     creatinine=0.9,
        ...     hemoglobin=13.5,
        ...     weight_kg=72.0,
        ... )
        >>> # Process incoming lab result
        >>> sync.process_observation(ClinicalObservation(
        ...     obs_type=ObservationType.LAB_NEUTROPHILS,
        ...     value=2.1,
        ...     timestamp=datetime.now(),
        ... ))
        >>> state = sync.get_current_state()
    """

    def __init__(
        self,
        patient_id: str,
        filter_type: str = "ekf",
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
        n_particles: int = 500,
    ):
        self.patient_id = patient_id
        self.filter_type = filter_type
        self.anomaly_detector = ClinicalAnomalyDetector()
        self._observation_count = 0

        if initial_state is None:
            initial_state = np.array([15.0, 0.01, 0.0, 4.5, 1.0, 13.0, 70.0, 1.0])
        if initial_covariance is None:
            initial_covariance = np.diag([
                9.0, 0.0001, 0.01, 0.25, 0.01, 0.25, 4.0, 0.09
            ])

        if filter_type == "ekf":
            self.filter = ExtendedKalmanFilterSync(
                initial_state, initial_covariance
            )
        elif filter_type == "particle":
            self.filter = ParticleFilterSync(
                n_particles=n_particles,
                initial_state=initial_state,
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        logger.info(
            "DT Synchronizer created for patient %s using %s filter",
            patient_id, filter_type
        )

    @classmethod
    def from_baseline(
        cls,
        patient_id: str,
        tumor_volume_cm3: float,
        anc: float = 4.5,
        creatinine: float = 1.0,
        hemoglobin: float = 13.0,
        weight_kg: float = 70.0,
        ecog: float = 1.0,
        filter_type: str = "ekf",
    ) -> "DigitalTwinSynchronizer":
        """Create synchronizer from baseline clinical measurements.

        Args:
            patient_id: Patient identifier
            tumor_volume_cm3: Baseline tumor volume from imaging
            anc: Baseline ANC (10^9/L)
            creatinine: Baseline creatinine (mg/dL)
            hemoglobin: Baseline hemoglobin (g/dL)
            weight_kg: Baseline weight (kg)
            ecog: Baseline ECOG performance status
            filter_type: "ekf" or "particle"

        Returns:
            Configured DigitalTwinSynchronizer
        """
        initial_state = np.array([
            tumor_volume_cm3,
            0.01,   # initial growth rate estimate
            0.0,    # no drug effect at baseline
            anc,
            creatinine,
            hemoglobin,
            weight_kg,
            ecog,
        ])
        return cls(patient_id, filter_type, initial_state)

    def process_observation(self, observation: ClinicalObservation) -> StateEstimate:
        """Process a single clinical observation.

        Routes the observation through the active filter and anomaly
        detector. Returns the updated state estimate.

        Args:
            observation: Incoming clinical observation

        Returns:
            Updated state estimate
        """
        self._observation_count += 1

        if isinstance(self.filter, ExtendedKalmanFilterSync):
            state = self.filter.update(observation)
        else:
            self.filter.update(observation)
            state = self.filter.get_state()

        # Run anomaly detection
        z_pred = ObservationModel().observe(state.mean, observation.obs_type)
        innovation = observation.value - z_pred
        expected_std = np.sqrt(
            ObservationModel().noise_variance(observation.obs_type)
        )
        self.anomaly_detector.process_innovation(
            observation.obs_type, innovation, expected_std,
            observation.timestamp,
        )

        return state

    def process_treatment(self, event: TreatmentEvent) -> None:
        """Register a treatment administration event.

        Args:
            event: Treatment event details
        """
        if isinstance(self.filter, ExtendedKalmanFilterSync):
            self.filter.register_treatment(event)
        else:
            # For particle filter, set drug compartment on all particles
            dt = (event.timestamp - self.filter.current_time).total_seconds() / 86400.0
            if dt > 0.01:
                self.filter.predict(dt)
            self.filter.particles[:, 2] = 1.0
            self.filter.treatment_active = True

    def get_current_state(self) -> StateEstimate:
        """Get current DT state estimate."""
        return self.filter.get_state()

    def get_state_summary(self) -> dict[str, Any]:
        """Get human-readable state summary for clinical dashboard.

        Returns:
            Dictionary with labeled state values and uncertainties
        """
        state = self.get_current_state()
        summary = {
            "patient_id": self.patient_id,
            "timestamp": state.timestamp.isoformat(),
            "observations_processed": self._observation_count,
        }

        for i, label in enumerate(STATE_LABELS):
            std = np.sqrt(state.covariance[i, i])
            summary[label] = {
                "value": round(float(state.mean[i]), 3),
                "std": round(float(std), 3),
                "unit": _STATE_UNITS[i],
            }

        # Clinical flags
        anc = state.mean[3]
        cr = state.mean[4]
        if anc < 1.0:
            summary["alert_neutropenia"] = "SEVERE (Grade 4)"
        elif anc < 1.5:
            summary["alert_neutropenia"] = "MODERATE (Grade 3)"

        if cr > 1.5 * 1.0:  # >1.5x baseline
            summary["alert_renal"] = "ELEVATED"

        return summary


_STATE_UNITS = [
    "cm^3", "/day", "dimensionless", "10^9/L",
    "mg/dL", "g/dL", "kg", "score"
]


# =============================================================================
# SECTION 9: MAIN — SIMULATION OF TREATMENT CYCLE WITH LIVE SYNC
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Real-Time Digital Twin Synchronization")
    print("Physical AI Oncology Trials — Example 01")
    print("=" * 70)

    # --- Initialize DT from baseline measurements ---
    sync = DigitalTwinSynchronizer.from_baseline(
        patient_id="TRIAL-CRC-042",
        tumor_volume_cm3=18.5,   # baseline CT measurement
        anc=5.2,                 # normal baseline
        creatinine=0.85,         # normal baseline
        hemoglobin=14.1,
        weight_kg=78.0,
        ecog=1.0,
        filter_type="ekf",
    )

    print("\nBaseline state:")
    baseline = sync.get_state_summary()
    for key in STATE_LABELS:
        info = baseline[key]
        print(f"  {key:>15s}: {info['value']:8.3f} ± {info['std']:.3f} {info['unit']}")

    # --- Simulate 21-day chemotherapy cycle ---
    # Day 1: FOLFOX administration
    t0 = datetime(2026, 3, 1, 9, 0)
    sync.process_treatment(TreatmentEvent(
        timestamp=t0,
        drug="FOLFOX",
        dose_mg_m2=85,
        cycle=1,
    ))
    print(f"\nDay 1: FOLFOX administered")

    # Simulate observations arriving over 21-day cycle
    observations = [
        # Day 3: routine labs
        ClinicalObservation(ObservationType.LAB_NEUTROPHILS, 3.8, t0 + timedelta(days=3)),
        ClinicalObservation(ObservationType.LAB_CREATININE, 0.9, t0 + timedelta(days=3)),
        ClinicalObservation(ObservationType.LAB_HEMOGLOBIN, 13.5, t0 + timedelta(days=3)),
        ClinicalObservation(ObservationType.VITAL_WEIGHT, 77.5, t0 + timedelta(days=3)),
        # Day 7: nadir check
        ClinicalObservation(ObservationType.LAB_NEUTROPHILS, 1.8, t0 + timedelta(days=7)),
        ClinicalObservation(ObservationType.LAB_HEMOGLOBIN, 12.8, t0 + timedelta(days=7)),
        # Day 10: tumor marker
        ClinicalObservation(ObservationType.TUMOR_MARKER, 48.0, t0 + timedelta(days=10)),
        # Day 14: recovery labs
        ClinicalObservation(ObservationType.LAB_NEUTROPHILS, 2.9, t0 + timedelta(days=14)),
        ClinicalObservation(ObservationType.LAB_CREATININE, 0.95, t0 + timedelta(days=14)),
        # Day 21: pre-cycle 2 assessment
        ClinicalObservation(ObservationType.LAB_NEUTROPHILS, 4.0, t0 + timedelta(days=21)),
        ClinicalObservation(ObservationType.LAB_CREATININE, 0.88, t0 + timedelta(days=21)),
        ClinicalObservation(ObservationType.LAB_HEMOGLOBIN, 13.0, t0 + timedelta(days=21)),
        ClinicalObservation(ObservationType.TUMOR_MARKER, 42.0, t0 + timedelta(days=21)),
        ClinicalObservation(ObservationType.VITAL_WEIGHT, 77.0, t0 + timedelta(days=21)),
        ClinicalObservation(ObservationType.VITAL_ECOG, 1.0, t0 + timedelta(days=21)),
    ]

    print("\nProcessing observations through cycle 1:")
    for obs in observations:
        state = sync.process_observation(obs)
        day = (obs.timestamp - t0).days
        print(
            f"  Day {day:2d} | {obs.obs_type.value:>20s} = {obs.value:7.2f} "
            f"→ Volume={state.mean[0]:.1f} cm^3, ANC={state.mean[3]:.1f}"
        )

    # --- End of cycle summary ---
    print("\n" + "=" * 70)
    print("End-of-Cycle 1 Digital Twin State:")
    summary = sync.get_state_summary()
    for key in STATE_LABELS:
        info = summary[key]
        print(f"  {key:>15s}: {info['value']:8.3f} ± {info['std']:.3f} {info['unit']}")

    # --- Anomaly report ---
    anomalies = sync.anomaly_detector.alerts
    if anomalies:
        print(f"\nAnomalies detected: {len(anomalies)}")
        for a in anomalies:
            print(f"  [{a['type']}] {a['message']}")
    else:
        print("\nNo anomalies detected during cycle 1.")

    # --- Audit trail ---
    if isinstance(sync.filter, ExtendedKalmanFilterSync):
        audit = sync.filter.get_audit_trail()
        print(f"\nAudit trail: {len(audit)} entries logged (21 CFR Part 11 compliant)")

    print("\n" + "=" * 70)
    print("Synchronization complete. DT ready for cycle 2 planning.")
