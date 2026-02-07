"""
=============================================================================
EXAMPLE 03: Real-Time Adaptive Treatment Agent
=============================================================================

Implements an agentic system that processes streaming multi-modal data
(robot telemetry, patient vitals, intraoperative imaging) and makes
real-time treatment adjustment recommendations during robotic oncology
procedures.

CLINICAL CONTEXT:
-----------------
During robotic oncology surgery, the surgical team must continuously
integrate information from multiple sources:
  - Robot force/torque sensors detecting tissue resistance
  - Patient hemodynamic monitoring (heart rate, BP, SpO2)
  - Intraoperative imaging (ultrasound, fluorescence, endoscopy)
  - Pathology results (frozen sections, margin status)
  - Treatment protocol constraints

An adaptive treatment agent monitors these streams, detects anomalies,
correlates cross-modal events, and recommends adjustments in real time.

DISTINCTION FROM EXISTING EXAMPLES:
------------------------------------
- examples-new/01_realtime_safety_monitoring.py: Sensor-level safety
  (force limits, joint limits, hardware watchdogs)
- This example: Agent-level reasoning that correlates multi-modal data
  streams and makes clinical treatment decisions

ARCHITECTURE:
-------------
  Sensors  -->  Stream    -->  Anomaly   -->  Decision  -->  Recommendation
  (multi-     Processor     Detection     Engine        Engine
   modal)                                    |
                                         Clinical
                                         Knowledge
                                          Base

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - numpy >= 1.24.0 (signal processing)

Optional:
    - scipy >= 1.11.0 (filtering)
    - anthropic >= 0.40.0 (LLM reasoning for complex decisions)

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# SECTION 1: DATA STREAM MODELS
# =============================================================================


class StreamType(Enum):
    """Types of data streams monitored by the agent."""

    FORCE_TORQUE = "force_torque"
    PATIENT_VITALS = "patient_vitals"
    IMAGING = "imaging"
    PATHOLOGY = "pathology"
    ROBOT_STATE = "robot_state"
    ANESTHESIA = "anesthesia"


class AlertSeverity(Enum):
    """Severity levels for agent-generated alerts."""

    INFO = "info"
    ADVISORY = "advisory"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class TreatmentAction(Enum):
    """Possible treatment adjustment actions."""

    CONTINUE = "continue_as_planned"
    SLOW_DOWN = "reduce_speed"
    PAUSE = "pause_procedure"
    ADJUST_APPROACH = "adjust_surgical_approach"
    CHANGE_INSTRUMENT = "change_instrument"
    INCREASE_MARGIN = "increase_resection_margin"
    REDUCE_FORCE = "reduce_applied_force"
    REQUEST_IMAGING = "request_additional_imaging"
    REQUEST_PATHOLOGY = "request_frozen_section"
    CONVERT_OPEN = "convert_to_open_surgery"
    ABORT = "abort_procedure"


@dataclass
class DataSample:
    """A single timestamped data sample from any stream."""

    stream_type: StreamType
    timestamp: float
    values: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class AnomalyEvent:
    """Detected anomaly in a data stream."""

    stream_type: StreamType
    timestamp: float
    anomaly_type: str
    severity: AlertSeverity
    description: str
    values: dict = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class TreatmentRecommendation:
    """Agent recommendation for treatment adjustment."""

    action: TreatmentAction
    severity: AlertSeverity
    rationale: str
    supporting_evidence: list = field(default_factory=list)
    confidence: float = 0.0
    requires_confirmation: bool = True
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for logging and display."""
        return {
            "action": self.action.value,
            "severity": self.severity.value,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "requires_confirmation": self.requires_confirmation,
            "timestamp": self.timestamp,
        }


# =============================================================================
# SECTION 2: STREAM PROCESSORS
# =============================================================================


class StreamBuffer:
    """Sliding window buffer for a data stream."""

    def __init__(self, stream_type: StreamType, window_seconds: float = 30.0):
        self.stream_type = stream_type
        self.window_seconds = window_seconds
        self._samples: deque = deque()
        self._anomalies: list[AnomalyEvent] = []

    def add_sample(self, sample: DataSample) -> None:
        """Add a sample and prune old data."""
        self._samples.append(sample)
        cutoff = time.time() - self.window_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def get_window(self) -> list[DataSample]:
        """Get all samples in the current window."""
        return list(self._samples)

    def get_latest(self) -> Optional[DataSample]:
        """Get the most recent sample."""
        return self._samples[-1] if self._samples else None

    @property
    def sample_count(self) -> int:
        return len(self._samples)


class ForceTorqueProcessor:
    """
    Process force/torque sensor data for anomaly detection.

    Monitors for:
    - Excessive force during tissue interaction
    - Sudden force spikes indicating unexpected tissue contact
    - Force trends indicating tissue compliance changes
    - Torque anomalies during instrument manipulation
    """

    def __init__(self):
        self.force_threshold_n = 5.0
        self.spike_threshold_n = 3.0  # Sudden change threshold
        self.torque_threshold_nm = 0.5
        self._prev_force_magnitude = 0.0

    def process(self, buffer: StreamBuffer) -> list[AnomalyEvent]:
        """Process force/torque stream for anomalies."""
        anomalies = []
        latest = buffer.get_latest()
        if latest is None:
            return anomalies

        forces = latest.values.get("forces", [0.0, 0.0, 0.0])
        torques = latest.values.get("torques", [0.0, 0.0, 0.0])

        force_magnitude = math.sqrt(sum(f * f for f in forces))
        torque_magnitude = math.sqrt(sum(t * t for t in torques))

        # Check absolute force threshold
        if force_magnitude > self.force_threshold_n:
            anomalies.append(
                AnomalyEvent(
                    stream_type=StreamType.FORCE_TORQUE,
                    timestamp=latest.timestamp,
                    anomaly_type="excessive_force",
                    severity=AlertSeverity.WARNING if force_magnitude < 8.0 else AlertSeverity.CRITICAL,
                    description=f"Force magnitude {force_magnitude:.1f}N exceeds threshold {self.force_threshold_n}N",
                    values={"force_n": force_magnitude, "threshold_n": self.force_threshold_n},
                    confidence=0.95,
                )
            )

        # Check for force spikes
        force_delta = abs(force_magnitude - self._prev_force_magnitude)
        if force_delta > self.spike_threshold_n:
            anomalies.append(
                AnomalyEvent(
                    stream_type=StreamType.FORCE_TORQUE,
                    timestamp=latest.timestamp,
                    anomaly_type="force_spike",
                    severity=AlertSeverity.WARNING,
                    description=f"Sudden force change of {force_delta:.1f}N detected",
                    values={"delta_n": force_delta, "current_n": force_magnitude},
                    confidence=0.90,
                )
            )

        # Check torque threshold
        if torque_magnitude > self.torque_threshold_nm:
            anomalies.append(
                AnomalyEvent(
                    stream_type=StreamType.FORCE_TORQUE,
                    timestamp=latest.timestamp,
                    anomaly_type="excessive_torque",
                    severity=AlertSeverity.ADVISORY,
                    description=f"Torque magnitude {torque_magnitude:.2f}Nm exceeds threshold",
                    values={"torque_nm": torque_magnitude},
                    confidence=0.85,
                )
            )

        self._prev_force_magnitude = force_magnitude
        return anomalies


class VitalsProcessor:
    """
    Process patient vitals for hemodynamic anomalies.

    Monitors for:
    - Tachycardia/bradycardia indicating pain or hemorrhage
    - Hypotension suggesting blood loss
    - Desaturation indicating airway or ventilation issues
    - Composite hemodynamic instability scores
    """

    def __init__(self):
        self.hr_range = (50.0, 120.0)
        self.sbp_range = (90.0, 160.0)
        self.spo2_min = 92.0
        self.etco2_range = (30.0, 45.0)

    def process(self, buffer: StreamBuffer) -> list[AnomalyEvent]:
        """Process vitals stream for anomalies."""
        anomalies = []
        latest = buffer.get_latest()
        if latest is None:
            return anomalies

        hr = latest.values.get("heart_rate_bpm", 72.0)
        sbp = latest.values.get("systolic_bp", 120.0)
        spo2 = latest.values.get("spo2_percent", 98.0)
        etco2 = latest.values.get("etco2_mmhg", 35.0)

        # Heart rate anomaly
        if hr < self.hr_range[0] or hr > self.hr_range[1]:
            condition = "tachycardia" if hr > self.hr_range[1] else "bradycardia"
            anomalies.append(
                AnomalyEvent(
                    stream_type=StreamType.PATIENT_VITALS,
                    timestamp=latest.timestamp,
                    anomaly_type=condition,
                    severity=AlertSeverity.WARNING,
                    description=f"Heart rate {hr:.0f} bpm ({condition})",
                    values={"heart_rate_bpm": hr, "normal_range": list(self.hr_range)},
                    confidence=0.95,
                )
            )

        # Blood pressure anomaly
        if sbp < self.sbp_range[0]:
            anomalies.append(
                AnomalyEvent(
                    stream_type=StreamType.PATIENT_VITALS,
                    timestamp=latest.timestamp,
                    anomaly_type="hypotension",
                    severity=AlertSeverity.CRITICAL if sbp < 80.0 else AlertSeverity.WARNING,
                    description=f"Systolic BP {sbp:.0f} mmHg (hypotension)",
                    values={"systolic_bp": sbp, "minimum_threshold": self.sbp_range[0]},
                    confidence=0.95,
                )
            )

        # SpO2 anomaly
        if spo2 < self.spo2_min:
            anomalies.append(
                AnomalyEvent(
                    stream_type=StreamType.PATIENT_VITALS,
                    timestamp=latest.timestamp,
                    anomaly_type="desaturation",
                    severity=AlertSeverity.CRITICAL if spo2 < 88.0 else AlertSeverity.WARNING,
                    description=f"SpO2 {spo2:.0f}% (desaturation)",
                    values={"spo2_percent": spo2, "threshold": self.spo2_min},
                    confidence=0.98,
                )
            )

        # EtCO2 anomaly
        if etco2 < self.etco2_range[0] or etco2 > self.etco2_range[1]:
            anomalies.append(
                AnomalyEvent(
                    stream_type=StreamType.PATIENT_VITALS,
                    timestamp=latest.timestamp,
                    anomaly_type="etco2_abnormal",
                    severity=AlertSeverity.ADVISORY,
                    description=f"EtCO2 {etco2:.0f} mmHg outside normal range",
                    values={"etco2_mmhg": etco2, "normal_range": list(self.etco2_range)},
                    confidence=0.85,
                )
            )

        return anomalies


class ImagingProcessor:
    """
    Process intraoperative imaging data for clinical events.

    Monitors for:
    - Margin adequacy from real-time imaging updates
    - Tissue perfusion changes from fluorescence imaging
    - New findings requiring plan adjustment
    """

    def __init__(self):
        self.min_margin_mm = 5.0

    def process(self, buffer: StreamBuffer) -> list[AnomalyEvent]:
        """Process imaging stream for anomalies."""
        anomalies = []
        latest = buffer.get_latest()
        if latest is None:
            return anomalies

        margin_mm = latest.values.get("estimated_margin_mm")
        if margin_mm is not None and margin_mm < self.min_margin_mm:
            anomalies.append(
                AnomalyEvent(
                    stream_type=StreamType.IMAGING,
                    timestamp=latest.timestamp,
                    anomaly_type="inadequate_margin",
                    severity=AlertSeverity.WARNING,
                    description=f"Estimated margin {margin_mm:.1f}mm below minimum {self.min_margin_mm}mm",
                    values={"margin_mm": margin_mm, "minimum_mm": self.min_margin_mm},
                    confidence=latest.values.get("confidence", 0.8),
                )
            )

        perfusion_change = latest.values.get("perfusion_change_percent")
        if perfusion_change is not None and abs(perfusion_change) > 30.0:
            anomalies.append(
                AnomalyEvent(
                    stream_type=StreamType.IMAGING,
                    timestamp=latest.timestamp,
                    anomaly_type="perfusion_change",
                    severity=AlertSeverity.ADVISORY,
                    description=f"Tissue perfusion changed by {perfusion_change:.0f}%",
                    values={"perfusion_change_percent": perfusion_change},
                    confidence=0.75,
                )
            )

        return anomalies


# =============================================================================
# SECTION 3: CROSS-MODAL CORRELATION ENGINE
# =============================================================================


@dataclass
class CorrelatedEvent:
    """An event correlated across multiple data streams."""

    event_type: str
    severity: AlertSeverity
    description: str
    contributing_anomalies: list = field(default_factory=list)
    clinical_significance: str = ""
    confidence: float = 0.0
    timestamp: float = 0.0


class CrossModalCorrelator:
    """
    Correlates anomalies across data streams to identify
    clinically significant events.

    CORRELATION RULES:
    ------------------
    1. Force spike + vital sign change = possible hemorrhage
    2. Force increase + imaging change = tissue boundary transition
    3. Hypotension + tachycardia = hemodynamic instability
    4. Inadequate margin + force feedback = resection boundary concern
    """

    def __init__(self, correlation_window_seconds: float = 10.0):
        self.correlation_window = correlation_window_seconds
        self._recent_anomalies: deque = deque()

    def add_anomalies(self, anomalies: list[AnomalyEvent]) -> None:
        """Add new anomalies to the correlation buffer."""
        for anomaly in anomalies:
            self._recent_anomalies.append(anomaly)

        # Prune old anomalies
        cutoff = time.time() - self.correlation_window
        while self._recent_anomalies and self._recent_anomalies[0].timestamp < cutoff:
            self._recent_anomalies.popleft()

    def detect_correlations(self) -> list[CorrelatedEvent]:
        """Detect cross-modal correlations among recent anomalies."""
        correlations = []
        anomaly_list = list(self._recent_anomalies)

        # Group anomalies by type
        force_anomalies = [a for a in anomaly_list if a.stream_type == StreamType.FORCE_TORQUE]
        vital_anomalies = [a for a in anomaly_list if a.stream_type == StreamType.PATIENT_VITALS]
        imaging_anomalies = [a for a in anomaly_list if a.stream_type == StreamType.IMAGING]

        # Rule 1: Force spike + hemodynamic change = possible hemorrhage
        force_spikes = [a for a in force_anomalies if a.anomaly_type == "force_spike"]
        vital_changes = [a for a in vital_anomalies if a.anomaly_type in ("tachycardia", "hypotension")]

        if force_spikes and vital_changes:
            correlations.append(
                CorrelatedEvent(
                    event_type="possible_hemorrhage",
                    severity=AlertSeverity.CRITICAL,
                    description=(
                        "Force spike correlated with hemodynamic changes suggests "
                        "possible vascular injury or hemorrhage"
                    ),
                    contributing_anomalies=[force_spikes[-1].description, vital_changes[-1].description],
                    clinical_significance=(
                        "Concurrent force anomaly and hemodynamic instability may indicate "
                        "inadvertent vessel injury. Requires immediate assessment."
                    ),
                    confidence=0.85,
                    timestamp=time.time(),
                )
            )

        # Rule 2: Hypotension + tachycardia = hemodynamic instability
        hypotension = [a for a in vital_anomalies if a.anomaly_type == "hypotension"]
        tachycardia = [a for a in vital_anomalies if a.anomaly_type == "tachycardia"]

        if hypotension and tachycardia:
            correlations.append(
                CorrelatedEvent(
                    event_type="hemodynamic_instability",
                    severity=AlertSeverity.CRITICAL,
                    description="Concurrent hypotension and tachycardia indicate hemodynamic instability",
                    contributing_anomalies=[hypotension[-1].description, tachycardia[-1].description],
                    clinical_significance=(
                        "Combined hypotension and tachycardia is a compensatory response. "
                        "Assess for hemorrhage, pneumothorax, or anesthetic cause."
                    ),
                    confidence=0.92,
                    timestamp=time.time(),
                )
            )

        # Rule 3: Inadequate margin + force increase = resection concern
        margin_issues = [a for a in imaging_anomalies if a.anomaly_type == "inadequate_margin"]
        force_high = [a for a in force_anomalies if a.anomaly_type == "excessive_force"]

        if margin_issues and force_high:
            correlations.append(
                CorrelatedEvent(
                    event_type="resection_boundary_concern",
                    severity=AlertSeverity.WARNING,
                    description="Inadequate margin with high force suggests tissue boundary proximity",
                    contributing_anomalies=[margin_issues[-1].description, force_high[-1].description],
                    clinical_significance=(
                        "High force near inadequate margins may indicate the resection plane "
                        "is too close to the tumor. Consider adjusting approach."
                    ),
                    confidence=0.78,
                    timestamp=time.time(),
                )
            )

        return correlations


# =============================================================================
# SECTION 4: TREATMENT DECISION ENGINE
# =============================================================================


class TreatmentDecisionEngine:
    """
    Maps anomalies and correlated events to treatment recommendations.

    Uses a rule-based decision framework with escalation levels
    aligned to clinical severity. Each recommendation includes
    rationale and supporting evidence for the surgical team.

    DECISION HIERARCHY:
    -------------------
    1. Single-stream anomalies -> Targeted adjustments
    2. Correlated events -> Broader procedure modifications
    3. Multiple correlated events -> Procedure-level decisions
    """

    def __init__(self):
        self._recommendation_history: list[TreatmentRecommendation] = []

    def evaluate(
        self,
        anomalies: list[AnomalyEvent],
        correlations: list[CorrelatedEvent],
    ) -> list[TreatmentRecommendation]:
        """
        Generate treatment recommendations from anomalies and correlations.

        Args:
            anomalies: Individual stream anomalies
            correlations: Cross-modal correlated events

        Returns:
            Prioritized list of treatment recommendations
        """
        recommendations = []

        # Process correlated events first (higher priority)
        for event in correlations:
            rec = self._handle_correlated_event(event)
            if rec:
                recommendations.append(rec)

        # Process individual anomalies
        for anomaly in anomalies:
            rec = self._handle_single_anomaly(anomaly)
            if rec:
                # Don't duplicate if correlation already covered
                if not any(r.action == rec.action for r in recommendations):
                    recommendations.append(rec)

        # Sort by severity (most severe first)
        severity_order = {
            AlertSeverity.EMERGENCY: 0,
            AlertSeverity.CRITICAL: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.ADVISORY: 3,
            AlertSeverity.INFO: 4,
        }
        recommendations.sort(key=lambda r: severity_order.get(r.severity, 5))

        self._recommendation_history.extend(recommendations)
        return recommendations

    def _handle_correlated_event(self, event: CorrelatedEvent) -> Optional[TreatmentRecommendation]:
        """Generate recommendation from a correlated event."""
        event_actions = {
            "possible_hemorrhage": (
                TreatmentAction.PAUSE,
                "Pause procedure to assess for vascular injury. "
                "Inspect surgical field, check instrument tip position, "
                "and verify hemostasis before continuing.",
            ),
            "hemodynamic_instability": (
                TreatmentAction.PAUSE,
                "Hemodynamic instability detected. Pause all robot motion. "
                "Communicate with anesthesia team for resuscitation. "
                "Assess for surgical cause (hemorrhage, pneumothorax).",
            ),
            "resection_boundary_concern": (
                TreatmentAction.ADJUST_APPROACH,
                "Resection plane may be too close to tumor. "
                "Consider widening margins or requesting intraoperative imaging "
                "to confirm margin adequacy.",
            ),
        }

        action_info = event_actions.get(event.event_type)
        if action_info is None:
            return None

        action, rationale = action_info
        return TreatmentRecommendation(
            action=action,
            severity=event.severity,
            rationale=rationale,
            supporting_evidence=event.contributing_anomalies,
            confidence=event.confidence,
            requires_confirmation=True,
            timestamp=time.time(),
        )

    def _handle_single_anomaly(self, anomaly: AnomalyEvent) -> Optional[TreatmentRecommendation]:
        """Generate recommendation from a single anomaly."""
        anomaly_actions = {
            "excessive_force": (
                TreatmentAction.REDUCE_FORCE,
                AlertSeverity.WARNING,
                "Applied force exceeds safe threshold. Reduce force or "
                "adjust approach angle to decrease tissue resistance.",
            ),
            "force_spike": (
                TreatmentAction.SLOW_DOWN,
                AlertSeverity.ADVISORY,
                "Sudden force change detected. Reduce speed and verify instrument tip position before continuing.",
            ),
            "tachycardia": (
                TreatmentAction.CONTINUE,
                AlertSeverity.ADVISORY,
                "Tachycardia detected. Monitor trend. Coordinate with anesthesia if persistent.",
            ),
            "hypotension": (
                TreatmentAction.SLOW_DOWN,
                AlertSeverity.WARNING,
                "Hypotension detected. Reduce procedural stimulation. "
                "Coordinate with anesthesia for volume resuscitation.",
            ),
            "desaturation": (
                TreatmentAction.PAUSE,
                AlertSeverity.CRITICAL,
                "Oxygen desaturation detected. Pause robot motion. "
                "Priority is airway management. Assess for "
                "pneumothorax if thoracic procedure.",
            ),
            "inadequate_margin": (
                TreatmentAction.REQUEST_IMAGING,
                AlertSeverity.WARNING,
                "Estimated surgical margin may be inadequate. Request "
                "updated intraoperative imaging or frozen section "
                "to confirm margin status.",
            ),
            "perfusion_change": (
                TreatmentAction.REQUEST_IMAGING,
                AlertSeverity.ADVISORY,
                "Tissue perfusion change detected. May indicate devascularization. Consider additional imaging.",
            ),
        }

        action_info = anomaly_actions.get(anomaly.anomaly_type)
        if action_info is None:
            return None

        action, severity, rationale = action_info
        return TreatmentRecommendation(
            action=action,
            severity=severity,
            rationale=rationale,
            supporting_evidence=[anomaly.description],
            confidence=anomaly.confidence,
            requires_confirmation=severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY),
            timestamp=time.time(),
        )

    def get_recommendation_history(self) -> list[dict]:
        """Get all recommendations as serializable dicts."""
        return [r.to_dict() for r in self._recommendation_history]


# =============================================================================
# SECTION 5: ADAPTIVE TREATMENT AGENT
# =============================================================================


class AdaptiveTreatmentAgent:
    """
    Real-time adaptive treatment agent for robotic oncology procedures.

    Continuously processes multi-modal data streams, detects anomalies,
    correlates cross-modal events, and generates treatment recommendations.

    AGENT LOOP:
    -----------
    1. Receive data samples from all streams
    2. Buffer and process each stream for anomalies
    3. Correlate anomalies across streams
    4. Generate treatment recommendations
    5. Present recommendations to surgical team
    6. Log all decisions for audit trail

    HUMAN-IN-THE-LOOP:
    ------------------
    All critical recommendations require explicit confirmation
    from the surgical team. The agent provides decision support,
    not autonomous control.
    """

    def __init__(
        self,
        patient_id: str,
        procedure_type: str,
        correlation_window_seconds: float = 10.0,
    ):
        self.patient_id = patient_id
        self.procedure_type = procedure_type
        self._start_time = time.time()

        # Stream buffers
        self._buffers: dict[StreamType, StreamBuffer] = {
            StreamType.FORCE_TORQUE: StreamBuffer(StreamType.FORCE_TORQUE, window_seconds=30.0),
            StreamType.PATIENT_VITALS: StreamBuffer(StreamType.PATIENT_VITALS, window_seconds=60.0),
            StreamType.IMAGING: StreamBuffer(StreamType.IMAGING, window_seconds=120.0),
            StreamType.ROBOT_STATE: StreamBuffer(StreamType.ROBOT_STATE, window_seconds=30.0),
        }

        # Processors
        self._force_processor = ForceTorqueProcessor()
        self._vitals_processor = VitalsProcessor()
        self._imaging_processor = ImagingProcessor()

        # Correlation and decision engines
        self._correlator = CrossModalCorrelator(correlation_window_seconds)
        self._decision_engine = TreatmentDecisionEngine()

        # State tracking
        self._total_samples_processed = 0
        self._total_anomalies_detected = 0
        self._total_recommendations = 0
        self._decision_log: list[dict] = []

        logger.info(f"AdaptiveTreatmentAgent initialized for patient {patient_id}, procedure: {procedure_type}")

    def ingest_sample(self, sample: DataSample) -> list[TreatmentRecommendation]:
        """
        Ingest a single data sample and return any recommendations.

        This is the main entry point for the agent processing loop.
        Called at the sampling rate of each data stream.

        Args:
            sample: Timestamped data sample from any stream

        Returns:
            List of treatment recommendations (may be empty)
        """
        self._total_samples_processed += 1

        # Buffer the sample
        buffer = self._buffers.get(sample.stream_type)
        if buffer is None:
            return []
        buffer.add_sample(sample)

        # Process for anomalies based on stream type
        anomalies = self._process_stream(sample.stream_type, buffer)
        self._total_anomalies_detected += len(anomalies)

        if not anomalies:
            return []

        # Correlate with other streams
        self._correlator.add_anomalies(anomalies)
        correlations = self._correlator.detect_correlations()

        # Generate recommendations
        recommendations = self._decision_engine.evaluate(anomalies, correlations)
        self._total_recommendations += len(recommendations)

        # Log decision
        if recommendations:
            self._decision_log.append(
                {
                    "timestamp": time.time(),
                    "elapsed_seconds": time.time() - self._start_time,
                    "anomalies": [{"type": a.anomaly_type, "severity": a.severity.value} for a in anomalies],
                    "correlations": [{"type": c.event_type, "severity": c.severity.value} for c in correlations],
                    "recommendations": [r.to_dict() for r in recommendations],
                }
            )

        return recommendations

    def _process_stream(self, stream_type: StreamType, buffer: StreamBuffer) -> list[AnomalyEvent]:
        """Route stream processing to appropriate processor."""
        processors = {
            StreamType.FORCE_TORQUE: self._force_processor,
            StreamType.PATIENT_VITALS: self._vitals_processor,
            StreamType.IMAGING: self._imaging_processor,
        }

        processor = processors.get(stream_type)
        if processor is None:
            return []

        return processor.process(buffer)

    def get_status(self) -> dict:
        """Get current agent status."""
        return {
            "patient_id": self.patient_id,
            "procedure_type": self.procedure_type,
            "elapsed_seconds": time.time() - self._start_time,
            "samples_processed": self._total_samples_processed,
            "anomalies_detected": self._total_anomalies_detected,
            "recommendations_generated": self._total_recommendations,
            "stream_buffer_sizes": {st.value: buf.sample_count for st, buf in self._buffers.items()},
        }

    def get_decision_log(self) -> list[dict]:
        """Get complete decision log for audit."""
        return self._decision_log


# =============================================================================
# SECTION 6: DEMO
# =============================================================================


def run_adaptive_treatment_demo():
    """
    Demonstrate the real-time adaptive treatment agent.

    Simulates a robotic procedure with multi-modal data streams
    including normal operation, force anomalies, vital sign changes,
    and correlated events.
    """
    logger.info("=" * 60)
    logger.info("REAL-TIME ADAPTIVE TREATMENT AGENT DEMO")
    logger.info("=" * 60)

    agent = AdaptiveTreatmentAgent(
        patient_id="PT-2026-0042",
        procedure_type="robotic_lobectomy",
        correlation_window_seconds=5.0,
    )

    print("\n--- Phase 1: Normal Operation ---")
    # Normal force/torque data
    for i in range(5):
        recs = agent.ingest_sample(
            DataSample(
                stream_type=StreamType.FORCE_TORQUE,
                timestamp=time.time(),
                values={"forces": [0.5, 0.3, 1.2], "torques": [0.05, 0.03, 0.02]},
            )
        )
        if recs:
            for r in recs:
                print(f"  [{r.severity.value.upper()}] {r.action.value}: {r.rationale}")

    # Normal vitals
    recs = agent.ingest_sample(
        DataSample(
            stream_type=StreamType.PATIENT_VITALS,
            timestamp=time.time(),
            values={"heart_rate_bpm": 75, "systolic_bp": 118, "spo2_percent": 98, "etco2_mmhg": 36},
        )
    )
    print(f"  Normal operation: {agent.get_status()['anomalies_detected']} anomalies")

    print("\n--- Phase 2: Force Anomaly During Dissection ---")
    # Sudden force increase
    recs = agent.ingest_sample(
        DataSample(
            stream_type=StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            values={"forces": [2.0, 1.5, 6.5], "torques": [0.1, 0.08, 0.05]},
        )
    )
    for r in recs:
        print(f"  [{r.severity.value.upper()}] {r.action.value}: {r.rationale}")

    print("\n--- Phase 3: Hemodynamic Response ---")
    # Vital sign changes following force event
    recs = agent.ingest_sample(
        DataSample(
            stream_type=StreamType.PATIENT_VITALS,
            timestamp=time.time(),
            values={"heart_rate_bpm": 125, "systolic_bp": 85, "spo2_percent": 96, "etco2_mmhg": 38},
        )
    )
    for r in recs:
        print(f"  [{r.severity.value.upper()}] {r.action.value}: {r.rationale}")

    print("\n--- Phase 4: Margin Concern From Imaging ---")
    # Imaging shows thin margin
    recs = agent.ingest_sample(
        DataSample(
            stream_type=StreamType.IMAGING,
            timestamp=time.time(),
            values={"estimated_margin_mm": 3.5, "confidence": 0.82},
        )
    )
    for r in recs:
        print(f"  [{r.severity.value.upper()}] {r.action.value}: {r.rationale}")

    print("\n--- Phase 5: Recovery to Normal ---")
    recs = agent.ingest_sample(
        DataSample(
            stream_type=StreamType.PATIENT_VITALS,
            timestamp=time.time(),
            values={"heart_rate_bpm": 80, "systolic_bp": 115, "spo2_percent": 98, "etco2_mmhg": 35},
        )
    )
    recs2 = agent.ingest_sample(
        DataSample(
            stream_type=StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            values={"forces": [0.8, 0.4, 1.5], "torques": [0.04, 0.03, 0.02]},
        )
    )
    total_recs = len(recs) + len(recs2)
    print(f"  Recovery: {total_recs} new recommendations (expect 0)")

    # Print summary
    print("\n" + "=" * 60)
    print("AGENT SESSION SUMMARY")
    print("=" * 60)
    status = agent.get_status()
    print(f"Samples processed: {status['samples_processed']}")
    print(f"Anomalies detected: {status['anomalies_detected']}")
    print(f"Recommendations generated: {status['recommendations_generated']}")
    print(f"Elapsed time: {status['elapsed_seconds']:.1f}s")

    print("\nDecision Log:")
    for entry in agent.get_decision_log():
        print(
            f"  t={entry['elapsed_seconds']:.1f}s: "
            f"{len(entry['anomalies'])} anomalies, "
            f"{len(entry['correlations'])} correlations, "
            f"{len(entry['recommendations'])} recommendations"
        )

    return status


if __name__ == "__main__":
    result = run_adaptive_treatment_demo()
    print(f"\nDemo result: {result}")
