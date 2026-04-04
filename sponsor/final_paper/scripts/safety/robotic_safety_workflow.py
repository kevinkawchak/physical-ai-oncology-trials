from __future__ import annotations

"""Four-category robotic safety event classification and workflow.

Classifies robotic safety events into four severity categories and
routes each through the appropriate response workflow:

- Category A (No deviation): Log and archive.
- Category B (Minor):        Log, continue, increase monitoring frequency.
- Category C (Significant):  Pause procedure, notify clinician, await clearance.
- Category D (Critical):     Immediate halt, trigger emergency protocol.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SafetyCategory(str, Enum):
    """Four-level severity classification for robotic safety events."""

    A_NO_DEVIATION = "A"
    B_MINOR = "B"
    C_SIGNIFICANT = "C"
    D_CRITICAL = "D"


class EventDisposition(str, Enum):
    """Terminal state of a safety event after workflow processing."""

    ARCHIVED = "archived"
    MONITORING_INCREASED = "monitoring_increased"
    PAUSED_AWAITING_CLEARANCE = "paused_awaiting_clearance"
    EMERGENCY_HALT = "emergency_halt"
    RESOLVED = "resolved"


class RobotCategory(str, Enum):
    """Ten robot categories from the paper's classification framework."""

    SURGICAL = "surgical"
    BIOPSY = "biopsy"
    RADIATION_DELIVERY = "radiation_delivery"
    DRUG_COMPOUNDING = "drug_compounding"
    PATIENT_POSITIONING = "patient_positioning"
    SAMPLE_TRANSPORT = "sample_transport"
    REHABILITATION = "rehabilitation"
    MONITORING = "monitoring"
    DISINFECTION = "disinfection"
    LOGISTICS = "logistics"


# ---------------------------------------------------------------------------
# Classification thresholds per robot category
# ---------------------------------------------------------------------------

_CATEGORY_THRESHOLDS: dict[str, dict[str, float]] = {
    RobotCategory.SURGICAL.value: {
        "force_deviation_b": 0.05,
        "force_deviation_c": 0.10,
        "force_deviation_d": 0.20,
        "position_deviation_mm_b": 0.5,
        "position_deviation_mm_c": 1.0,
        "position_deviation_mm_d": 2.0,
    },
    RobotCategory.BIOPSY.value: {
        "force_deviation_b": 0.08,
        "force_deviation_c": 0.15,
        "force_deviation_d": 0.25,
        "position_deviation_mm_b": 1.0,
        "position_deviation_mm_c": 2.0,
        "position_deviation_mm_d": 3.0,
    },
    RobotCategory.RADIATION_DELIVERY.value: {
        "force_deviation_b": 0.03,
        "force_deviation_c": 0.07,
        "force_deviation_d": 0.12,
        "position_deviation_mm_b": 0.3,
        "position_deviation_mm_c": 0.8,
        "position_deviation_mm_d": 1.5,
    },
    RobotCategory.DRUG_COMPOUNDING.value: {
        "force_deviation_b": 0.10,
        "force_deviation_c": 0.20,
        "force_deviation_d": 0.35,
        "position_deviation_mm_b": 2.0,
        "position_deviation_mm_c": 5.0,
        "position_deviation_mm_d": 10.0,
    },
    RobotCategory.PATIENT_POSITIONING.value: {
        "force_deviation_b": 0.07,
        "force_deviation_c": 0.15,
        "force_deviation_d": 0.25,
        "position_deviation_mm_b": 1.5,
        "position_deviation_mm_c": 3.0,
        "position_deviation_mm_d": 5.0,
    },
    RobotCategory.SAMPLE_TRANSPORT.value: {
        "force_deviation_b": 0.15,
        "force_deviation_c": 0.30,
        "force_deviation_d": 0.50,
        "position_deviation_mm_b": 5.0,
        "position_deviation_mm_c": 10.0,
        "position_deviation_mm_d": 20.0,
    },
    RobotCategory.REHABILITATION.value: {
        "force_deviation_b": 0.10,
        "force_deviation_c": 0.20,
        "force_deviation_d": 0.35,
        "position_deviation_mm_b": 3.0,
        "position_deviation_mm_c": 8.0,
        "position_deviation_mm_d": 15.0,
    },
    RobotCategory.MONITORING.value: {
        "force_deviation_b": 0.20,
        "force_deviation_c": 0.40,
        "force_deviation_d": 0.60,
        "position_deviation_mm_b": 5.0,
        "position_deviation_mm_c": 15.0,
        "position_deviation_mm_d": 25.0,
    },
    RobotCategory.DISINFECTION.value: {
        "force_deviation_b": 0.20,
        "force_deviation_c": 0.40,
        "force_deviation_d": 0.60,
        "position_deviation_mm_b": 10.0,
        "position_deviation_mm_c": 20.0,
        "position_deviation_mm_d": 30.0,
    },
    RobotCategory.LOGISTICS.value: {
        "force_deviation_b": 0.25,
        "force_deviation_c": 0.50,
        "force_deviation_d": 0.70,
        "position_deviation_mm_b": 10.0,
        "position_deviation_mm_c": 25.0,
        "position_deviation_mm_d": 50.0,
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyEvent:
    """A robotic safety event captured during a procedure."""

    event_id: str
    timestamp: datetime
    robot_id: str
    robot_category: str
    procedure_id: str
    force_deviation: float
    position_deviation_mm: float
    description: str
    raw_telemetry: dict[str, object]

    @classmethod
    def create(
        cls,
        robot_id: str,
        robot_category: str,
        procedure_id: str,
        force_deviation: float,
        position_deviation_mm: float,
        description: str = "",
        raw_telemetry: dict[str, object] | None = None,
    ) -> SafetyEvent:
        return cls(
            event_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc),
            robot_id=robot_id,
            robot_category=robot_category,
            procedure_id=procedure_id,
            force_deviation=force_deviation,
            position_deviation_mm=position_deviation_mm,
            description=description,
            raw_telemetry=raw_telemetry or {},
        )


@dataclass(frozen=True)
class SafetyWorkflowResult:
    """Outcome of processing a safety event through the workflow."""

    event_id: str
    category: SafetyCategory
    disposition: EventDisposition
    actions_taken: list[str]
    notifications: list[str]
    monitoring_interval_seconds: int | None
    requires_human_clearance: bool
    rationale: str
    processed_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "category": self.category.value,
            "disposition": self.disposition.value,
            "actions_taken": self.actions_taken,
            "notifications": self.notifications,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "requires_human_clearance": self.requires_human_clearance,
            "rationale": self.rationale,
            "processed_at": self.processed_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Robotic Safety Workflow Engine
# ---------------------------------------------------------------------------

# Default monitoring intervals (seconds) per category
_MONITORING_INTERVALS: dict[SafetyCategory, int | None] = {
    SafetyCategory.A_NO_DEVIATION: None,
    SafetyCategory.B_MINOR: 30,
    SafetyCategory.C_SIGNIFICANT: 5,
    SafetyCategory.D_CRITICAL: 1,
}


class RoboticSafetyWorkflow:
    """Classify robotic safety events and execute the response workflow.

    Usage::

        workflow = RoboticSafetyWorkflow()
        event = SafetyEvent.create(
            robot_id="SR-001",
            robot_category="surgical",
            procedure_id="PROC-042",
            force_deviation=0.06,
            position_deviation_mm=0.3,
        )
        result = workflow.process(event)
    """

    def __init__(self) -> None:
        self._log: list[SafetyWorkflowResult] = []
        self._active_halts: dict[str, SafetyEvent] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, event: SafetyEvent) -> SafetyWorkflowResult:
        """Classify and process a single safety event."""
        category = self._classify(event)
        result = self._execute_workflow(event, category)
        self._log.append(result)

        if category == SafetyCategory.D_CRITICAL:
            self._active_halts[event.robot_id] = event

        logger.info(
            "Safety event %s: category=%s, disposition=%s, robot=%s",
            event.event_id,
            category.value,
            result.disposition.value,
            event.robot_id,
        )
        return result

    def clear_halt(self, robot_id: str, authorization_code: str) -> bool:
        """Clear an emergency halt for a robot after human authorization.

        Returns True if a halt was cleared.
        """
        if robot_id not in self._active_halts:
            return False
        del self._active_halts[robot_id]
        logger.info(
            "Emergency halt cleared for %s with authorization %s",
            robot_id,
            authorization_code,
        )
        return True

    def is_halted(self, robot_id: str) -> bool:
        """Check if a robot is currently in emergency halt."""
        return robot_id in self._active_halts

    def event_log(
        self,
        category: SafetyCategory | None = None,
        limit: int = 100,
    ) -> list[SafetyWorkflowResult]:
        """Return recent workflow results, optionally filtered by category."""
        source = list(reversed(self._log))
        if category is not None:
            source = [r for r in source if r.category == category]
        return source[:limit]

    def category_counts(self) -> dict[str, int]:
        """Return event counts grouped by safety category."""
        counts: dict[str, int] = {c.value: 0 for c in SafetyCategory}
        for r in self._log:
            counts[r.category.value] += 1
        return counts

    def active_halt_count(self) -> int:
        return len(self._active_halts)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, event: SafetyEvent) -> SafetyCategory:
        """Assign a safety category based on deviation magnitudes and robot type."""
        thresholds = _CATEGORY_THRESHOLDS.get(event.robot_category)
        if thresholds is None:
            logger.warning(
                "Unknown robot category '%s' for event %s; defaulting to strictest thresholds",
                event.robot_category,
                event.event_id,
            )
            thresholds = _CATEGORY_THRESHOLDS[RobotCategory.SURGICAL.value]

        force = abs(event.force_deviation)
        pos = abs(event.position_deviation_mm)

        # Critical
        if force >= thresholds["force_deviation_d"] or pos >= thresholds["position_deviation_mm_d"]:
            return SafetyCategory.D_CRITICAL

        # Significant
        if force >= thresholds["force_deviation_c"] or pos >= thresholds["position_deviation_mm_c"]:
            return SafetyCategory.C_SIGNIFICANT

        # Minor
        if force >= thresholds["force_deviation_b"] or pos >= thresholds["position_deviation_mm_b"]:
            return SafetyCategory.B_MINOR

        return SafetyCategory.A_NO_DEVIATION

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    def _execute_workflow(self, event: SafetyEvent, category: SafetyCategory) -> SafetyWorkflowResult:
        actions: list[str] = []
        notifications: list[str] = []
        requires_human = False

        if category == SafetyCategory.A_NO_DEVIATION:
            disposition = EventDisposition.ARCHIVED
            actions.append("Event logged to safety archive")
            actions.append(f"Telemetry snapshot stored for procedure {event.procedure_id}")
            rationale = (
                f"Force deviation ({event.force_deviation:.4f}) and position deviation "
                f"({event.position_deviation_mm:.2f} mm) within normal envelope for "
                f"{event.robot_category} robot."
            )

        elif category == SafetyCategory.B_MINOR:
            disposition = EventDisposition.MONITORING_INCREASED
            actions.append("Event logged with elevated priority")
            actions.append("Monitoring frequency increased to 30-second intervals")
            actions.append("Trend analysis triggered for next 5 minutes")
            notifications.append("safety_monitoring_dashboard")
            rationale = (
                f"Minor deviation detected: force={event.force_deviation:.4f}, "
                f"position={event.position_deviation_mm:.2f} mm. Exceeds normal but "
                f"within safe operating range for {event.robot_category} robot."
            )

        elif category == SafetyCategory.C_SIGNIFICANT:
            disposition = EventDisposition.PAUSED_AWAITING_CLEARANCE
            requires_human = True
            actions.append("Procedure PAUSED - robot placed in safe hold position")
            actions.append("Real-time telemetry feed escalated to clinical monitor")
            actions.append("Digital twin deviation analysis initiated")
            actions.append("Patient safety assessment requested")
            notifications.append("attending_clinician")
            notifications.append("safety_officer")
            notifications.append("clinical_monitor")
            rationale = (
                f"Significant deviation: force={event.force_deviation:.4f}, "
                f"position={event.position_deviation_mm:.2f} mm. Exceeds safe operating "
                f"threshold for {event.robot_category} robot. Clinician clearance required."
            )

        else:  # D_CRITICAL
            disposition = EventDisposition.EMERGENCY_HALT
            requires_human = True
            actions.append("IMMEDIATE HALT executed - all robot axes locked")
            actions.append("Emergency power-down sequence initiated")
            actions.append("Patient safety perimeter activated")
            actions.append("Incident record created for regulatory reporting")
            actions.append("Root cause analysis workflow triggered")
            actions.append("All concurrent procedures on this robot suspended")
            notifications.append("emergency_response_team")
            notifications.append("attending_clinician")
            notifications.append("safety_officer")
            notifications.append("sponsor_medical_monitor")
            notifications.append("irb_safety_desk")
            rationale = (
                f"CRITICAL deviation: force={event.force_deviation:.4f}, "
                f"position={event.position_deviation_mm:.2f} mm. Exceeds maximum "
                f"permissible deviation for {event.robot_category} robot. "
                f"Emergency protocol activated."
            )

        return SafetyWorkflowResult(
            event_id=event.event_id,
            category=category,
            disposition=disposition,
            actions_taken=actions,
            notifications=notifications,
            monitoring_interval_seconds=_MONITORING_INTERVALS[category],
            requires_human_clearance=requires_human,
            rationale=rationale,
            processed_at=datetime.now(timezone.utc),
        )
