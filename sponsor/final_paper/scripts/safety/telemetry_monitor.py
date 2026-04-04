"""Continuous telemetry monitoring for all 10 robot categories.

Tracks force/torque, position deviation, and communication status with
graduated warnings (advisory, caution, halt). Supports category-specific
thresholds for surgical robots, cobots, RT systems, and all other types.

Usage:
    python telemetry_monitor.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class AlarmLevel(Enum):
    """Graduated alarm levels for telemetry monitoring."""

    NOMINAL = "nominal"
    ADVISORY = "advisory"
    CAUTION = "caution"
    HALT = "halt"


class RobotCategory(Enum):
    """Ten robot categories in the Physical AI trial site."""

    SURGICAL = "surgical_robot"
    COBOT = "collaborative_robot"
    RT_POSITIONING = "rt_positioning"
    NEEDLE_PLACEMENT = "needle_placement"
    SOCIAL_COMPANION = "social_companion"
    REHAB_EXOSKELETON = "rehab_exoskeleton"
    RT_MOTION_TRACKING = "rt_motion_tracking"
    IMAGING_ASSISTANT = "imaging_assistant"
    TRANSPORT_LOGISTICS = "transport_logistics"
    ENVIRONMENTAL_MONITOR = "environmental_monitor"


@dataclass
class CategoryThresholds:
    """Category-specific safety thresholds for telemetry monitoring."""

    force_advisory_n: float = 5.0
    force_caution_n: float = 15.0
    force_halt_n: float = 25.0
    position_advisory_mm: float = 1.0
    position_caution_mm: float = 3.0
    position_halt_mm: float = 5.0
    comm_timeout_advisory_ms: int = 100
    comm_timeout_caution_ms: int = 500
    comm_timeout_halt_ms: int = 1000


CATEGORY_THRESHOLDS: dict[str, CategoryThresholds] = {
    RobotCategory.SURGICAL.value: CategoryThresholds(
        force_advisory_n=3.0,
        force_caution_n=8.0,
        force_halt_n=15.0,
        position_advisory_mm=0.5,
        position_caution_mm=1.5,
        position_halt_mm=3.0,
        comm_timeout_advisory_ms=50,
        comm_timeout_caution_ms=200,
        comm_timeout_halt_ms=500,
    ),
    RobotCategory.COBOT.value: CategoryThresholds(
        force_advisory_n=5.0,
        force_caution_n=15.0,
        force_halt_n=25.0,
        position_advisory_mm=1.0,
        position_caution_mm=3.0,
        position_halt_mm=5.0,
    ),
    RobotCategory.RT_POSITIONING.value: CategoryThresholds(
        position_advisory_mm=0.3,
        position_caution_mm=1.0,
        position_halt_mm=2.0,
    ),
    RobotCategory.NEEDLE_PLACEMENT.value: CategoryThresholds(
        force_advisory_n=2.0,
        force_caution_n=5.0,
        force_halt_n=10.0,
        position_advisory_mm=0.2,
        position_caution_mm=0.8,
        position_halt_mm=1.5,
    ),
    RobotCategory.SOCIAL_COMPANION.value: CategoryThresholds(
        force_advisory_n=10.0,
        force_caution_n=20.0,
        force_halt_n=30.0,
        position_advisory_mm=5.0,
        position_caution_mm=15.0,
        position_halt_mm=25.0,
    ),
    RobotCategory.REHAB_EXOSKELETON.value: CategoryThresholds(
        force_advisory_n=8.0,
        force_caution_n=20.0,
        force_halt_n=35.0,
        position_advisory_mm=2.0,
        position_caution_mm=5.0,
        position_halt_mm=10.0,
    ),
    RobotCategory.RT_MOTION_TRACKING.value: CategoryThresholds(
        position_advisory_mm=0.5,
        position_caution_mm=1.5,
        position_halt_mm=3.0,
    ),
    RobotCategory.IMAGING_ASSISTANT.value: CategoryThresholds(
        position_advisory_mm=1.0,
        position_caution_mm=3.0,
        position_halt_mm=5.0,
    ),
    RobotCategory.TRANSPORT_LOGISTICS.value: CategoryThresholds(
        force_advisory_n=15.0,
        force_caution_n=30.0,
        force_halt_n=50.0,
        position_advisory_mm=10.0,
        position_caution_mm=25.0,
        position_halt_mm=50.0,
    ),
    RobotCategory.ENVIRONMENTAL_MONITOR.value: CategoryThresholds(
        force_advisory_n=20.0,
        force_caution_n=40.0,
        force_halt_n=60.0,
        position_advisory_mm=15.0,
        position_caution_mm=30.0,
        position_halt_mm=50.0,
    ),
}


@dataclass
class TelemetryReading:
    """Single telemetry reading from a robot sensor."""

    robot_id: str
    category: str
    timestamp: str
    force_n: float = 0.0
    position_deviation_mm: float = 0.0
    comm_latency_ms: int = 10
    temperature_c: float = 21.0
    battery_pct: float = 100.0
    error_codes: list[str] = field(default_factory=list)


@dataclass
class AlarmEvent:
    """Alarm event generated from telemetry analysis."""

    robot_id: str
    category: str
    timestamp: str
    alarm_level: str
    trigger_source: str
    measured_value: float
    threshold_value: float
    recommended_action: str


class TelemetryMonitor:
    """Continuous telemetry monitor for all robot categories."""

    def __init__(self) -> None:
        self.alarm_history: list[AlarmEvent] = []
        self.reading_count = 0

    def _get_thresholds(self, category: str) -> CategoryThresholds:
        """Get category-specific thresholds, falling back to defaults."""
        return CATEGORY_THRESHOLDS.get(category, CategoryThresholds())

    def evaluate_reading(self, reading: TelemetryReading) -> list[AlarmEvent]:
        """Evaluate a telemetry reading against category-specific thresholds."""
        self.reading_count += 1
        thresholds = self._get_thresholds(reading.category)
        alarms: list[AlarmEvent] = []

        if reading.force_n >= thresholds.force_halt_n:
            alarms.append(
                AlarmEvent(
                    robot_id=reading.robot_id,
                    category=reading.category,
                    timestamp=reading.timestamp,
                    alarm_level=AlarmLevel.HALT.value,
                    trigger_source="force",
                    measured_value=reading.force_n,
                    threshold_value=thresholds.force_halt_n,
                    recommended_action="Immediate halt - force limit exceeded",
                )
            )
        elif reading.force_n >= thresholds.force_caution_n:
            alarms.append(
                AlarmEvent(
                    robot_id=reading.robot_id,
                    category=reading.category,
                    timestamp=reading.timestamp,
                    alarm_level=AlarmLevel.CAUTION.value,
                    trigger_source="force",
                    measured_value=reading.force_n,
                    threshold_value=thresholds.force_caution_n,
                    recommended_action="Reduce speed, notify clinician",
                )
            )
        elif reading.force_n >= thresholds.force_advisory_n:
            alarms.append(
                AlarmEvent(
                    robot_id=reading.robot_id,
                    category=reading.category,
                    timestamp=reading.timestamp,
                    alarm_level=AlarmLevel.ADVISORY.value,
                    trigger_source="force",
                    measured_value=reading.force_n,
                    threshold_value=thresholds.force_advisory_n,
                    recommended_action="Log and monitor - force approaching limit",
                )
            )

        if reading.position_deviation_mm >= thresholds.position_halt_mm:
            alarms.append(
                AlarmEvent(
                    robot_id=reading.robot_id,
                    category=reading.category,
                    timestamp=reading.timestamp,
                    alarm_level=AlarmLevel.HALT.value,
                    trigger_source="position",
                    measured_value=reading.position_deviation_mm,
                    threshold_value=thresholds.position_halt_mm,
                    recommended_action="Immediate halt - position deviation critical",
                )
            )
        elif reading.position_deviation_mm >= thresholds.position_caution_mm:
            alarms.append(
                AlarmEvent(
                    robot_id=reading.robot_id,
                    category=reading.category,
                    timestamp=reading.timestamp,
                    alarm_level=AlarmLevel.CAUTION.value,
                    trigger_source="position",
                    measured_value=reading.position_deviation_mm,
                    threshold_value=thresholds.position_caution_mm,
                    recommended_action="Pause and recalibrate if deviation persists",
                )
            )

        if reading.comm_latency_ms >= thresholds.comm_timeout_halt_ms:
            alarms.append(
                AlarmEvent(
                    robot_id=reading.robot_id,
                    category=reading.category,
                    timestamp=reading.timestamp,
                    alarm_level=AlarmLevel.HALT.value,
                    trigger_source="communication",
                    measured_value=float(reading.comm_latency_ms),
                    threshold_value=float(thresholds.comm_timeout_halt_ms),
                    recommended_action="Communication lost - initiate safe stop",
                )
            )

        self.alarm_history.extend(alarms)
        return alarms

    def get_summary(self) -> dict:
        """Return monitoring summary statistics."""
        level_counts: dict[str, int] = {}
        for alarm in self.alarm_history:
            level_counts[alarm.alarm_level] = level_counts.get(alarm.alarm_level, 0) + 1
        return {
            "total_readings": self.reading_count,
            "total_alarms": len(self.alarm_history),
            "alarms_by_level": level_counts,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }


def main() -> None:
    """Demonstrate telemetry monitoring with sample readings."""
    monitor = TelemetryMonitor()
    sample_readings = [
        TelemetryReading("SURG-01", RobotCategory.SURGICAL.value, "09:15:00", force_n=2.0, position_deviation_mm=0.3),
        TelemetryReading("SURG-01", RobotCategory.SURGICAL.value, "09:15:05", force_n=7.5, position_deviation_mm=0.8),
        TelemetryReading("COBOT-02", RobotCategory.COBOT.value, "09:20:00", force_n=4.0, position_deviation_mm=0.5),
        TelemetryReading("NEEDLE-01", RobotCategory.NEEDLE_PLACEMENT.value, "09:25:00", force_n=1.5),
        TelemetryReading("TRACK-01", RobotCategory.RT_MOTION_TRACKING.value, "09:30:00", position_deviation_mm=0.4),
        TelemetryReading("COMPN-03", RobotCategory.SOCIAL_COMPANION.value, "09:35:00", force_n=2.0),
    ]
    for reading in sample_readings:
        alarms = monitor.evaluate_reading(reading)
        if alarms:
            for alarm in alarms:
                print(
                    f"[{alarm.alarm_level.upper()}] {alarm.robot_id}: "
                    f"{alarm.trigger_source} = {alarm.measured_value} (threshold: {alarm.threshold_value})"
                )
    summary = monitor.get_summary()
    print("\nMonitoring Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
