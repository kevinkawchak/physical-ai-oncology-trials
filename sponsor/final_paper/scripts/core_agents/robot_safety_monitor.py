"""Robot safety monitor: robotic procedure safety, telemetry anomaly detection, emergency stop."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RobotState(Enum):
    IDLE = "idle"
    CALIBRATING = "calibrating"
    OPERATING = "operating"
    PAUSED = "paused"
    ESTOP = "emergency_stop"
    FAULT = "fault"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ESTOP = "emergency_stop"


@dataclass
class TelemetryReading:
    timestamp: datetime
    joint_torques_nm: list[float] = field(default_factory=list)
    end_effector_force_n: float = 0.0
    position_error_mm: float = 0.0
    velocity_mps: float = 0.0
    temperature_c: float = 25.0


@dataclass
class SafetyAlert:
    alert_id: str = field(default_factory=lambda: f"ALR-{uuid.uuid4().hex[:8]}")
    severity: AlertSeverity = AlertSeverity.INFO
    source: str = ""
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False


@dataclass
class ProcedureLog:
    procedure_id: str
    robot_id: str
    subject_id: str
    start_time: datetime
    end_time: datetime | None = None
    state: RobotState = RobotState.IDLE
    alerts: list[SafetyAlert] = field(default_factory=list)
    telemetry: list[TelemetryReading] = field(default_factory=list)


SAFETY_THRESHOLDS: dict[str, float] = {
    "max_force_n": 50.0,
    "max_torque_nm": 40.0,
    "max_position_error_mm": 2.0,
    "max_velocity_mps": 0.25,
    "max_temperature_c": 55.0,
    "ewma_alpha": 0.3,
}


class RobotSafetyMonitor:
    """Monitors robotic procedure safety with telemetry anomaly detection and e-stop logic."""

    def __init__(self, robot_id: str, thresholds: dict[str, float] | None = None) -> None:
        self.robot_id = robot_id
        self.thresholds = thresholds or dict(SAFETY_THRESHOLDS)
        self.procedures: list[ProcedureLog] = []
        self.alerts: list[SafetyAlert] = []
        self._ewma_force: float = 0.0

    def start_procedure(self, procedure_id: str, subject_id: str) -> ProcedureLog:
        log = ProcedureLog(procedure_id=procedure_id, robot_id=self.robot_id,
                           subject_id=subject_id, start_time=datetime.utcnow(), state=RobotState.OPERATING)
        self.procedures.append(log)
        return log

    def ingest_telemetry(self, procedure_id: str, reading: TelemetryReading) -> list[SafetyAlert]:
        proc = self._find_procedure(procedure_id)
        proc.telemetry.append(reading)
        alpha = self.thresholds["ewma_alpha"]
        self._ewma_force = alpha * reading.end_effector_force_n + (1 - alpha) * self._ewma_force
        return self._check_limits(proc, reading)

    def _check_limits(self, proc: ProcedureLog, reading: TelemetryReading) -> list[SafetyAlert]:
        new_alerts: list[SafetyAlert] = []
        checks = [
            (reading.end_effector_force_n, "max_force_n", "Force exceeded"),
            (reading.position_error_mm, "max_position_error_mm", "Position error exceeded"),
            (reading.velocity_mps, "max_velocity_mps", "Velocity limit exceeded"),
            (reading.temperature_c, "max_temperature_c", "Temperature exceeded"),
        ]
        for value, key, msg in checks:
            limit = self.thresholds[key]
            if value > limit:
                severity = AlertSeverity.ESTOP if value > limit * 1.5 else AlertSeverity.CRITICAL
                alert = SafetyAlert(severity=severity, source=self.robot_id, message=f"{msg}: {value:.2f} > {limit}")
                new_alerts.append(alert)
        for torque in reading.joint_torques_nm:
            if abs(torque) > self.thresholds["max_torque_nm"]:
                alert = SafetyAlert(severity=AlertSeverity.CRITICAL, source=self.robot_id,
                                    message=f"Joint torque {torque:.1f} Nm exceeds limit")
                new_alerts.append(alert)
        if new_alerts:
            proc.alerts.extend(new_alerts)
            self.alerts.extend(new_alerts)
            if any(a.severity == AlertSeverity.ESTOP for a in new_alerts):
                self.trigger_estop(proc.procedure_id)
        return new_alerts

    def trigger_estop(self, procedure_id: str) -> None:
        proc = self._find_procedure(procedure_id)
        proc.state = RobotState.ESTOP
        proc.end_time = datetime.utcnow()

    def anomaly_score(self, procedure_id: str) -> float:
        proc = self._find_procedure(procedure_id)
        if len(proc.telemetry) < 2:
            return 0.0
        forces = [t.end_effector_force_n for t in proc.telemetry]
        mean_f = sum(forces) / len(forces)
        std_f = math.sqrt(sum((f - mean_f) ** 2 for f in forces) / len(forces))
        z = abs(forces[-1] - mean_f) / max(std_f, 1e-9)
        return min(z / 3.0, 1.0)

    def procedure_summary(self, procedure_id: str) -> dict[str, object]:
        proc = self._find_procedure(procedure_id)
        crit = sum(1 for a in proc.alerts if a.severity in (AlertSeverity.CRITICAL, AlertSeverity.ESTOP))
        return {"procedure_id": proc.procedure_id, "state": proc.state.value,
                "telemetry_points": len(proc.telemetry), "alerts": len(proc.alerts),
                "critical_alerts": crit, "anomaly_score": round(self.anomaly_score(procedure_id), 3)}

    def _find_procedure(self, procedure_id: str) -> ProcedureLog:
        for p in self.procedures:
            if p.procedure_id == procedure_id:
                return p
        msg = f"Procedure {procedure_id} not found"
        raise KeyError(msg)


def main() -> None:
    monitor = RobotSafetyMonitor("FRANKA-001")
    proc = monitor.start_procedure("PROC-001", "SUBJ-01")
    for i in range(5):
        reading = TelemetryReading(
            timestamp=datetime.utcnow(), joint_torques_nm=[10.0 + i, 15.0, 12.0],
            end_effector_force_n=20.0 + i * 8, position_error_mm=0.5 + i * 0.3, velocity_mps=0.1,
        )
        alerts = monitor.ingest_telemetry(proc.procedure_id, reading)
        if alerts:
            print(f"  Step {i}: {len(alerts)} alert(s) - {alerts[0].message}")
    print(f"Summary: {monitor.procedure_summary(proc.procedure_id)}")


if __name__ == "__main__":
    main()
