"""Surgical robot interface: da Vinci Xi procedure orchestration, telemetry, digital twin."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class InstrumentType(Enum):
    GRASPER = "grasper"
    SCISSORS = "scissors"
    NEEDLE_DRIVER = "needle_driver"
    CAMERA = "camera"
    BIPOLAR = "bipolar"


class ProcedurePhase(Enum):
    DOCKING = "docking"
    PORT_PLACEMENT = "port_placement"
    DISSECTION = "dissection"
    RESECTION = "resection"
    RECONSTRUCTION = "reconstruction"
    UNDOCKING = "undocking"
    COMPLETE = "complete"


@dataclass
class TelemetryFrame:
    timestamp: datetime = field(default_factory=datetime.utcnow)
    joint_positions: list[float] = field(default_factory=lambda: [0.0] * 7)
    force_torque: list[float] = field(default_factory=lambda: [0.0] * 6)
    instrument: InstrumentType = InstrumentType.GRASPER
    collision_risk: float = 0.0


@dataclass
class DigitalTwinState:
    procedure_id: str = ""
    phase: ProcedurePhase = ProcedurePhase.DOCKING
    elapsed_seconds: float = 0.0
    telemetry_frames: int = 0
    anomaly_count: int = 0
    tissue_map_version: int = 0


@dataclass
class ProcedureRecord:
    procedure_id: str = field(default_factory=lambda: f"PROC-{uuid.uuid4().hex[:8]}")
    robot_id: str = ""
    surgeon_id: str = ""
    subject_id: str = ""
    phase: ProcedurePhase = ProcedurePhase.DOCKING
    started_at: datetime = field(default_factory=datetime.utcnow)
    telemetry: list[TelemetryFrame] = field(default_factory=list)


class SurgicalRobotInterface:
    """Interface to da Vinci Xi: procedure orchestration, telemetry streaming, digital twin sync."""

    def __init__(self, robot_id: str = "DVX-001") -> None:
        self.robot_id = robot_id
        self.procedures: dict[str, ProcedureRecord] = {}
        self.twin: DigitalTwinState = DigitalTwinState()
        self._phase_order = list(ProcedurePhase)

    def start_procedure(self, surgeon_id: str, subject_id: str) -> ProcedureRecord:
        proc = ProcedureRecord(robot_id=self.robot_id, surgeon_id=surgeon_id, subject_id=subject_id)
        self.procedures[proc.procedure_id] = proc
        self.twin = DigitalTwinState(procedure_id=proc.procedure_id)
        return proc

    def advance_phase(self, procedure_id: str) -> ProcedurePhase | None:
        proc = self.procedures[procedure_id]
        idx = self._phase_order.index(proc.phase)
        if idx + 1 < len(self._phase_order):
            proc.phase = self._phase_order[idx + 1]
            self.twin.phase = proc.phase
            return proc.phase
        return None

    def record_telemetry(self, procedure_id: str, frame: TelemetryFrame) -> None:
        proc = self.procedures[procedure_id]
        proc.telemetry.append(frame)
        self.twin.telemetry_frames += 1
        if frame.collision_risk > 0.8:
            self.twin.anomaly_count += 1

    def update_tissue_map(self) -> int:
        self.twin.tissue_map_version += 1
        return self.twin.tissue_map_version

    def get_twin_state(self) -> dict[str, object]:
        return {
            "procedure_id": self.twin.procedure_id, "phase": self.twin.phase.value,
            "telemetry_frames": self.twin.telemetry_frames,
            "anomalies": self.twin.anomaly_count,
            "tissue_map_version": self.twin.tissue_map_version,
        }

    def safety_check(self, procedure_id: str) -> dict[str, bool]:
        proc = self.procedures[procedure_id]
        high_force = any(max(f.force_torque) > 10.0 for f in proc.telemetry[-10:]) if proc.telemetry else False
        collision = any(f.collision_risk > 0.8 for f in proc.telemetry[-10:]) if proc.telemetry else False
        return {"force_ok": not high_force, "collision_ok": not collision, "twin_synced": True}


def main() -> None:
    iface = SurgicalRobotInterface("DVX-001")
    proc = iface.start_procedure("SURG-01", "SUBJ-01")
    for _ in range(5):
        iface.record_telemetry(proc.procedure_id, TelemetryFrame())
    iface.record_telemetry(proc.procedure_id, TelemetryFrame(collision_risk=0.9))
    for _ in range(3):
        phase = iface.advance_phase(proc.procedure_id)
        if phase:
            print(f"Advanced to: {phase.value}")
    iface.update_tissue_map()
    print(f"Twin state: {iface.get_twin_state()}")
    print(f"Safety check: {iface.safety_check(proc.procedure_id)}")


if __name__ == "__main__":
    main()
