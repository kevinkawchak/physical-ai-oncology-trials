"""Cobot interface: Franka Emika lab automation, pharmacy dosing, calibration workflows."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskType(Enum):
    LAB_PIPETTING = "lab_pipetting"
    PHARMACY_DOSING = "pharmacy_dosing"
    SAMPLE_TRANSPORT = "sample_transport"
    RACK_LOADING = "rack_loading"
    CALIBRATION = "calibration"


class CobotState(Enum):
    IDLE = "idle"
    EXECUTING = "executing"
    PAUSED = "paused"
    CALIBRATING = "calibrating"
    ERROR = "error"


@dataclass
class CalibrationRecord:
    cal_id: str = field(default_factory=lambda: f"CAL-{uuid.uuid4().hex[:6]}")
    position_error_mm: float = 0.0
    force_error_n: float = 0.0
    passed: bool = True
    performed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DosingRecord:
    dose_id: str = field(default_factory=lambda: f"DOSE-{uuid.uuid4().hex[:6]}")
    subject_id: str = ""
    drug_code: str = ""
    target_mg: float = 0.0
    actual_mg: float = 0.0
    deviation_pct: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CobotTask:
    task_id: str = field(default_factory=lambda: f"CT-{uuid.uuid4().hex[:6]}")
    task_type: TaskType = TaskType.LAB_PIPETTING
    cobot_id: str = ""
    status: CobotState = CobotState.IDLE
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str = ""


class CobotInterface:
    """Interface to Franka Emika Panda: lab automation, pharmacy dosing, calibration."""

    MAX_DOSE_DEVIATION_PCT: float = 2.0

    def __init__(self, cobot_id: str = "FEP-001") -> None:
        self.cobot_id = cobot_id
        self.state = CobotState.IDLE
        self.tasks: list[CobotTask] = []
        self.calibrations: list[CalibrationRecord] = []
        self.dosing_log: list[DosingRecord] = []

    def run_calibration(self, position_error: float = 0.05, force_error: float = 0.1) -> CalibrationRecord:
        self.state = CobotState.CALIBRATING
        passed = position_error < 0.1 and force_error < 0.5
        rec = CalibrationRecord(position_error_mm=position_error, force_error_n=force_error, passed=passed)
        self.calibrations.append(rec)
        self.state = CobotState.IDLE if passed else CobotState.ERROR
        return rec

    def execute_task(self, task_type: TaskType) -> CobotTask:
        if self.state != CobotState.IDLE:
            task = CobotTask(task_type=task_type, cobot_id=self.cobot_id, status=CobotState.ERROR,
                             error_message="Cobot not idle")
            self.tasks.append(task)
            return task
        task = CobotTask(task_type=task_type, cobot_id=self.cobot_id, status=CobotState.EXECUTING,
                         started_at=datetime.utcnow())
        self.state = CobotState.EXECUTING
        task.status = CobotState.IDLE
        task.completed_at = datetime.utcnow()
        self.state = CobotState.IDLE
        self.tasks.append(task)
        return task

    def prepare_dose(self, subject_id: str, drug_code: str, target_mg: float,
                     actual_mg: float) -> DosingRecord:
        deviation = abs(actual_mg - target_mg) / target_mg * 100 if target_mg > 0 else 0.0
        rec = DosingRecord(subject_id=subject_id, drug_code=drug_code, target_mg=target_mg,
                           actual_mg=actual_mg, deviation_pct=round(deviation, 2))
        self.dosing_log.append(rec)
        if deviation > self.MAX_DOSE_DEVIATION_PCT:
            self.state = CobotState.ERROR
        return rec

    def is_calibrated(self) -> bool:
        if not self.calibrations:
            return False
        return self.calibrations[-1].passed

    def summary(self) -> dict[str, object]:
        return {
            "cobot_id": self.cobot_id, "state": self.state.value,
            "tasks_completed": sum(1 for t in self.tasks if t.completed_at),
            "calibrated": self.is_calibrated(),
            "doses_prepared": len(self.dosing_log),
        }


def main() -> None:
    cobot = CobotInterface("FEP-001")
    cal = cobot.run_calibration(0.04, 0.2)
    print(f"Calibration passed: {cal.passed}")
    cobot.execute_task(TaskType.LAB_PIPETTING)
    cobot.execute_task(TaskType.SAMPLE_TRANSPORT)
    dose = cobot.prepare_dose("SUBJ-01", "DRUG-A", 250.0, 248.5)
    print(f"Dose deviation: {dose.deviation_pct}%")
    print(f"Summary: {cobot.summary()}")


if __name__ == "__main__":
    main()
