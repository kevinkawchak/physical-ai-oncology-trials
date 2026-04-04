"""Robotic pharmacy interface: Franka Emika pharmacy dose calculation, preparation, dispensing."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DoseFormulation(Enum):
    IV_INFUSION = "iv_infusion"
    SUBCUTANEOUS = "subcutaneous"
    ORAL_CAPSULE = "oral_capsule"


class PreparationStatus(Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    READY = "ready"
    DISPENSED = "dispensed"
    REJECTED = "rejected"
    ERROR = "error"


class RobotState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class DoseCalculation:
    subject_id: str
    weight_kg: float
    bsa_m2: float
    dose_mg_per_kg: float | None = None
    dose_mg_per_m2: float | None = None
    flat_dose_mg: float | None = None
    calculated_dose_mg: float = 0.0
    volume_ml: float = 0.0

    def compute(self, concentration: float = 10.0) -> float:
        if self.flat_dose_mg is not None:
            self.calculated_dose_mg = self.flat_dose_mg
        elif self.dose_mg_per_kg is not None:
            self.calculated_dose_mg = round(self.dose_mg_per_kg * self.weight_kg, 2)
        elif self.dose_mg_per_m2 is not None:
            self.calculated_dose_mg = round(self.dose_mg_per_m2 * self.bsa_m2, 2)
        self.volume_ml = round(self.calculated_dose_mg / max(concentration, 0.01), 2)
        return self.calculated_dose_mg


@dataclass
class PreparationRecord:
    prep_id: str = field(default_factory=lambda: f"PREP-{uuid.uuid4().hex[:6]}")
    subject_id: str = ""
    lot_number: str = ""
    formulation: DoseFormulation = DoseFormulation.IV_INFUSION
    dose_mg: float = 0.0
    volume_ml: float = 0.0
    status: PreparationStatus = PreparationStatus.PENDING
    robot_arm_id: str = "franka_panda_01"
    prepared_at: datetime | None = None
    barcode: str = ""
    qc_weight_mg: float = 0.0
    qc_passed: bool = False


class RoboticPharmacyInterface:
    """Interface for Franka Emika robotic pharmacy: dose calculation, preparation, dispensing."""

    WEIGHT_TOLERANCE_PCT = 5.0

    def __init__(self, robot_id: str = "franka_panda_01") -> None:
        self.robot_id = robot_id
        self.state = RobotState.IDLE
        self.preparations: list[PreparationRecord] = []
        self.dispensing_log: list[dict[str, str]] = []

    def calculate_dose(self, subject_id: str, weight_kg: float, height_cm: float,
                       dose_mg_per_kg: float | None = None, dose_mg_per_m2: float | None = None,
                       flat_dose_mg: float | None = None, concentration: float = 10.0) -> DoseCalculation:
        bsa = round(0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725), 4)
        calc = DoseCalculation(subject_id=subject_id, weight_kg=weight_kg, bsa_m2=bsa,
                               dose_mg_per_kg=dose_mg_per_kg, dose_mg_per_m2=dose_mg_per_m2,
                               flat_dose_mg=flat_dose_mg)
        calc.compute(concentration)
        return calc

    def initiate_preparation(self, calc: DoseCalculation, lot_number: str,
                             formulation: DoseFormulation) -> PreparationRecord:
        if self.state == RobotState.ERROR:
            msg = "Robot in error state; cannot prepare"
            raise RuntimeError(msg)
        self.state = RobotState.ACTIVE
        barcode = f"BC-{uuid.uuid4().hex[:10].upper()}"
        record = PreparationRecord(
            subject_id=calc.subject_id, lot_number=lot_number,
            formulation=formulation, dose_mg=calc.calculated_dose_mg,
            volume_ml=calc.volume_ml, status=PreparationStatus.PREPARING,
            robot_arm_id=self.robot_id, barcode=barcode,
        )
        self.preparations.append(record)
        return record

    def gravimetric_qc(self, prep_id: str, measured_weight_mg: float) -> bool:
        for record in self.preparations:
            if record.prep_id == prep_id:
                record.qc_weight_mg = measured_weight_mg
                tolerance = record.dose_mg * self.WEIGHT_TOLERANCE_PCT / 100.0
                record.qc_passed = abs(measured_weight_mg - record.dose_mg) <= tolerance
                record.status = PreparationStatus.READY if record.qc_passed else PreparationStatus.REJECTED
                record.prepared_at = datetime.utcnow()
                return record.qc_passed
        msg = f"Preparation {prep_id} not found"
        raise KeyError(msg)

    def dispense(self, prep_id: str, pharmacist_id: str) -> PreparationRecord:
        for record in self.preparations:
            if record.prep_id == prep_id:
                if record.status != PreparationStatus.READY:
                    msg = f"Preparation {prep_id} not ready: {record.status.value}"
                    raise ValueError(msg)
                record.status = PreparationStatus.DISPENSED
                self.dispensing_log.append({
                    "prep_id": prep_id, "subject": record.subject_id,
                    "pharmacist": pharmacist_id,
                })
                self.state = RobotState.IDLE
                return record
        msg = f"Preparation {prep_id} not found"
        raise KeyError(msg)

    def emergency_stop(self) -> None:
        self.state = RobotState.ERROR
        for rec in self.preparations:
            if rec.status == PreparationStatus.PREPARING:
                rec.status = PreparationStatus.ERROR

    def production_summary(self) -> dict[str, object]:
        dispensed = sum(1 for p in self.preparations if p.status == PreparationStatus.DISPENSED)
        rejected = sum(1 for p in self.preparations if p.status == PreparationStatus.REJECTED)
        return {"robot": self.robot_id, "state": self.state.value,
                "total": len(self.preparations), "dispensed": dispensed, "rejected": rejected}


def main() -> None:
    pharmacy = RoboticPharmacyInterface("franka_panda_01")
    calc = pharmacy.calculate_dose("SUBJ-001", weight_kg=72.5, height_cm=175.0, dose_mg_per_kg=3.0)
    print(f"Dose: {calc.calculated_dose_mg} mg, Volume: {calc.volume_ml} mL, BSA: {calc.bsa_m2}")
    record = pharmacy.initiate_preparation(calc, "LOT-2026-001", DoseFormulation.IV_INFUSION)
    pharmacy.gravimetric_qc(record.prep_id, calc.calculated_dose_mg * 1.01)
    pharmacy.dispense(record.prep_id, "PHARM-001")
    print(f"Summary: {pharmacy.production_summary()}")


if __name__ == "__main__":
    main()
