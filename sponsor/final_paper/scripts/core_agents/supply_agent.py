"""Supply agent: lot release, inventory tracking, demand forecasting, chain of custody."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class LotStatus(Enum):
    QUARANTINE = "quarantine"
    RELEASED = "released"
    REJECTED = "rejected"
    EXPIRED = "expired"
    RECALLED = "recalled"


class CustodyEventType(Enum):
    MANUFACTURED = "manufactured"
    QC_RELEASED = "qc_released"
    SHIPPED = "shipped"
    RECEIVED = "received"
    DISPENSED = "dispensed"


@dataclass
class CustodyEvent:
    event_id: str = field(default_factory=lambda: f"EVT-{uuid.uuid4().hex[:6]}")
    event_type: CustodyEventType = CustodyEventType.MANUFACTURED
    location: str = ""
    performed_by: str = ""
    event_date: date = field(default_factory=date.today)
    temperature_c: float | None = None


@dataclass
class DrugLot:
    lot_number: str
    product_name: str
    quantity: int = 0
    units_available: int = 0
    status: LotStatus = LotStatus.QUARANTINE
    manufactured_date: date = field(default_factory=date.today)
    expiry_date: date | None = None
    storage_temp_range: tuple[float, float] = (2.0, 8.0)
    custody_chain: list[CustodyEvent] = field(default_factory=list)


@dataclass
class DepotInventory:
    depot_id: str
    location: str
    lots: dict[str, int] = field(default_factory=dict)


class SupplyAgent:
    """Manages lot release, inventory tracking, demand forecasting, and chain of custody."""

    def __init__(self, study_id: str) -> None:
        self.study_id = study_id
        self.lots: dict[str, DrugLot] = {}
        self.depots: dict[str, DepotInventory] = {}

    def register_lot(self, lot_number: str, product: str, quantity: int, shelf_life_days: int = 365) -> DrugLot:
        lot = DrugLot(
            lot_number=lot_number,
            product_name=product,
            quantity=quantity,
            units_available=quantity,
            expiry_date=date.today() + timedelta(days=shelf_life_days),
        )
        lot.custody_chain.append(CustodyEvent(event_type=CustodyEventType.MANUFACTURED, location="CMO"))
        self.lots[lot_number] = lot
        return lot

    def release_lot(self, lot_number: str, released_by: str) -> DrugLot:
        lot = self.lots[lot_number]
        lot.status = LotStatus.RELEASED
        lot.custody_chain.append(
            CustodyEvent(event_type=CustodyEventType.QC_RELEASED, location="QC Lab", performed_by=released_by)
        )
        return lot

    def register_depot(self, depot_id: str, location: str) -> DepotInventory:
        depot = DepotInventory(depot_id=depot_id, location=location)
        self.depots[depot_id] = depot
        return depot

    def ship_to_depot(self, lot_number: str, depot_id: str, units: int) -> None:
        lot = self.lots[lot_number]
        if lot.status != LotStatus.RELEASED:
            msg = f"Lot {lot_number} not released"
            raise ValueError(msg)
        if units > lot.units_available:
            msg = f"Insufficient units: {lot.units_available} available"
            raise ValueError(msg)
        lot.units_available -= units
        lot.custody_chain.append(CustodyEvent(event_type=CustodyEventType.SHIPPED, location=depot_id))
        depot = self.depots[depot_id]
        depot.lots[lot_number] = depot.lots.get(lot_number, 0) + units

    def dispense(self, depot_id: str, lot_number: str, subject_id: str, units: int = 1) -> CustodyEvent:
        depot = self.depots[depot_id]
        available = depot.lots.get(lot_number, 0)
        if available < units:
            msg = f"Depot {depot_id} has {available} units of {lot_number}"
            raise ValueError(msg)
        depot.lots[lot_number] -= units
        event = CustodyEvent(event_type=CustodyEventType.DISPENSED, location=depot_id, performed_by=subject_id)
        self.lots[lot_number].custody_chain.append(event)
        return event

    def forecast_demand(
        self, enrollment_rate: float, treatment_duration_weeks: int, doses_per_week: float
    ) -> dict[str, float]:
        monthly_enrollment = enrollment_rate * 30
        active_subjects_steady = enrollment_rate * treatment_duration_weeks * 7
        monthly_demand = active_subjects_steady * doses_per_week * (30 / 7)
        buffer = monthly_demand * 0.2
        return {
            "monthly_enrollment": round(monthly_enrollment, 1),
            "active_subjects_steady_state": round(active_subjects_steady, 1),
            "monthly_demand_units": round(monthly_demand, 1),
            "recommended_buffer": round(buffer, 1),
            "total_monthly_need": round(monthly_demand + buffer, 1),
        }

    def expiring_lots(self, within_days: int = 90) -> list[DrugLot]:
        cutoff = date.today() + timedelta(days=within_days)
        return [
            lot
            for lot in self.lots.values()
            if lot.expiry_date and lot.expiry_date <= cutoff and lot.status == LotStatus.RELEASED
        ]

    def inventory_summary(self) -> dict[str, object]:
        total_available = sum(lot.units_available for lot in self.lots.values() if lot.status == LotStatus.RELEASED)
        return {
            "study": self.study_id,
            "total_lots": len(self.lots),
            "released_lots": sum(1 for lt in self.lots.values() if lt.status == LotStatus.RELEASED),
            "total_units_available": total_available,
            "depots": len(self.depots),
            "expiring_90d": len(self.expiring_lots(90)),
        }


def main() -> None:
    agent = SupplyAgent("STU-001")
    agent.register_lot("LOT-2026-001", "PAI-101 10mg", 500)
    agent.release_lot("LOT-2026-001", "QC_Manager")
    agent.register_depot("DEPOT-US-01", "US Central")
    agent.register_depot("DEPOT-EU-01", "EU Central")
    agent.ship_to_depot("LOT-2026-001", "DEPOT-US-01", 200)
    agent.ship_to_depot("LOT-2026-001", "DEPOT-EU-01", 150)
    agent.dispense("DEPOT-US-01", "LOT-2026-001", "SUBJ-001")
    forecast = agent.forecast_demand(enrollment_rate=0.5, treatment_duration_weeks=24, doses_per_week=1.0)
    print(f"Forecast: {forecast}")
    print(f"Inventory: {agent.inventory_summary()}")


if __name__ == "__main__":
    main()
