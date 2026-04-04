"""IRT demand forecaster: IRT-driven demand forecasting, depot optimization, cold-chain logistics."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class StorageCondition(Enum):
    AMBIENT = "ambient_15_25c"
    REFRIGERATED = "refrigerated_2_8c"
    FROZEN = "frozen_neg20c"


class ShipmentStatus(Enum):
    PLANNED = "planned"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    EXCURSION = "temperature_excursion"


@dataclass
class DepotProfile:
    depot_id: str
    region: str
    storage: StorageCondition = StorageCondition.REFRIGERATED
    capacity_units: int = 1000
    current_stock: int = 0
    lead_time_days: int = 14
    active_sites: int = 0

    @property
    def weeks_of_stock(self) -> float:
        if self.active_sites == 0:
            return float("inf")
        return self.current_stock / max(self.active_sites * 2.0, 0.01)


@dataclass
class ColdChainShipment:
    shipment_id: str = field(default_factory=lambda: f"SHP-{uuid.uuid4().hex[:6]}")
    origin_depot: str = ""
    destination: str = ""
    units: int = 0
    storage: StorageCondition = StorageCondition.REFRIGERATED
    status: ShipmentStatus = ShipmentStatus.PLANNED
    ship_date: date = field(default_factory=date.today)
    temp_readings: list[float] = field(default_factory=list)

    def check_excursion(self) -> bool:
        limits = {StorageCondition.AMBIENT: (15.0, 25.0), StorageCondition.REFRIGERATED: (2.0, 8.0),
                  StorageCondition.FROZEN: (-25.0, -15.0)}
        lo, hi = limits.get(self.storage, (2.0, 8.0))
        for t in self.temp_readings:
            if t < lo or t > hi:
                self.status = ShipmentStatus.EXCURSION
                return True
        return False


@dataclass
class DemandForecast:
    depot_id: str
    projected_demand_4wk: float = 0.0
    reorder_point: float = 0.0
    recommended_order: float = 0.0
    safety_stock: float = 0.0


class IRTDemandForecaster:
    """IRT-driven demand forecasting with depot optimization and cold-chain management."""

    def __init__(self, study_id: str, doses_per_subject_week: float = 1.0) -> None:
        self.study_id = study_id
        self.doses_per_subject_week = doses_per_subject_week
        self.depots: dict[str, DepotProfile] = {}
        self.shipments: list[ColdChainShipment] = []

    def register_depot(self, depot_id: str, region: str, capacity: int,
                       storage: StorageCondition, lead_time_days: int = 14) -> DepotProfile:
        depot = DepotProfile(depot_id=depot_id, region=region, capacity_units=capacity,
                             storage=storage, lead_time_days=lead_time_days)
        self.depots[depot_id] = depot
        return depot

    def update_enrollment(self, depot_id: str, active_sites: int, current_stock: int) -> None:
        depot = self.depots[depot_id]
        depot.active_sites = active_sites
        depot.current_stock = current_stock

    def forecast_depot(self, depot_id: str) -> DemandForecast:
        depot = self.depots[depot_id]
        weekly_demand = depot.active_sites * self.doses_per_subject_week * 2.5
        demand_4wk = weekly_demand * 4
        lead_time_weeks = depot.lead_time_days / 7
        safety_stock = weekly_demand * math.sqrt(lead_time_weeks) * 1.65
        reorder_point = weekly_demand * lead_time_weeks + safety_stock
        recommended = max(demand_4wk + safety_stock - depot.current_stock, 0)
        return DemandForecast(
            depot_id=depot_id, projected_demand_4wk=round(demand_4wk, 1),
            reorder_point=round(reorder_point, 1),
            recommended_order=round(recommended, 1),
            safety_stock=round(safety_stock, 1),
        )

    def create_shipment(self, origin: str, destination: str, units: int,
                        storage: StorageCondition) -> ColdChainShipment:
        shipment = ColdChainShipment(origin_depot=origin, destination=destination,
                                     units=units, storage=storage)
        self.shipments.append(shipment)
        return shipment

    def record_temperature(self, shipment_id: str, readings: list[float]) -> bool:
        for s in self.shipments:
            if s.shipment_id == shipment_id:
                s.temp_readings.extend(readings)
                return s.check_excursion()
        return False

    def depots_below_reorder(self) -> list[str]:
        return [did for did in self.depots
                if self.depots[did].current_stock <= self.forecast_depot(did).reorder_point]

    def supply_dashboard(self) -> dict[str, object]:
        excursions = sum(1 for s in self.shipments if s.status == ShipmentStatus.EXCURSION)
        return {
            "study": self.study_id, "depots": len(self.depots),
            "total_stock": sum(d.current_stock for d in self.depots.values()),
            "below_reorder": len(self.depots_below_reorder()),
            "active_shipments": sum(1 for s in self.shipments if s.status == ShipmentStatus.IN_TRANSIT),
            "excursions": excursions,
        }


def main() -> None:
    fc = IRTDemandForecaster("STU-001", doses_per_subject_week=1.0)
    fc.register_depot("DEPOT-US", "North America", 2000, StorageCondition.REFRIGERATED)
    fc.register_depot("DEPOT-EU", "Europe", 1500, StorageCondition.REFRIGERATED)
    fc.update_enrollment("DEPOT-US", active_sites=8, current_stock=300)
    fc.update_enrollment("DEPOT-EU", active_sites=5, current_stock=80)
    us = fc.forecast_depot("DEPOT-US")
    print(f"US forecast: demand_4wk={us.projected_demand_4wk}, order={us.recommended_order}")
    shipment = fc.create_shipment("DEPOT-US", "SITE-EU-03", 50, StorageCondition.REFRIGERATED)
    fc.record_temperature(shipment.shipment_id, [4.1, 4.5, 9.2, 5.0])
    print(f"Excursion: {shipment.status.value}")
    print(f"Dashboard: {fc.supply_dashboard()}")


if __name__ == "__main__":
    main()
