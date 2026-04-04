"""Robotic capability registry: USL scoring (10 categories, 29 instances), capability matching, maintenance."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class USLCategory(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    SENSING = "sensing"
    DISPENSING = "dispensing"
    IMAGING = "imaging"
    BIOPSY = "biopsy"
    SUTURING = "suturing"
    TRANSPORT = "transport"
    STERILIZATION = "sterilization"
    CALIBRATION = "calibration"


class MaintenanceStatus(Enum):
    OPERATIONAL = "operational"
    SCHEDULED = "scheduled"
    IN_MAINTENANCE = "in_maintenance"
    OFFLINE = "offline"


@dataclass
class USLScore:
    category: USLCategory
    level: int = 0  # 0-5
    certified: bool = False
    last_assessed: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RobotInstance:
    robot_id: str = field(default_factory=lambda: f"ROB-{uuid.uuid4().hex[:6]}")
    model: str = ""
    site_id: str = ""
    capabilities: list[USLScore] = field(default_factory=list)
    maintenance_status: MaintenanceStatus = MaintenanceStatus.OPERATIONAL
    next_maintenance: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=90))
    total_procedures: int = 0


ROBOT_FLEET: list[dict[str, str]] = [
    {"model": "da Vinci Xi", "prefix": "DVX"},
    {"model": "Franka Emika Panda", "prefix": "FEP"},
    {"model": "UR10e", "prefix": "UR10"},
]


class RoboticCapabilityRegistry:
    """Registry of 29 robot instances across 10 USL categories with capability matching."""

    def __init__(self) -> None:
        self.robots: dict[str, RobotInstance] = {}

    def register_robot(self, model: str, site_id: str, capabilities: list[USLScore]) -> RobotInstance:
        robot = RobotInstance(model=model, site_id=site_id, capabilities=capabilities)
        self.robots[robot.robot_id] = robot
        return robot

    def match_capability(self, category: USLCategory, min_level: int = 3) -> list[RobotInstance]:
        matched: list[RobotInstance] = []
        for robot in self.robots.values():
            if robot.maintenance_status != MaintenanceStatus.OPERATIONAL:
                continue
            for cap in robot.capabilities:
                if cap.category == category and cap.level >= min_level and cap.certified:
                    matched.append(robot)
                    break
        return sorted(
            matched,
            key=lambda r: max((c.level for c in r.capabilities if c.category == category), default=0),
            reverse=True,
        )

    def schedule_maintenance(self, robot_id: str) -> bool:
        robot = self.robots.get(robot_id)
        if robot and robot.maintenance_status == MaintenanceStatus.OPERATIONAL:
            robot.maintenance_status = MaintenanceStatus.SCHEDULED
            return True
        return False

    def complete_maintenance(self, robot_id: str) -> bool:
        robot = self.robots.get(robot_id)
        if robot and robot.maintenance_status in (MaintenanceStatus.SCHEDULED, MaintenanceStatus.IN_MAINTENANCE):
            robot.maintenance_status = MaintenanceStatus.OPERATIONAL
            robot.next_maintenance = datetime.utcnow() + timedelta(days=90)
            return True
        return False

    def fleet_summary(self) -> dict[str, object]:
        by_status = {s.value: 0 for s in MaintenanceStatus}
        for robot in self.robots.values():
            by_status[robot.maintenance_status.value] += 1
        return {
            "total_robots": len(self.robots),
            "by_status": by_status,
            "total_procedures": sum(r.total_procedures for r in self.robots.values()),
        }

    def populate_fleet(self, sites: list[str]) -> None:
        """Populate registry with 29 robot instances across sites."""
        instance_count = 0
        for site_id in sites:
            for spec in ROBOT_FLEET:
                caps = [USLScore(category=cat, level=4, certified=True) for cat in list(USLCategory)[:5]]
                self.register_robot(spec["model"], site_id, caps)
                instance_count += 1
                if instance_count >= 29:
                    return


def main() -> None:
    registry = RoboticCapabilityRegistry()
    sites = ["SITE-A", "SITE-B", "SITE-C", "SITE-D", "SITE-E", "SITE-F", "SITE-G", "SITE-H", "SITE-I", "SITE-J"]
    registry.populate_fleet(sites)
    print(f"Fleet: {registry.fleet_summary()}")
    matches = registry.match_capability(USLCategory.MANIPULATION, min_level=3)
    print(f"Manipulation-capable robots: {len(matches)}")
    first_id = next(iter(registry.robots))
    registry.schedule_maintenance(first_id)
    print(f"After scheduling maintenance: {registry.fleet_summary()}")


if __name__ == "__main__":
    main()
