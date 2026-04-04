"""Site gateway: site qualification (PSL scoring), training delivery, scheduling, robot-task authorization."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class SiteStatus(Enum):
    SCREENING = "screening"
    QUALIFIED = "qualified"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class TrainingModule(Enum):
    GCP = "gcp"
    PROTOCOL = "protocol"
    ROBOT_OPERATION = "robot_operation"
    SAFETY_PROCEDURES = "safety_procedures"
    DATA_ENTRY = "data_entry"


@dataclass
class PSLScore:
    infrastructure: float = 0.0
    personnel: float = 0.0
    regulatory: float = 0.0
    technology: float = 0.0
    patient_access: float = 0.0

    @property
    def total(self) -> float:
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        scores = [self.infrastructure, self.personnel, self.regulatory, self.technology, self.patient_access]
        return round(sum(w * s for w, s in zip(weights, scores)), 2)


@dataclass
class TrainingRecord:
    module: TrainingModule
    personnel_id: str
    completed: bool = False
    completed_at: datetime | None = None
    score: float = 0.0


@dataclass
class SiteRecord:
    site_id: str = field(default_factory=lambda: f"SITE-{uuid.uuid4().hex[:4].upper()}")
    name: str = ""
    status: SiteStatus = SiteStatus.SCREENING
    psl_score: PSLScore = field(default_factory=PSLScore)
    training: list[TrainingRecord] = field(default_factory=list)
    authorized_robots: list[str] = field(default_factory=list)
    activated_at: datetime | None = None


class SiteGateway:
    """Manages site qualification via PSL scoring, training delivery, and robot-task authorization."""

    QUALIFICATION_THRESHOLD: float = 70.0

    def __init__(self) -> None:
        self.sites: dict[str, SiteRecord] = {}

    def register_site(self, name: str, psl: PSLScore) -> SiteRecord:
        site = SiteRecord(name=name, psl_score=psl)
        if psl.total >= self.QUALIFICATION_THRESHOLD:
            site.status = SiteStatus.QUALIFIED
        self.sites[site.site_id] = site
        return site

    def deliver_training(self, site_id: str, module: TrainingModule,
                         personnel_id: str, score: float) -> TrainingRecord:
        rec = TrainingRecord(module=module, personnel_id=personnel_id,
                             completed=score >= 80.0, score=score,
                             completed_at=datetime.utcnow() if score >= 80.0 else None)
        self.sites[site_id].training.append(rec)
        return rec

    def all_training_complete(self, site_id: str) -> bool:
        site = self.sites[site_id]
        required = set(TrainingModule)
        completed = {t.module for t in site.training if t.completed}
        return required.issubset(completed)

    def activate_site(self, site_id: str) -> bool:
        site = self.sites[site_id]
        if site.status == SiteStatus.QUALIFIED and self.all_training_complete(site_id):
            site.status = SiteStatus.ACTIVE
            site.activated_at = datetime.utcnow()
            return True
        return False

    def authorize_robot(self, site_id: str, robot_id: str) -> bool:
        site = self.sites.get(site_id)
        if site and site.status == SiteStatus.ACTIVE:
            site.authorized_robots.append(robot_id)
            return True
        return False

    def is_robot_authorized(self, site_id: str, robot_id: str) -> bool:
        site = self.sites.get(site_id)
        return site is not None and robot_id in site.authorized_robots

    def summary(self) -> dict[str, object]:
        by_status = {s.value: 0 for s in SiteStatus}
        for site in self.sites.values():
            by_status[site.status.value] += 1
        return {"total_sites": len(self.sites), "by_status": by_status}


def main() -> None:
    gw = SiteGateway()
    psl = PSLScore(infrastructure=85, personnel=80, regulatory=75, technology=80, patient_access=70)
    site = gw.register_site("Memorial Oncology Center", psl)
    print(f"PSL score: {psl.total}, status: {site.status.value}")
    for mod in TrainingModule:
        gw.deliver_training(site.site_id, mod, "PI-001", score=90.0)
    activated = gw.activate_site(site.site_id)
    print(f"Activated: {activated}")
    gw.authorize_robot(site.site_id, "DVX-001")
    print(f"Robot authorized: {gw.is_robot_authorized(site.site_id, 'DVX-001')}")
    print(f"Gateway summary: {gw.summary()}")


if __name__ == "__main__":
    main()
