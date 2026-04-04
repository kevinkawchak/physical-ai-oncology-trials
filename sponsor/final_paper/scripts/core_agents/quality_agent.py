"""Quality agent: SOP enforcement, CAPA lifecycle, audit readiness."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class SOPStatus(Enum):
    DRAFT = "draft"
    EFFECTIVE = "effective"
    UNDER_REVISION = "under_revision"
    RETIRED = "retired"


class CAPAType(Enum):
    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"


class CAPAStatus(Enum):
    OPEN = "open"
    INVESTIGATION = "investigation"
    ACTION_PLAN = "action_plan"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"
    CLOSED = "closed"


@dataclass
class SOP:
    sop_id: str
    title: str
    version: str = "1.0"
    status: SOPStatus = SOPStatus.DRAFT
    effective_date: date | None = None
    review_due: date | None = None
    owner: str = ""

    @property
    def needs_review(self) -> bool:
        if self.review_due is None:
            return False
        return date.today() >= self.review_due


@dataclass
class CAPARecord:
    capa_id: str = field(default_factory=lambda: f"CAPA-{uuid.uuid4().hex[:6]}")
    capa_type: CAPAType = CAPAType.CORRECTIVE
    status: CAPAStatus = CAPAStatus.OPEN
    description: str = ""
    root_cause: str = ""
    action_plan: str = ""
    opened_date: date = field(default_factory=date.today)
    due_date: date | None = None
    closed_date: date | None = None
    effectiveness_verified: bool = False


@dataclass
class AuditFinding:
    finding_id: str
    category: str
    description: str
    severity: str  # critical, major, minor, observation
    capa_id: str | None = None


@dataclass
class AuditReport:
    audit_id: str
    audit_type: str  # internal, external, regulatory
    audit_date: date
    findings: list[AuditFinding] = field(default_factory=list)
    score: float = 0.0


class QualityAgent:
    """Manages SOP enforcement, CAPA lifecycle, and audit readiness."""

    def __init__(self) -> None:
        self.sops: dict[str, SOP] = {}
        self.capas: list[CAPARecord] = []
        self.audits: list[AuditReport] = []

    def register_sop(self, sop: SOP) -> str:
        self.sops[sop.sop_id] = sop
        return sop.sop_id

    def activate_sop(self, sop_id: str) -> None:
        sop = self.sops[sop_id]
        sop.status = SOPStatus.EFFECTIVE
        sop.effective_date = date.today()
        sop.review_due = date.today() + timedelta(days=730)

    def sops_due_for_review(self) -> list[SOP]:
        return [s for s in self.sops.values() if s.needs_review]

    def open_capa(self, capa_type: CAPAType, description: str, due_days: int = 30) -> CAPARecord:
        capa = CAPARecord(
            capa_type=capa_type, description=description, due_date=date.today() + timedelta(days=due_days)
        )
        self.capas.append(capa)
        return capa

    def advance_capa(self, capa_id: str, new_status: CAPAStatus, notes: str = "") -> None:
        for capa in self.capas:
            if capa.capa_id == capa_id:
                capa.status = new_status
                if new_status == CAPAStatus.INVESTIGATION:
                    capa.root_cause = notes
                elif new_status == CAPAStatus.ACTION_PLAN:
                    capa.action_plan = notes
                elif new_status == CAPAStatus.CLOSED:
                    capa.closed_date = date.today()
                break

    def overdue_capas(self) -> list[CAPARecord]:
        today = date.today()
        return [c for c in self.capas if c.status != CAPAStatus.CLOSED and c.due_date and c.due_date < today]

    def audit_readiness_score(self) -> float:
        if not self.sops:
            return 0.0
        effective = sum(1 for s in self.sops.values() if s.status == SOPStatus.EFFECTIVE)
        overdue_review = len(self.sops_due_for_review())
        open_capas = sum(1 for c in self.capas if c.status != CAPAStatus.CLOSED)
        sop_score = effective / len(self.sops)
        review_penalty = min(overdue_review * 0.05, 0.2)
        capa_penalty = min(open_capas * 0.03, 0.15)
        return max(sop_score - review_penalty - capa_penalty, 0.0)

    def quality_dashboard(self) -> dict[str, object]:
        return {
            "total_sops": len(self.sops),
            "effective_sops": sum(1 for s in self.sops.values() if s.status == SOPStatus.EFFECTIVE),
            "sops_due_review": len(self.sops_due_for_review()),
            "open_capas": sum(1 for c in self.capas if c.status != CAPAStatus.CLOSED),
            "overdue_capas": len(self.overdue_capas()),
            "audit_readiness": round(self.audit_readiness_score(), 3),
        }


def main() -> None:
    agent = QualityAgent()
    for i in range(1, 6):
        sop = SOP(f"SOP-{i:03d}", f"Standard Procedure {i}", owner="Quality Lead")
        agent.register_sop(sop)
        agent.activate_sop(sop.sop_id)
    capa = agent.open_capa(CAPAType.CORRECTIVE, "Protocol deviation at Site 03")
    agent.advance_capa(capa.capa_id, CAPAStatus.INVESTIGATION, "Inadequate training")
    print(f"Dashboard: {agent.quality_dashboard()}")


if __name__ == "__main__":
    main()
