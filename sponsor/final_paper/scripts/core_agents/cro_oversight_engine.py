"""CRO oversight engine: 11-area accountability matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class OversightArea(Enum):
    SITE_MANAGEMENT = "site_management"
    MONITORING = "monitoring"
    DATA_MANAGEMENT = "data_management"
    MEDICAL_MONITORING = "medical_monitoring"
    SAFETY_REPORTING = "safety_reporting"
    REGULATORY = "regulatory"
    BIOSTATISTICS = "biostatistics"
    MEDICAL_WRITING = "medical_writing"
    SUPPLY_CHAIN = "supply_chain"
    QUALITY_ASSURANCE = "quality_assurance"
    PROJECT_MANAGEMENT = "project_management"


class Responsibility(Enum):
    SPONSOR = "sponsor"
    CRO = "cro"
    SHARED = "shared"


class ComplianceLevel(Enum):
    FULL = "full"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


@dataclass
class AccountabilityEntry:
    area: OversightArea
    responsibility: Responsibility = Responsibility.SHARED
    sponsor_contact: str = ""
    cro_contact: str = ""
    kpis: list[str] = field(default_factory=list)
    compliance: ComplianceLevel = ComplianceLevel.NOT_ASSESSED
    last_reviewed: date | None = None
    findings: list[str] = field(default_factory=list)


@dataclass
class OversightReview:
    review_id: str
    review_date: date = field(default_factory=date.today)
    scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    action_items: list[str] = field(default_factory=list)


class CROOversightEngine:
    """Manages 11-area CRO accountability matrix per ICH E6(R2) sponsor oversight obligations."""

    def __init__(self, cro_name: str) -> None:
        self.cro_name = cro_name
        self.matrix: dict[OversightArea, AccountabilityEntry] = {}
        self.reviews: list[OversightReview] = []
        self._initialize_matrix()

    def _initialize_matrix(self) -> None:
        defaults: dict[OversightArea, Responsibility] = {
            OversightArea.SITE_MANAGEMENT: Responsibility.CRO,
            OversightArea.MONITORING: Responsibility.CRO,
            OversightArea.DATA_MANAGEMENT: Responsibility.CRO,
            OversightArea.MEDICAL_MONITORING: Responsibility.SPONSOR,
            OversightArea.SAFETY_REPORTING: Responsibility.SHARED,
            OversightArea.REGULATORY: Responsibility.SHARED,
            OversightArea.BIOSTATISTICS: Responsibility.CRO,
            OversightArea.MEDICAL_WRITING: Responsibility.SHARED,
            OversightArea.SUPPLY_CHAIN: Responsibility.SPONSOR,
            OversightArea.QUALITY_ASSURANCE: Responsibility.SHARED,
            OversightArea.PROJECT_MANAGEMENT: Responsibility.CRO,
        }
        for area, resp in defaults.items():
            self.matrix[area] = AccountabilityEntry(area=area, responsibility=resp)

    def assign_contacts(self, area: OversightArea, sponsor_contact: str, cro_contact: str) -> None:
        entry = self.matrix[area]
        entry.sponsor_contact = sponsor_contact
        entry.cro_contact = cro_contact

    def set_kpis(self, area: OversightArea, kpis: list[str]) -> None:
        self.matrix[area].kpis = kpis

    def assess_compliance(self, area: OversightArea, level: ComplianceLevel, findings: list[str] | None = None) -> None:
        entry = self.matrix[area]
        entry.compliance = level
        entry.last_reviewed = date.today()
        entry.findings = findings or []

    def conduct_review(self, review_id: str) -> OversightReview:
        scores: dict[str, float] = {}
        action_items: list[str] = []
        score_map = {
            ComplianceLevel.FULL: 1.0,
            ComplianceLevel.PARTIAL: 0.6,
            ComplianceLevel.NON_COMPLIANT: 0.2,
            ComplianceLevel.NOT_ASSESSED: 0.0,
        }
        for area, entry in self.matrix.items():
            score = score_map.get(entry.compliance, 0.0)
            scores[area.value] = score
            if entry.compliance in (ComplianceLevel.NON_COMPLIANT, ComplianceLevel.PARTIAL):
                action_items.append(f"Address {area.value}: {', '.join(entry.findings)}")
        overall = sum(scores.values()) / max(len(scores), 1)
        review = OversightReview(
            review_id=review_id, scores=scores, overall_score=round(overall, 3), action_items=action_items
        )
        self.reviews.append(review)
        return review

    def compliance_heat_map(self) -> list[dict[str, str]]:
        return [
            {"area": e.area.value, "responsibility": e.responsibility.value, "compliance": e.compliance.value}
            for e in self.matrix.values()
        ]

    def summary(self) -> dict[str, object]:
        full = sum(1 for e in self.matrix.values() if e.compliance == ComplianceLevel.FULL)
        return {"cro": self.cro_name, "areas": len(self.matrix), "fully_compliant": full, "reviews": len(self.reviews)}


def main() -> None:
    engine = CROOversightEngine("GlobalCRO Inc.")
    engine.assign_contacts(OversightArea.MONITORING, "Dr. Smith", "J. Jones")
    engine.set_kpis(OversightArea.MONITORING, ["visit_completion_rate", "query_resolution_time"])
    engine.assess_compliance(OversightArea.MONITORING, ComplianceLevel.FULL)
    engine.assess_compliance(OversightArea.DATA_MANAGEMENT, ComplianceLevel.PARTIAL, ["High query rate"])
    engine.assess_compliance(OversightArea.SAFETY_REPORTING, ComplianceLevel.FULL)
    review = engine.conduct_review("REV-Q1-2026")
    print(f"Overall: {review.overall_score}, Actions: {len(review.action_items)}")
    print(f"Summary: {engine.summary()}")


if __name__ == "__main__":
    main()
