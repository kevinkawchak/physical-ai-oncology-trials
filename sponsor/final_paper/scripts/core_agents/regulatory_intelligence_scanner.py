"""Regulatory intelligence scanner: guidance monitoring, impact assessment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class GuidanceSource(Enum):
    FDA = "FDA"
    EMA = "EMA"
    ICH = "ICH"
    PMDA = "PMDA"
    MHRA = "MHRA"


class ImpactLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuidanceCategory(Enum):
    CLINICAL = "clinical"
    CMC = "cmc"
    NONCLINICAL = "nonclinical"
    LABELING = "labeling"
    SAFETY = "safety"
    DIGITAL_HEALTH = "digital_health"
    AI_ML = "ai_ml"


@dataclass
class GuidanceDocument:
    doc_id: str
    title: str
    source: GuidanceSource
    category: GuidanceCategory
    published_date: date
    effective_date: date | None = None
    url: str = ""
    summary: str = ""
    fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        content = f"{self.title}|{self.source.value}|{self.published_date.isoformat()}"
        self.fingerprint = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.fingerprint


@dataclass
class ImpactAssessment:
    guidance_id: str
    product: str
    impact_level: ImpactLevel = ImpactLevel.NONE
    affected_areas: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    assessed_date: date = field(default_factory=date.today)


@dataclass
class ScanResult:
    scan_date: date
    new_documents: list[GuidanceDocument] = field(default_factory=list)
    updated_documents: list[GuidanceDocument] = field(default_factory=list)


class RegulatoryIntelligenceScanner:
    """Monitors regulatory guidance and assesses impact on development programs."""

    def __init__(self) -> None:
        self.known_documents: dict[str, GuidanceDocument] = {}
        self.assessments: list[ImpactAssessment] = []
        self.watchlist_keywords: list[str] = [
            "oncology",
            "robotics",
            "artificial intelligence",
            "adaptive design",
            "companion diagnostic",
            "real-world evidence",
            "decentralized trial",
            "cell therapy",
            "digital endpoint",
            "patient-reported outcome",
        ]

    def ingest_document(self, doc: GuidanceDocument) -> bool:
        doc.compute_fingerprint()
        existing = self.known_documents.get(doc.doc_id)
        if existing and existing.fingerprint == doc.fingerprint:
            return False
        self.known_documents[doc.doc_id] = doc
        return True

    def scan_for_relevance(self, doc: GuidanceDocument) -> float:
        text = f"{doc.title} {doc.summary}".lower()
        matches = sum(1 for kw in self.watchlist_keywords if kw in text)
        return min(matches / max(len(self.watchlist_keywords), 1) * 3.0, 1.0)

    def assess_impact(self, doc_id: str, product: str, affected_areas: list[str]) -> ImpactAssessment:
        doc = self.known_documents[doc_id]
        relevance = self.scan_for_relevance(doc)
        if relevance >= 0.6:
            level = ImpactLevel.HIGH
        elif relevance >= 0.3:
            level = ImpactLevel.MEDIUM
        elif relevance > 0:
            level = ImpactLevel.LOW
        else:
            level = ImpactLevel.NONE
        actions: list[str] = []
        if level in (ImpactLevel.HIGH, ImpactLevel.CRITICAL):
            actions.append(f"Review {doc.title} for protocol impact")
            actions.append("Schedule regulatory strategy meeting")
        if doc.category == GuidanceCategory.AI_ML:
            actions.append("Update AI/ML validation documentation")
        assessment = ImpactAssessment(
            guidance_id=doc_id, product=product, impact_level=level, affected_areas=affected_areas, action_items=actions
        )
        self.assessments.append(assessment)
        return assessment

    def high_impact_alerts(self) -> list[ImpactAssessment]:
        return [a for a in self.assessments if a.impact_level in (ImpactLevel.HIGH, ImpactLevel.CRITICAL)]

    def dashboard(self) -> dict[str, object]:
        return {
            "total_documents": len(self.known_documents),
            "assessments": len(self.assessments),
            "high_impact": len(self.high_impact_alerts()),
            "sources": list({d.source.value for d in self.known_documents.values()}),
        }


def main() -> None:
    scanner = RegulatoryIntelligenceScanner()
    doc = GuidanceDocument(
        "FDA-2026-001",
        "Artificial Intelligence in Oncology Clinical Trials",
        GuidanceSource.FDA,
        GuidanceCategory.AI_ML,
        date(2026, 3, 1),
        summary="Guidance on use of artificial intelligence and adaptive design in oncology trials",
    )
    scanner.ingest_document(doc)
    assessment = scanner.assess_impact("FDA-2026-001", "PAI-101", ["protocol", "SAP"])
    print(f"Impact: {assessment.impact_level.value}, Actions: {assessment.action_items}")
    print(f"Dashboard: {scanner.dashboard()}")


if __name__ == "__main__":
    main()
