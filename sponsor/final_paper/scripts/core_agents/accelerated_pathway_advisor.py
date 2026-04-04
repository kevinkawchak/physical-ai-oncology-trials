"""Accelerated pathway advisor: eligibility for Breakthrough, Fast Track, PRIME, etc."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class Pathway(Enum):
    BREAKTHROUGH = "breakthrough_therapy"
    FAST_TRACK = "fast_track"
    ACCELERATED_APPROVAL = "accelerated_approval"
    PRIORITY_REVIEW = "priority_review"
    PRIME = "prime"  # EMA
    RMAT = "rmat"  # Regenerative Medicine


class EligibilityCriterion(Enum):
    SERIOUS_CONDITION = "serious_condition"
    UNMET_NEED = "unmet_need"
    PRELIMINARY_CLINICAL_EVIDENCE = "preliminary_clinical_evidence"
    SUBSTANTIAL_IMPROVEMENT = "substantial_improvement"
    SURROGATE_ENDPOINT = "surrogate_endpoint"
    NONCLINICAL_PROMISE = "nonclinical_promise"
    INNOVATIVE_THERAPY = "innovative_therapy"


PATHWAY_REQUIREMENTS: dict[Pathway, list[EligibilityCriterion]] = {
    Pathway.BREAKTHROUGH: [
        EligibilityCriterion.SERIOUS_CONDITION,
        EligibilityCriterion.PRELIMINARY_CLINICAL_EVIDENCE,
        EligibilityCriterion.SUBSTANTIAL_IMPROVEMENT,
    ],
    Pathway.FAST_TRACK: [EligibilityCriterion.SERIOUS_CONDITION, EligibilityCriterion.UNMET_NEED],
    Pathway.ACCELERATED_APPROVAL: [
        EligibilityCriterion.SERIOUS_CONDITION,
        EligibilityCriterion.UNMET_NEED,
        EligibilityCriterion.SURROGATE_ENDPOINT,
    ],
    Pathway.PRIORITY_REVIEW: [EligibilityCriterion.SERIOUS_CONDITION, EligibilityCriterion.SUBSTANTIAL_IMPROVEMENT],
    Pathway.PRIME: [
        EligibilityCriterion.UNMET_NEED,
        EligibilityCriterion.INNOVATIVE_THERAPY,
        EligibilityCriterion.NONCLINICAL_PROMISE,
    ],
    Pathway.RMAT: [EligibilityCriterion.SERIOUS_CONDITION, EligibilityCriterion.PRELIMINARY_CLINICAL_EVIDENCE],
}


@dataclass
class EvidenceItem:
    criterion: EligibilityCriterion
    description: str
    strength: float  # 0-1
    source: str = ""


@dataclass
class PathwayAssessment:
    pathway: Pathway
    eligible: bool
    score: float
    met_criteria: list[EligibilityCriterion] = field(default_factory=list)
    gaps: list[EligibilityCriterion] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class DesignationRequest:
    request_id: str
    product_name: str
    indication: str
    pathway: Pathway
    evidence: list[EvidenceItem] = field(default_factory=list)
    submitted_date: date | None = None
    granted: bool = False


class AcceleratedPathwayAdvisor:
    """Evaluates eligibility for accelerated regulatory pathways."""

    def __init__(self, product_name: str, indication: str) -> None:
        self.product_name = product_name
        self.indication = indication
        self.evidence: list[EvidenceItem] = []
        self.assessments: list[PathwayAssessment] = []

    def add_evidence(self, item: EvidenceItem) -> None:
        self.evidence.append(item)

    def assess_pathway(self, pathway: Pathway) -> PathwayAssessment:
        required = PATHWAY_REQUIREMENTS.get(pathway, [])
        evidence_map = {e.criterion: e for e in self.evidence}
        met: list[EligibilityCriterion] = []
        gaps: list[EligibilityCriterion] = []
        total_strength = 0.0
        for req in required:
            ev = evidence_map.get(req)
            if ev and ev.strength >= 0.5:
                met.append(req)
                total_strength += ev.strength
            else:
                gaps.append(req)
        score = total_strength / max(len(required), 1)
        eligible = len(gaps) == 0 and score >= 0.6
        rec = "Recommend submission" if eligible else f"Address gaps: {[g.value for g in gaps]}"
        assessment = PathwayAssessment(
            pathway=pathway, eligible=eligible, score=round(score, 3), met_criteria=met, gaps=gaps, recommendation=rec
        )
        self.assessments.append(assessment)
        return assessment

    def assess_all_pathways(self) -> list[PathwayAssessment]:
        results: list[PathwayAssessment] = []
        for pathway in Pathway:
            results.append(self.assess_pathway(pathway))
        return results

    def best_pathway(self) -> PathwayAssessment | None:
        eligible = [a for a in self.assessments if a.eligible]
        if not eligible:
            return None
        return max(eligible, key=lambda a: a.score)

    def generate_briefing_doc(self, pathway: Pathway) -> dict[str, object]:
        assessment = next((a for a in self.assessments if a.pathway == pathway), None)
        return {
            "product": self.product_name,
            "indication": self.indication,
            "pathway": pathway.value,
            "eligible": assessment.eligible if assessment else False,
            "evidence_items": len(self.evidence),
            "recommendation": assessment.recommendation if assessment else "Not assessed",
        }


def main() -> None:
    advisor = AcceleratedPathwayAdvisor("PAI-101", "3L+ NSCLC")
    advisor.add_evidence(EvidenceItem(EligibilityCriterion.SERIOUS_CONDITION, "Stage IV NSCLC", 0.95))
    advisor.add_evidence(EvidenceItem(EligibilityCriterion.UNMET_NEED, "No approved 3L therapy", 0.85))
    advisor.add_evidence(EvidenceItem(EligibilityCriterion.PRELIMINARY_CLINICAL_EVIDENCE, "Ph1 ORR 42%", 0.75))
    advisor.add_evidence(EvidenceItem(EligibilityCriterion.SUBSTANTIAL_IMPROVEMENT, "ORR 2x SOC", 0.70))
    results = advisor.assess_all_pathways()
    for r in results:
        print(f"{r.pathway.value}: eligible={r.eligible}, score={r.score}")
    best = advisor.best_pathway()
    if best:
        print(f"Best pathway: {best.pathway.value}")


if __name__ == "__main__":
    main()
