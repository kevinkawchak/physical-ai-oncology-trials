"""Governance decision engine: seven-gate decision framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class GateOutcome(Enum):
    GO = "GO"
    NO_GO = "NO_GO"
    CONDITIONAL = "CONDITIONAL"
    DEFER = "DEFER"


class GateName(Enum):
    G1_PORTFOLIO_ENTRY = "G1_Portfolio_Entry"
    G2_PRECLINICAL_COMPLETE = "G2_Preclinical_Complete"
    G3_FIH_READY = "G3_FIH_Ready"
    G4_DOSE_SELECTED = "G4_Dose_Selected"
    G5_PIVOTAL_READY = "G5_Pivotal_Ready"
    G6_FILING_READY = "G6_Filing_Ready"
    G7_LAUNCH_READY = "G7_Launch_Ready"


@dataclass
class GateCriterion:
    name: str
    category: str  # efficacy, safety, cmc, regulatory, commercial
    threshold: float
    actual: float = 0.0
    weight: float = 1.0

    @property
    def met(self) -> bool:
        return self.actual >= self.threshold

    @property
    def weighted_score(self) -> float:
        return self.weight * min(self.actual / max(self.threshold, 1e-9), 1.0)


@dataclass
class GateReview:
    gate: GateName
    criteria: list[GateCriterion] = field(default_factory=list)
    outcome: GateOutcome = GateOutcome.DEFER
    reviewed_at: datetime | None = None
    reviewer: str = ""
    conditions: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.criteria:
            return 0.0
        return sum(1 for c in self.criteria if c.met) / len(self.criteria)

    @property
    def composite_score(self) -> float:
        total_weight = sum(c.weight for c in self.criteria)
        if total_weight == 0:
            return 0.0
        return sum(c.weighted_score for c in self.criteria) / total_weight


GATE_TEMPLATES: dict[GateName, list[tuple[str, str, float]]] = {
    GateName.G1_PORTFOLIO_ENTRY: [
        ("unmet_need_score", "commercial", 0.6),
        ("target_validation", "efficacy", 0.7),
        ("ip_freedom", "regulatory", 0.8),
    ],
    GateName.G3_FIH_READY: [
        ("glp_tox_complete", "safety", 1.0),
        ("cmc_release_spec", "cmc", 1.0),
        ("ind_safety_pkg", "regulatory", 1.0),
        ("fih_protocol_approved", "efficacy", 1.0),
    ],
    GateName.G5_PIVOTAL_READY: [
        ("rp2d_selected", "efficacy", 1.0),
        ("biomarker_validated", "efficacy", 0.7),
        ("safety_acceptable", "safety", 0.8),
        ("registration_path_agreed", "regulatory", 1.0),
        ("pivotal_protocol_final", "efficacy", 1.0),
    ],
}


class GovernanceDecisionEngine:
    """Seven-gate governance framework for oncology asset progression."""

    def __init__(self) -> None:
        self.reviews: list[GateReview] = []

    def create_review(self, gate: GateName, reviewer: str) -> GateReview:
        review = GateReview(gate=gate, reviewer=reviewer)
        template = GATE_TEMPLATES.get(gate, [])
        for name, category, threshold in template:
            review.criteria.append(GateCriterion(name=name, category=category, threshold=threshold))
        return review

    def evaluate(self, review: GateReview, go_threshold: float = 0.8, conditional_threshold: float = 0.6) -> GateReview:
        score = review.composite_score
        if review.pass_rate == 1.0 and score >= go_threshold:
            review.outcome = GateOutcome.GO
        elif score >= conditional_threshold:
            review.outcome = GateOutcome.CONDITIONAL
            unmet = [c.name for c in review.criteria if not c.met]
            review.conditions = [f"Resolve: {n}" for n in unmet]
        else:
            review.outcome = GateOutcome.NO_GO
        review.reviewed_at = datetime.utcnow()
        self.reviews.append(review)
        return review

    def history(self, gate: GateName | None = None) -> list[dict[str, object]]:
        filtered = self.reviews if gate is None else [r for r in self.reviews if r.gate == gate]
        return [
            {"gate": r.gate.value, "outcome": r.outcome.value, "score": round(r.composite_score, 3)} for r in filtered
        ]


def main() -> None:
    engine = GovernanceDecisionEngine()
    review = engine.create_review(GateName.G5_PIVOTAL_READY, "Dr. Smith")
    actuals = [1.0, 0.75, 0.85, 1.0, 1.0]
    for criterion, actual in zip(review.criteria, actuals):
        criterion.actual = actual
    result = engine.evaluate(review)
    print(f"Gate: {result.gate.value} | Outcome: {result.outcome.value} | Score: {result.composite_score:.3f}")


if __name__ == "__main__":
    main()
