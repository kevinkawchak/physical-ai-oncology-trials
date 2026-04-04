"""Clinical accountability agent: eligibility rules, dose-schedule optimization."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CriterionType(Enum):
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


class DoseSchedule(Enum):
    QD = "once_daily"
    BID = "twice_daily"
    QW = "once_weekly"
    Q2W = "every_2_weeks"
    Q3W = "every_3_weeks"


@dataclass
class EligibilityCriterion:
    criterion_id: str
    criterion_type: CriterionType
    description: str
    parameter: str
    operator: str  # >=, <=, ==, !=, in, not_in
    value: object

    def evaluate(self, patient_data: dict[str, object]) -> bool:
        actual = patient_data.get(self.parameter)
        if actual is None:
            return False
        ops: dict[str, object] = {
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "in": lambda a, b: a in b,
            "not_in": lambda a, b: a not in b,
        }
        fn = ops.get(self.operator)
        if fn is None:
            return False
        return fn(actual, self.value)  # type: ignore[operator]


@dataclass
class DoseOption:
    dose_mg: float
    schedule: DoseSchedule
    predicted_efficacy: float
    predicted_toxicity_grade3_plus: float
    convenience_score: float = 0.0

    @property
    def utility(self) -> float:
        return self.predicted_efficacy * 0.5 + (1 - self.predicted_toxicity_grade3_plus) * 0.3 + self.convenience_score * 0.2


@dataclass
class ScreeningResult:
    patient_id: str
    eligible: bool
    failed_criteria: list[str] = field(default_factory=list)
    passed_criteria: list[str] = field(default_factory=list)


class ClinicalAccountabilityAgent:
    """Manages eligibility screening and dose-schedule optimization."""

    def __init__(self, protocol_id: str) -> None:
        self.protocol_id = protocol_id
        self.criteria: list[EligibilityCriterion] = []
        self.dose_options: list[DoseOption] = []

    def add_criterion(self, criterion: EligibilityCriterion) -> None:
        self.criteria.append(criterion)

    def screen_patient(self, patient_id: str, patient_data: dict[str, object]) -> ScreeningResult:
        failed: list[str] = []
        passed: list[str] = []
        for c in self.criteria:
            result = c.evaluate(patient_data)
            expected = c.criterion_type == CriterionType.INCLUSION
            if result == expected:
                passed.append(c.criterion_id)
            else:
                failed.append(c.criterion_id)
        return ScreeningResult(patient_id=patient_id, eligible=len(failed) == 0, failed_criteria=failed, passed_criteria=passed)

    def add_dose_option(self, option: DoseOption) -> None:
        self.dose_options.append(option)

    def rank_doses(self) -> list[DoseOption]:
        return sorted(self.dose_options, key=lambda d: d.utility, reverse=True)

    def recommended_dose(self) -> DoseOption | None:
        ranked = self.rank_doses()
        return ranked[0] if ranked else None

    def build_standard_oncology_criteria(self) -> None:
        defaults = [
            EligibilityCriterion("INC-01", CriterionType.INCLUSION, "Age >= 18", "age", ">=", 18),
            EligibilityCriterion("INC-02", CriterionType.INCLUSION, "ECOG PS 0-1", "ecog", "<=", 1),
            EligibilityCriterion("INC-03", CriterionType.INCLUSION, "Adequate hepatic", "alt_iu_l", "<=", 150),
            EligibilityCriterion("INC-04", CriterionType.INCLUSION, "Adequate renal", "creatinine_clearance", ">=", 60),
            EligibilityCriterion("EXC-01", CriterionType.EXCLUSION, "Active brain mets", "active_brain_mets", "==", True),
        ]
        for c in defaults:
            self.add_criterion(c)


def main() -> None:
    agent = ClinicalAccountabilityAgent("PROTO-2026-001")
    agent.build_standard_oncology_criteria()
    patient = {"age": 62, "ecog": 1, "alt_iu_l": 45, "creatinine_clearance": 85, "active_brain_mets": False}
    result = agent.screen_patient("PAT-001", patient)
    print(f"Patient {result.patient_id}: eligible={result.eligible}, failed={result.failed_criteria}")
    agent.add_dose_option(DoseOption(200, DoseSchedule.Q3W, 0.55, 0.12, 0.8))
    agent.add_dose_option(DoseOption(400, DoseSchedule.Q3W, 0.70, 0.28, 0.8))
    rec = agent.recommended_dose()
    if rec:
        print(f"Recommended: {rec.dose_mg}mg {rec.schedule.value} (utility={rec.utility:.3f})")


if __name__ == "__main__":
    main()
