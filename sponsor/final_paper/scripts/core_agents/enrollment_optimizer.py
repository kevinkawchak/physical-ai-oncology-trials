"""Enrollment optimizer: enrollment prediction, EHR pre-screening, diversity tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta


@dataclass
class DemographicTarget:
    category: str
    target_pct: float
    actual_count: int = 0
    total_enrolled: int = 0

    @property
    def actual_pct(self) -> float:
        if self.total_enrolled == 0:
            return 0.0
        return self.actual_count / self.total_enrolled

    @property
    def gap(self) -> float:
        return self.target_pct - self.actual_pct


@dataclass
class EHRPreScreenHit:
    patient_id: str
    site_id: str
    match_score: float
    criteria_met: list[str] = field(default_factory=list)
    criteria_unmet: list[str] = field(default_factory=list)


@dataclass
class EnrollmentProjection:
    site_id: str
    current_enrolled: int
    monthly_rate: float
    target: int
    projected_completion: date | None = None

    def compute_completion(self, as_of: date | None = None) -> date:
        ref = as_of or date.today()
        remaining = self.target - self.current_enrolled
        if self.monthly_rate <= 0 or remaining <= 0:
            self.projected_completion = ref
            return ref
        months_needed = math.ceil(remaining / self.monthly_rate)
        self.projected_completion = ref + timedelta(days=months_needed * 30)
        return self.projected_completion


@dataclass
class DiversityReport:
    study_id: str
    total_enrolled: int
    targets: list[DemographicTarget] = field(default_factory=list)

    @property
    def meets_all_targets(self) -> bool:
        return all(t.gap <= 0.02 for t in self.targets)

    @property
    def underrepresented(self) -> list[str]:
        return [t.category for t in self.targets if t.gap > 0.05]


class EnrollmentOptimizer:
    """Predicts enrollment timelines, runs EHR pre-screening, and tracks diversity."""

    def __init__(self, study_id: str, total_target: int) -> None:
        self.study_id = study_id
        self.total_target = total_target
        self.projections: dict[str, EnrollmentProjection] = {}
        self.prescreened: list[EHRPreScreenHit] = []
        self.diversity_targets: list[DemographicTarget] = []

    def add_site_projection(self, proj: EnrollmentProjection) -> None:
        proj.compute_completion()
        self.projections[proj.site_id] = proj

    def study_completion_date(self) -> date | None:
        dates = [p.projected_completion for p in self.projections.values() if p.projected_completion]
        return max(dates) if dates else None

    def enrollment_velocity(self) -> float:
        return sum(p.monthly_rate for p in self.projections.values())

    def run_ehr_prescreen(
        self, patients: list[dict[str, object]], required_criteria: list[str]
    ) -> list[EHRPreScreenHit]:
        hits: list[EHRPreScreenHit] = []
        for pt in patients:
            met = [c for c in required_criteria if pt.get(c)]
            unmet = [c for c in required_criteria if not pt.get(c)]
            score = len(met) / max(len(required_criteria), 1)
            if score >= 0.6:
                hits.append(
                    EHRPreScreenHit(
                        patient_id=str(pt.get("id", "")),
                        site_id=str(pt.get("site", "")),
                        match_score=round(score, 2),
                        criteria_met=met,
                        criteria_unmet=unmet,
                    )
                )
        self.prescreened = hits
        return hits

    def set_diversity_targets(self) -> None:
        self.diversity_targets = [
            DemographicTarget("Black/African American", 0.13),
            DemographicTarget("Hispanic/Latino", 0.18),
            DemographicTarget("Asian", 0.06),
            DemographicTarget("Female", 0.50),
            DemographicTarget("Age >= 65", 0.35),
        ]

    def update_diversity(self, category: str, count: int, total: int) -> None:
        for t in self.diversity_targets:
            if t.category == category:
                t.actual_count = count
                t.total_enrolled = total

    def diversity_report(self) -> DiversityReport:
        total = max(t.total_enrolled for t in self.diversity_targets) if self.diversity_targets else 0
        return DiversityReport(study_id=self.study_id, total_enrolled=total, targets=self.diversity_targets)


def main() -> None:
    opt = EnrollmentOptimizer("STU-001", 200)
    opt.add_site_projection(EnrollmentProjection("S01", 15, 3.5, 40))
    opt.add_site_projection(EnrollmentProjection("S02", 8, 2.0, 35))
    print(f"Velocity: {opt.enrollment_velocity()} pts/mo, Completion: {opt.study_completion_date()}")
    patients = [
        {"id": "P1", "site": "S01", "age_gte_18": True, "ecog_01": True, "measurable_disease": True},
        {"id": "P2", "site": "S01", "age_gte_18": True, "ecog_01": False, "measurable_disease": True},
    ]
    hits = opt.run_ehr_prescreen(patients, ["age_gte_18", "ecog_01", "measurable_disease"])
    print(f"Pre-screen hits: {len(hits)}")
    opt.set_diversity_targets()
    opt.update_diversity("Black/African American", 5, 50)
    report = opt.diversity_report()
    print(f"Underrepresented: {report.underrepresented}")


if __name__ == "__main__":
    main()
