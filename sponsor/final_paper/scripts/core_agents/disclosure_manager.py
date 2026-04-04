"""Disclosure manager: ClinicalTrials.gov posting, CTIS results submission."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class Registry(Enum):
    CLINICALTRIALS_GOV = "clinicaltrials_gov"
    CTIS = "ctis"
    EUCTR = "euctr"
    JAPIC = "japic"
    ANZCTR = "anzctr"


class DisclosureStatus(Enum):
    NOT_STARTED = "not_started"
    DRAFT = "draft"
    QC_REVIEW = "qc_review"
    SUBMITTED = "submitted"
    POSTED = "posted"
    UPDATE_NEEDED = "update_needed"


class ResultsSection(Enum):
    PARTICIPANT_FLOW = "participant_flow"
    BASELINE_CHARACTERISTICS = "baseline_characteristics"
    OUTCOME_MEASURES = "outcome_measures"
    ADVERSE_EVENTS = "adverse_events"
    LIMITATIONS = "limitations"


@dataclass
class RegistryEntry:
    registry: Registry
    nct_id: str = ""
    eudract_number: str = ""
    status: DisclosureStatus = DisclosureStatus.NOT_STARTED
    initial_post_date: date | None = None
    last_update_date: date | None = None
    results_due_date: date | None = None


@dataclass
class ResultsSubmission:
    submission_id: str
    registry: Registry
    study_id: str
    sections: dict[ResultsSection, bool] = field(default_factory=dict)
    submitted_date: date | None = None
    status: DisclosureStatus = DisclosureStatus.NOT_STARTED

    @property
    def completion_pct(self) -> float:
        if not self.sections:
            return 0.0
        return sum(1 for v in self.sections.values() if v) / len(self.sections) * 100


class DisclosureManager:
    """Manages trial registration and results disclosure across global registries."""

    def __init__(self, study_id: str) -> None:
        self.study_id = study_id
        self.entries: dict[Registry, RegistryEntry] = {}
        self.results: list[ResultsSubmission] = []

    def register_trial(self, registry: Registry, nct_id: str = "", eudract: str = "") -> RegistryEntry:
        entry = RegistryEntry(registry=registry, nct_id=nct_id, eudract_number=eudract)
        self.entries[registry] = entry
        return entry

    def post_registration(self, registry: Registry) -> None:
        entry = self.entries[registry]
        entry.status = DisclosureStatus.POSTED
        entry.initial_post_date = date.today()
        entry.results_due_date = date.today() + timedelta(days=365)

    def update_registration(self, registry: Registry) -> None:
        entry = self.entries[registry]
        entry.last_update_date = date.today()
        entry.status = DisclosureStatus.POSTED

    def start_results_submission(self, registry: Registry) -> ResultsSubmission:
        submission = ResultsSubmission(
            submission_id=f"RES-{self.study_id}-{registry.value}",
            registry=registry,
            study_id=self.study_id,
            sections={s: False for s in ResultsSection},
        )
        self.results.append(submission)
        return submission

    def complete_results_section(self, submission_id: str, section: ResultsSection) -> None:
        for sub in self.results:
            if sub.submission_id == submission_id:
                sub.sections[section] = True
                break

    def submit_results(self, submission_id: str) -> bool:
        for sub in self.results:
            if sub.submission_id == submission_id:
                if sub.completion_pct == 100.0:
                    sub.status = DisclosureStatus.SUBMITTED
                    sub.submitted_date = date.today()
                    return True
                return False
        return False

    def overdue_results(self) -> list[RegistryEntry]:
        today = date.today()
        return [
            e
            for e in self.entries.values()
            if e.results_due_date
            and e.results_due_date < today
            and not any(r.status == DisclosureStatus.SUBMITTED for r in self.results if r.registry == e.registry)
        ]

    def disclosure_dashboard(self) -> dict[str, object]:
        return {
            "study": self.study_id,
            "registries": len(self.entries),
            "posted": sum(1 for e in self.entries.values() if e.status == DisclosureStatus.POSTED),
            "results_submissions": len(self.results),
            "overdue": len(self.overdue_results()),
        }


def main() -> None:
    mgr = DisclosureManager("STU-001")
    mgr.register_trial(Registry.CLINICALTRIALS_GOV, nct_id="NCT12345678")
    mgr.register_trial(Registry.CTIS, eudract="2026-001234-56")
    mgr.post_registration(Registry.CLINICALTRIALS_GOV)
    sub = mgr.start_results_submission(Registry.CLINICALTRIALS_GOV)
    for section in ResultsSection:
        mgr.complete_results_section(sub.submission_id, section)
    success = mgr.submit_results(sub.submission_id)
    print(f"Results submitted: {success}")
    print(f"Dashboard: {mgr.disclosure_dashboard()}")


if __name__ == "__main__":
    main()
