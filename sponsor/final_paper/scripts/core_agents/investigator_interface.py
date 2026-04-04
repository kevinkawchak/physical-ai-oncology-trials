"""Investigator interface: protocol training, delegation log, AE reporting workflows."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class DelegationRole(Enum):
    PRINCIPAL_INVESTIGATOR = "principal_investigator"
    SUB_INVESTIGATOR = "sub_investigator"
    STUDY_COORDINATOR = "study_coordinator"
    PHARMACIST = "pharmacist"
    NURSE = "nurse"
    DATA_ENTRY = "data_entry"


class TrainingStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"


class AEReportStatus(Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    QUERIED = "queried"
    CLOSED = "closed"


@dataclass
class TrainingRecord:
    record_id: str = field(default_factory=lambda: f"TR-{uuid.uuid4().hex[:6]}")
    personnel_id: str = ""
    module_name: str = ""
    status: TrainingStatus = TrainingStatus.NOT_STARTED
    score: float = 0.0
    completed_date: date | None = None
    expiry_date: date | None = None


@dataclass
class DelegationEntry:
    personnel_id: str
    name: str
    role: DelegationRole
    tasks: list[str] = field(default_factory=list)
    delegated_date: date = field(default_factory=date.today)
    revoked: bool = False


@dataclass
class AEReport:
    report_id: str = field(default_factory=lambda: f"AER-{uuid.uuid4().hex[:6]}")
    subject_id: str = ""
    ae_term: str = ""
    onset_date: date | None = None
    severity: str = ""
    serious: bool = False
    status: AEReportStatus = AEReportStatus.DRAFT
    reporter: str = ""
    submitted_at: datetime | None = None


class InvestigatorInterface:
    """Manages investigator training, delegation logs, and AE reporting workflows."""

    REQUIRED_MODULES: list[str] = [
        "GCP Refresher",
        "Protocol-Specific Training",
        "EDC System Training",
        "Safety Reporting Procedures",
        "Robotic Procedure Familiarization",
    ]

    def __init__(self, site_id: str, pi_name: str) -> None:
        self.site_id = site_id
        self.pi_name = pi_name
        self.training: list[TrainingRecord] = []
        self.delegation_log: list[DelegationEntry] = []
        self.ae_reports: list[AEReport] = []

    def assign_training(self, personnel_id: str) -> list[TrainingRecord]:
        records: list[TrainingRecord] = []
        for module in self.REQUIRED_MODULES:
            rec = TrainingRecord(personnel_id=personnel_id, module_name=module)
            self.training.append(rec)
            records.append(rec)
        return records

    def complete_training(self, record_id: str, score: float) -> bool:
        for rec in self.training:
            if rec.record_id == record_id:
                rec.score = score
                if score >= 80.0:
                    rec.status = TrainingStatus.COMPLETED
                    rec.completed_date = date.today()
                    return True
                rec.status = TrainingStatus.IN_PROGRESS
                return False
        return False

    def is_fully_trained(self, personnel_id: str) -> bool:
        recs = [r for r in self.training if r.personnel_id == personnel_id]
        return len(recs) >= len(self.REQUIRED_MODULES) and all(r.status == TrainingStatus.COMPLETED for r in recs)

    def add_delegation(self, personnel_id: str, name: str, role: DelegationRole, tasks: list[str]) -> DelegationEntry:
        entry = DelegationEntry(personnel_id=personnel_id, name=name, role=role, tasks=tasks)
        self.delegation_log.append(entry)
        return entry

    def revoke_delegation(self, personnel_id: str) -> bool:
        for entry in self.delegation_log:
            if entry.personnel_id == personnel_id and not entry.revoked:
                entry.revoked = True
                return True
        return False

    def active_delegation(self) -> list[DelegationEntry]:
        return [e for e in self.delegation_log if not e.revoked]

    def submit_ae_report(self, subject_id: str, ae_term: str, severity: str, serious: bool, reporter: str) -> AEReport:
        report = AEReport(
            subject_id=subject_id,
            ae_term=ae_term,
            severity=severity,
            serious=serious,
            reporter=reporter,
            onset_date=date.today(),
            status=AEReportStatus.SUBMITTED,
            submitted_at=datetime.utcnow(),
        )
        self.ae_reports.append(report)
        return report

    def pending_ae_reports(self) -> list[AEReport]:
        return [r for r in self.ae_reports if r.status in (AEReportStatus.DRAFT, AEReportStatus.QUERIED)]

    def summary(self) -> dict[str, object]:
        active_staff = len({e.personnel_id for e in self.active_delegation()})
        return {
            "site_id": self.site_id,
            "pi": self.pi_name,
            "active_staff": active_staff,
            "training_complete": sum(1 for t in self.training if t.status == TrainingStatus.COMPLETED),
            "ae_reports": len(self.ae_reports),
            "pending_aes": len(self.pending_ae_reports()),
        }


def main() -> None:
    iface = InvestigatorInterface("SITE-01", "Dr. Nakamura")
    records = iface.assign_training("PI-001")
    for rec in records:
        iface.complete_training(rec.record_id, 92.0)
    iface.add_delegation(
        "PI-001", "Dr. Nakamura", DelegationRole.PRINCIPAL_INVESTIGATOR, ["consent", "ae_reporting", "oversight"]
    )
    iface.add_delegation("SC-001", "Jane Doe", DelegationRole.STUDY_COORDINATOR, ["data_entry", "scheduling"])
    iface.submit_ae_report("SUBJ-01", "Nausea", "Moderate", False, "PI-001")
    print(f"Fully trained: {iface.is_fully_trained('PI-001')}")
    print(f"Summary: {iface.summary()}")


if __name__ == "__main__":
    main()
