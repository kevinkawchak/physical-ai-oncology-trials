"""Patient interface: eConsent delivery, scheduling, ePRO collection."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum


class ConsentStatus(Enum):
    PENDING = "pending"
    VIEWED = "viewed"
    SIGNED = "signed"
    WITHDRAWN = "withdrawn"
    RE_CONSENTED = "re_consented"


class VisitStatus(Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    MISSED = "missed"
    RESCHEDULED = "rescheduled"


class PROInstrument(Enum):
    EORTC_QLQ_C30 = "EORTC_QLQ_C30"
    EQ_5D_5L = "EQ_5D_5L"
    PRO_CTCAE = "PRO_CTCAE"
    FACT_G = "FACT_G"
    PATIENT_GLOBAL_IMPRESSION = "PGI"


@dataclass
class EConsentDelivery:
    consent_id: str = field(default_factory=lambda: f"CON-{uuid.uuid4().hex[:6]}")
    subject_id: str = ""
    version: str = "1.0"
    language: str = "en"
    status: ConsentStatus = ConsentStatus.PENDING
    delivered_at: datetime = field(default_factory=datetime.utcnow)
    signed_at: datetime | None = None


@dataclass
class ScheduledVisit:
    visit_id: str = field(default_factory=lambda: f"VIS-{uuid.uuid4().hex[:6]}")
    subject_id: str = ""
    visit_name: str = ""
    visit_date: date = field(default_factory=date.today)
    window_days: int = 3
    status: VisitStatus = VisitStatus.SCHEDULED
    site_id: str = ""

    @property
    def window_start(self) -> date:
        return self.visit_date - timedelta(days=self.window_days)

    @property
    def window_end(self) -> date:
        return self.visit_date + timedelta(days=self.window_days)


@dataclass
class PROResponse:
    response_id: str = field(default_factory=lambda: f"PRO-{uuid.uuid4().hex[:6]}")
    subject_id: str = ""
    instrument: PROInstrument = PROInstrument.EORTC_QLQ_C30
    scores: dict[str, float] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)
    completion_pct: float = 1.0


class PatientInterface:
    """Manages eConsent delivery, visit scheduling, and ePRO collection for patients."""

    def __init__(self, study_id: str) -> None:
        self.study_id = study_id
        self.consents: dict[str, EConsentDelivery] = {}
        self.visits: list[ScheduledVisit] = []
        self.pro_responses: list[PROResponse] = []

    def deliver_consent(self, subject_id: str, version: str, language: str = "en") -> EConsentDelivery:
        consent = EConsentDelivery(subject_id=subject_id, version=version, language=language)
        self.consents[consent.consent_id] = consent
        return consent

    def sign_consent(self, consent_id: str) -> bool:
        consent = self.consents.get(consent_id)
        if consent and consent.status in (ConsentStatus.PENDING, ConsentStatus.VIEWED):
            consent.status = ConsentStatus.SIGNED
            consent.signed_at = datetime.utcnow()
            return True
        return False

    def withdraw_consent(self, consent_id: str) -> bool:
        consent = self.consents.get(consent_id)
        if consent and consent.status == ConsentStatus.SIGNED:
            consent.status = ConsentStatus.WITHDRAWN
            return True
        return False

    def schedule_visit(self, subject_id: str, visit_name: str, visit_date: date,
                       site_id: str, window_days: int = 3) -> ScheduledVisit:
        visit = ScheduledVisit(subject_id=subject_id, visit_name=visit_name,
                               visit_date=visit_date, site_id=site_id, window_days=window_days)
        self.visits.append(visit)
        return visit

    def upcoming_visits(self, subject_id: str, days: int = 14) -> list[ScheduledVisit]:
        cutoff = date.today() + timedelta(days=days)
        return [v for v in self.visits if v.subject_id == subject_id
                and v.status == VisitStatus.SCHEDULED and v.visit_date <= cutoff]

    def collect_pro(self, subject_id: str, instrument: PROInstrument,
                    scores: dict[str, float]) -> PROResponse:
        response = PROResponse(subject_id=subject_id, instrument=instrument, scores=scores)
        self.pro_responses.append(response)
        return response

    def pro_compliance(self, subject_id: str) -> float:
        responses = [r for r in self.pro_responses if r.subject_id == subject_id]
        if not responses:
            return 0.0
        return sum(r.completion_pct for r in responses) / len(responses)

    def summary(self) -> dict[str, object]:
        signed = sum(1 for c in self.consents.values() if c.status == ConsentStatus.SIGNED)
        return {"study": self.study_id, "consents_delivered": len(self.consents),
                "signed": signed, "visits_scheduled": len(self.visits),
                "pro_responses": len(self.pro_responses)}


def main() -> None:
    pi = PatientInterface("STU-001")
    consent = pi.deliver_consent("SUBJ-01", "2.0", "en")
    pi.sign_consent(consent.consent_id)
    pi.schedule_visit("SUBJ-01", "Screening", date.today() + timedelta(days=7), "SITE-01")
    pi.schedule_visit("SUBJ-01", "Cycle 1 Day 1", date.today() + timedelta(days=21), "SITE-01")
    pi.collect_pro("SUBJ-01", PROInstrument.EORTC_QLQ_C30, {"global_health": 75.0, "physical": 80.0})
    print(f"Upcoming: {len(pi.upcoming_visits('SUBJ-01'))}")
    print(f"PRO compliance: {pi.pro_compliance('SUBJ-01'):.0%}")
    print(f"Summary: {pi.summary()}")


if __name__ == "__main__":
    main()
