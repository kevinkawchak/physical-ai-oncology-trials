"""Data management agent: eConsent tracking, eSource validation, EDC management, query automation."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ConsentStatus(Enum):
    PENDING = "pending"
    SIGNED = "signed"
    WITHDRAWN = "withdrawn"
    RE_CONSENTED = "re_consented"


class QueryStatus(Enum):
    OPEN = "open"
    ANSWERED = "answered"
    CLOSED = "closed"


@dataclass
class EConsentRecord:
    subject_id: str
    version: str
    status: ConsentStatus = ConsentStatus.PENDING
    signed_at: datetime | None = None
    ip_hash: str = ""


@dataclass
class ESourceField:
    field_id: str
    form_name: str
    value: str = ""
    source_verified: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataQuery:
    query_id: str = field(default_factory=lambda: f"Q-{uuid.uuid4().hex[:8]}")
    subject_id: str = ""
    field_id: str = ""
    message: str = ""
    status: QueryStatus = QueryStatus.OPEN
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: datetime | None = None


class DataManagementAgent:
    """Manages eConsent lifecycle, eSource validation, EDC fields, and automated queries."""

    def __init__(self, study_id: str) -> None:
        self.study_id = study_id
        self.consents: dict[str, EConsentRecord] = {}
        self.fields: list[ESourceField] = []
        self.queries: list[DataQuery] = []

    def record_consent(self, subject_id: str, version: str) -> EConsentRecord:
        rec = EConsentRecord(subject_id=subject_id, version=version,
                             status=ConsentStatus.SIGNED, signed_at=datetime.utcnow())
        self.consents[subject_id] = rec
        return rec

    def withdraw_consent(self, subject_id: str) -> bool:
        rec = self.consents.get(subject_id)
        if rec and rec.status == ConsentStatus.SIGNED:
            rec.status = ConsentStatus.WITHDRAWN
            return True
        return False

    def validate_esource(self, fields: list[ESourceField]) -> list[DataQuery]:
        auto_queries: list[DataQuery] = []
        for f in fields:
            self.fields.append(f)
            if not f.value.strip():
                q = DataQuery(subject_id="", field_id=f.field_id, message=f"Missing value in {f.form_name}")
                self.queries.append(q)
                auto_queries.append(q)
            elif not f.source_verified:
                q = DataQuery(subject_id="", field_id=f.field_id, message=f"Source not verified for {f.field_id}")
                self.queries.append(q)
                auto_queries.append(q)
        return auto_queries

    def close_query(self, query_id: str, answer: str) -> bool:
        for q in self.queries:
            if q.query_id == query_id:
                q.status = QueryStatus.CLOSED
                q.closed_at = datetime.utcnow()
                q.message += f" | Answer: {answer}"
                return True
        return False

    def query_rate(self) -> dict[str, int]:
        return {s.value: sum(1 for q in self.queries if q.status == s) for s in QueryStatus}

    def summary(self) -> dict[str, object]:
        consented = sum(1 for c in self.consents.values() if c.status == ConsentStatus.SIGNED)
        return {"study": self.study_id, "consented": consented, "fields": len(self.fields),
                "queries": self.query_rate()}


def main() -> None:
    agent = DataManagementAgent("STU-001")
    agent.record_consent("SUBJ-01", "v2.0")
    agent.record_consent("SUBJ-02", "v2.0")
    fields = [
        ESourceField("F1", "Demographics", "35"),
        ESourceField("F2", "Vitals", ""),
        ESourceField("F3", "Labs", "7.2", source_verified=False),
    ]
    queries = agent.validate_esource(fields)
    print(f"Auto-generated queries: {len(queries)}")
    if queries:
        agent.close_query(queries[0].query_id, "Value entered: 120/80")
    print(f"Summary: {agent.summary()}")


if __name__ == "__main__":
    main()
