"""Audit trail manager: immutable audit trails, 21 CFR Part 11 compliance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EventCategory(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SIGN = "electronic_signature"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"


class Part11Control(Enum):
    AUDIT_TRAIL = "audit_trail"
    ELECTRONIC_SIGNATURE = "electronic_signature"
    ACCESS_CONTROL = "access_control"
    DATA_INTEGRITY = "data_integrity"
    SYSTEM_VALIDATION = "system_validation"
    BACKUP_RECOVERY = "backup_recovery"


@dataclass
class AuditEvent:
    event_id: str = ""
    category: EventCategory = EventCategory.CREATE
    user_id: str = ""
    resource_type: str = ""
    resource_id: str = ""
    old_value: str = ""
    new_value: str = ""
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    prev_hash: str = ""
    event_hash: str = ""


@dataclass
class Part11Assessment:
    control: Part11Control
    implemented: bool = False
    evidence: str = ""
    assessed_date: datetime = field(default_factory=datetime.utcnow)


class AuditTrailManager:
    """Manages immutable audit trails with hash chaining for 21 CFR Part 11 compliance."""

    def __init__(self, system_name: str) -> None:
        self.system_name = system_name
        self.events: list[AuditEvent] = []
        self.part11_assessments: list[Part11Assessment] = []
        self._counter = 0

    def _compute_hash(self, event: AuditEvent) -> str:
        payload = json.dumps(
            {
                "id": event.event_id,
                "user": event.user_id,
                "category": event.category.value,
                "resource": event.resource_id,
                "new": event.new_value,
                "prev": event.prev_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def record(
        self,
        category: EventCategory,
        user_id: str,
        resource_type: str,
        resource_id: str,
        old_value: str = "",
        new_value: str = "",
        reason: str = "",
    ) -> AuditEvent:
        self._counter += 1
        prev_hash = self.events[-1].event_hash if self.events else "genesis"
        event = AuditEvent(
            event_id=f"AE-{self._counter:06d}",
            category=category,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            prev_hash=prev_hash,
        )
        event.event_hash = self._compute_hash(event)
        self.events.append(event)
        return event

    def verify_chain(self) -> bool:
        for i, event in enumerate(self.events):
            expected_prev = self.events[i - 1].event_hash if i > 0 else "genesis"
            if event.prev_hash != expected_prev:
                return False
            if event.event_hash != self._compute_hash(event):
                return False
        return True

    def query(
        self,
        user_id: str | None = None,
        category: EventCategory | None = None,
        resource_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        results = self.events
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if category:
            results = [e for e in results if e.category == category]
        if resource_id:
            results = [e for e in results if e.resource_id == resource_id]
        return results[-limit:]

    def assess_part11(self, control: Part11Control, implemented: bool, evidence: str) -> Part11Assessment:
        assessment = Part11Assessment(control=control, implemented=implemented, evidence=evidence)
        self.part11_assessments.append(assessment)
        return assessment

    def part11_compliance_score(self) -> float:
        if not self.part11_assessments:
            return 0.0
        return sum(1 for a in self.part11_assessments if a.implemented) / len(self.part11_assessments)

    def summary(self) -> dict[str, object]:
        return {
            "system": self.system_name,
            "total_events": len(self.events),
            "chain_valid": self.verify_chain(),
            "part11_score": round(self.part11_compliance_score(), 3),
            "users": len({e.user_id for e in self.events}),
        }


def main() -> None:
    mgr = AuditTrailManager("PhysicalAI Platform")
    mgr.record(EventCategory.CREATE, "user1", "Protocol", "PROTO-001", new_value="v1.0")
    mgr.record(EventCategory.UPDATE, "user1", "Protocol", "PROTO-001", "v1.0", "v2.0", "Amendment 1")
    mgr.record(EventCategory.SIGN, "user2", "Protocol", "PROTO-001", new_value="approved")
    mgr.assess_part11(Part11Control.AUDIT_TRAIL, True, "Hash-chained audit log")
    mgr.assess_part11(Part11Control.ELECTRONIC_SIGNATURE, True, "PKI-based e-signatures")
    mgr.assess_part11(Part11Control.ACCESS_CONTROL, True, "RBAC implemented")
    print(f"Chain valid: {mgr.verify_chain()}")
    print(f"Summary: {mgr.summary()}")


if __name__ == "__main__":
    main()
