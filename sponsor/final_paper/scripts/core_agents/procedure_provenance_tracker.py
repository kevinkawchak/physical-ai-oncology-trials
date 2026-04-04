"""Procedure provenance tracker: immutable procedure records, telemetry integration, regulatory reports."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ProvenanceEventType(Enum):
    PROCEDURE_START = "procedure_start"
    PHASE_CHANGE = "phase_change"
    TELEMETRY_SNAPSHOT = "telemetry_snapshot"
    SAFETY_ALERT = "safety_alert"
    HUMAN_OVERRIDE = "human_override"
    PROCEDURE_END = "procedure_end"


@dataclass
class ProvenanceEvent:
    event_id: str = field(default_factory=lambda: f"PE-{uuid.uuid4().hex[:8]}")
    event_type: ProvenanceEventType = ProvenanceEventType.PROCEDURE_START
    procedure_id: str = ""
    actor: str = ""
    detail: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    prev_hash: str = ""
    event_hash: str = ""


@dataclass
class ProcedureTrace:
    procedure_id: str
    robot_id: str
    subject_id: str
    events: list[ProvenanceEvent] = field(default_factory=list)
    sealed: bool = False


class ProcedureProvenanceTracker:
    """Immutable chain of procedure events with hash linking for regulatory traceability."""

    def __init__(self) -> None:
        self.traces: dict[str, ProcedureTrace] = {}

    def _compute_hash(self, event: ProvenanceEvent) -> str:
        payload = f"{event.event_id}|{event.event_type.value}|{event.procedure_id}|{event.actor}|{event.prev_hash}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def start_trace(self, procedure_id: str, robot_id: str, subject_id: str) -> ProcedureTrace:
        trace = ProcedureTrace(procedure_id=procedure_id, robot_id=robot_id, subject_id=subject_id)
        self.traces[procedure_id] = trace
        self.record_event(procedure_id, ProvenanceEventType.PROCEDURE_START, robot_id, "Procedure initiated")
        return trace

    def record_event(self, procedure_id: str, event_type: ProvenanceEventType,
                     actor: str, detail: str) -> ProvenanceEvent:
        trace = self.traces[procedure_id]
        if trace.sealed:
            raise ValueError(f"Trace {procedure_id} is sealed")
        prev_hash = trace.events[-1].event_hash if trace.events else "genesis"
        event = ProvenanceEvent(event_type=event_type, procedure_id=procedure_id,
                                actor=actor, detail=detail, prev_hash=prev_hash)
        event.event_hash = self._compute_hash(event)
        trace.events.append(event)
        return event

    def seal_trace(self, procedure_id: str) -> bool:
        trace = self.traces.get(procedure_id)
        if trace and not trace.sealed:
            self.record_event(procedure_id, ProvenanceEventType.PROCEDURE_END, "system", "Trace sealed")
            trace.sealed = True
            return True
        return False

    def verify_chain(self, procedure_id: str) -> bool:
        trace = self.traces[procedure_id]
        for i, event in enumerate(trace.events):
            expected_prev = trace.events[i - 1].event_hash if i > 0 else "genesis"
            if event.prev_hash != expected_prev:
                return False
        return True

    def generate_regulatory_report(self, procedure_id: str) -> dict[str, object]:
        trace = self.traces[procedure_id]
        return {
            "procedure_id": procedure_id, "robot_id": trace.robot_id,
            "subject_id": trace.subject_id, "sealed": trace.sealed,
            "total_events": len(trace.events), "chain_valid": self.verify_chain(procedure_id),
            "events": [{"type": e.event_type.value, "actor": e.actor,
                        "detail": e.detail, "hash": e.event_hash} for e in trace.events],
        }

    def summary(self) -> dict[str, object]:
        return {
            "total_traces": len(self.traces),
            "sealed": sum(1 for t in self.traces.values() if t.sealed),
            "total_events": sum(len(t.events) for t in self.traces.values()),
        }


def main() -> None:
    tracker = ProcedureProvenanceTracker()
    tracker.start_trace("PROC-001", "DVX-001", "SUBJ-01")
    tracker.record_event("PROC-001", ProvenanceEventType.PHASE_CHANGE, "DVX-001", "Docking complete")
    tracker.record_event("PROC-001", ProvenanceEventType.TELEMETRY_SNAPSHOT, "DVX-001", "Frame 100 captured")
    tracker.record_event("PROC-001", ProvenanceEventType.HUMAN_OVERRIDE, "SURG-01", "Manual correction applied")
    tracker.seal_trace("PROC-001")
    report = tracker.generate_regulatory_report("PROC-001")
    print(f"Chain valid: {report['chain_valid']}")
    print(f"Total events: {report['total_events']}")
    print(json.dumps(tracker.summary(), indent=2))


if __name__ == "__main__":
    main()
