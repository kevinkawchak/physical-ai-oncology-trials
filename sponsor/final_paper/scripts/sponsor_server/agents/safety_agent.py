"""Safety agent -- adverse-event classification, signal detection, and E2B reporting.

Monitors incoming patient data each simulation hour, classifies adverse
events by CTCAE grade, evaluates causality, detects aggregate safety
signals, and generates E2B-compatible reports for expedited regulatory
submission.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sponsor_server.models import (
    SafetyEscalation,
    SafetySeverity,
    _now_iso,
    source_data_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Causality categories (WHO-UMC system)
# ---------------------------------------------------------------------------
CAUSALITY_CATEGORIES: list[str] = [
    "certain",
    "probable",
    "possible",
    "unlikely",
    "conditional",
    "unassessable",
    "unrelated",
]

# Severity thresholds for automatic escalation
ESCALATION_GRADE_THRESHOLD = 3  # Grade >= 3 triggers escalation
SIGNAL_COUNT_THRESHOLD = 3  # N events of same term triggers signal review

# ---------------------------------------------------------------------------
# In-memory event store (per-process)
# ---------------------------------------------------------------------------
_ae_store: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_adverse_event(
    hour_id: int,
    ae_term: str,
    grade: int = 1,
    serious: bool = False,
    causality: str = "unrelated",
    patient_id: str = "",
) -> SafetyEscalation:
    """Classify a single adverse event and return an escalation record.

    Parameters
    ----------
    hour_id:
        Simulation hour (0-23).
    ae_term:
        MedDRA preferred term for the adverse event.
    grade:
        CTCAE grade (1-5).
    serious:
        Whether the event meets ICH seriousness criteria.
    causality:
        WHO-UMC causality category.
    patient_id:
        Anonymised patient identifier.

    Returns
    -------
    SafetyEscalation
    """
    severity = _grade_to_severity(grade)
    needs_e2b = serious or grade >= ESCALATION_GRADE_THRESHOLD
    action = _determine_action(grade, serious, causality)

    record: dict[str, Any] = {
        "hour_id": hour_id,
        "ae_term": ae_term,
        "grade": grade,
        "serious": serious,
        "causality": causality,
        "patient_id": patient_id,
    }
    _ae_store.append(record)

    escalation = SafetyEscalation(
        escalation_id=f"AE-{uuid.uuid4().hex[:8]}",
        hour_id=hour_id,
        ae_term=ae_term,
        severity=severity.value,
        seriousness=serious,
        causality=causality,
        action_taken=action,
        e2b_submitted=needs_e2b,
        reporter_agent="safety_agent",
        timestamp=_now_iso(),
    )
    logger.info(
        "AE classified hour=%d term=%s grade=%d serious=%s e2b=%s",
        hour_id,
        ae_term,
        grade,
        serious,
        needs_e2b,
    )
    return escalation


def detect_safety_signals(hour_id: int) -> list[dict[str, Any]]:
    """Scan accumulated AE data for aggregate safety signals.

    A signal is flagged when the same preferred term appears
    at least ``SIGNAL_COUNT_THRESHOLD`` times across all hours
    up to and including *hour_id*.

    Returns a list of signal dicts with term, count, and max grade.
    """
    relevant = [ae for ae in _ae_store if ae["hour_id"] <= hour_id]
    term_counts: dict[str, list[dict[str, Any]]] = {}
    for ae in relevant:
        term_counts.setdefault(ae["ae_term"], []).append(ae)

    signals: list[dict[str, Any]] = []
    for term, events in term_counts.items():
        if len(events) >= SIGNAL_COUNT_THRESHOLD:
            max_grade = max(e["grade"] for e in events)
            signals.append({
                "ae_term": term,
                "count": len(events),
                "max_grade": max_grade,
                "source": source_data_path(hour_id),
                "flagged_at": _now_iso(),
            })
            logger.warning("Safety signal: term=%s count=%d max_grade=%d", term, len(events), max_grade)
    return signals


def generate_e2b_stub(escalation: SafetyEscalation) -> dict[str, Any]:
    """Produce a minimal E2B(R3)-shaped dict for an escalation.

    This is a *stub* for demonstration; a production implementation
    would generate full ICH E2B(R3) XML.
    """
    return {
        "icsr_id": escalation.escalation_id,
        "ae_term": escalation.ae_term,
        "severity": escalation.severity,
        "seriousness": escalation.seriousness,
        "causality": escalation.causality,
        "reporter": escalation.reporter_agent,
        "report_type": "expedited" if escalation.seriousness else "routine",
        "generated_at": _now_iso(),
    }


def get_ae_summary(hour_id: int) -> dict[str, Any]:
    """Return aggregate AE statistics up to a given hour."""
    relevant = [ae for ae in _ae_store if ae["hour_id"] <= hour_id]
    grade_dist: dict[int, int] = {}
    for ae in relevant:
        grade_dist[ae["grade"]] = grade_dist.get(ae["grade"], 0) + 1
    return {
        "total_aes": len(relevant),
        "grade_distribution": grade_dist,
        "serious_count": sum(1 for ae in relevant if ae["serious"]),
        "through_hour": hour_id,
    }


def reset_store() -> None:
    """Clear the in-memory AE store (useful for testing)."""
    _ae_store.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _grade_to_severity(grade: int) -> SafetySeverity:
    """Map CTCAE numeric grade to ``SafetySeverity`` enum."""
    mapping = {
        1: SafetySeverity.GRADE_1,
        2: SafetySeverity.GRADE_2,
        3: SafetySeverity.GRADE_3,
        4: SafetySeverity.GRADE_4,
        5: SafetySeverity.GRADE_5,
    }
    return mapping.get(grade, SafetySeverity.GRADE_1)


def _determine_action(grade: int, serious: bool, causality: str) -> str:
    """Decide the recommended action based on event attributes."""
    if grade >= 4 or (serious and causality in ("certain", "probable")):
        return "hold_dosing_and_notify_dsmb"
    if grade >= 3:
        return "dose_modification_and_monitor"
    if serious:
        return "enhanced_monitoring"
    return "routine_follow_up"
