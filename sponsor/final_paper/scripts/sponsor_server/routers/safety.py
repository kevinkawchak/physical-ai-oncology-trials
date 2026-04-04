"""Safety router -- adverse-event handling, escalation, and E2B reporting.

Wraps the safety agent's classification, signal detection, and E2B stub
generation behind FastAPI endpoints.
"""
from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False

from sponsor_server.agents import safety_agent
from sponsor_server.models import SafetyEscalation

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if _HAS_FASTAPI:
    router = APIRouter(prefix="/api/v1/sponsor/safety", tags=["safety"])

    @router.post("/ae/classify", response_model=None)
    async def classify_ae(
        hour_id: int = 0,
        ae_term: str = "",
        grade: int = 1,
        serious: bool = False,
        causality: str = "unrelated",
        patient_id: str = "",
    ) -> dict[str, Any]:
        """Classify an adverse event and return an escalation record."""
        escalation = safety_agent.classify_adverse_event(
            hour_id=hour_id,
            ae_term=ae_term,
            grade=grade,
            serious=serious,
            causality=causality,
            patient_id=patient_id,
        )
        return _escalation_to_dict(escalation)

    @router.get("/signals")
    async def detect_signals(hour_id: int = 0) -> list[dict[str, Any]]:
        """Run aggregate safety signal detection through a given hour."""
        return safety_agent.detect_safety_signals(hour_id)

    @router.post("/e2b/generate", response_model=None)
    async def generate_e2b(
        hour_id: int = 0,
        ae_term: str = "",
        grade: int = 1,
        serious: bool = True,
        causality: str = "possible",
        patient_id: str = "",
    ) -> dict[str, Any]:
        """Classify an AE and generate an E2B stub in one call."""
        escalation = safety_agent.classify_adverse_event(
            hour_id=hour_id,
            ae_term=ae_term,
            grade=grade,
            serious=serious,
            causality=causality,
            patient_id=patient_id,
        )
        return safety_agent.generate_e2b_stub(escalation)

    @router.get("/summary")
    async def ae_summary(hour_id: int = 0) -> dict[str, Any]:
        """Return aggregate AE statistics through a given hour."""
        return safety_agent.get_ae_summary(hour_id)

else:
    router = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------
def classify_ae_standalone(
    hour_id: int = 0,
    ae_term: str = "",
    grade: int = 1,
    serious: bool = False,
    causality: str = "unrelated",
    patient_id: str = "",
) -> dict[str, Any]:
    """Classify an AE without the server running."""
    escalation = safety_agent.classify_adverse_event(
        hour_id=hour_id,
        ae_term=ae_term,
        grade=grade,
        serious=serious,
        causality=causality,
        patient_id=patient_id,
    )
    return _escalation_to_dict(escalation)


def _escalation_to_dict(esc: SafetyEscalation) -> dict[str, Any]:
    """Convert a SafetyEscalation to a plain dict."""
    if hasattr(esc, "model_dump"):
        return esc.model_dump()  # type: ignore[union-attr]
    return {
        "escalation_id": esc.escalation_id,
        "hour_id": esc.hour_id,
        "ae_term": esc.ae_term,
        "severity": esc.severity,
        "seriousness": esc.seriousness,
        "causality": esc.causality,
        "action_taken": esc.action_taken,
        "e2b_submitted": esc.e2b_submitted,
        "reporter_agent": esc.reporter_agent,
        "timestamp": esc.timestamp,
    }
