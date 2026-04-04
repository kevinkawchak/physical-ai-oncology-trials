"""Governance router -- sponsor-level decisions and programme governance.

Provides endpoints for the portfolio agent to issue go/no-go decisions
and for the study orchestrator to manage escalations.
"""
from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, HTTPException

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False

from sponsor_server.agents import portfolio_agent, study_orchestrator
from sponsor_server.models import SponsorDecision

# ---------------------------------------------------------------------------
# Router (only instantiated when FastAPI is available)
# ---------------------------------------------------------------------------
if _HAS_FASTAPI:
    router = APIRouter(prefix="/api/v1/sponsor", tags=["governance"])

    @router.post("/decision", response_model=None)
    async def create_decision(hour_id: int = 0, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Issue a portfolio-level go/no-go decision."""
        decision = portfolio_agent.evaluate_portfolio(hour_id, context)
        return _decision_to_dict(decision)

    @router.get("/escalations")
    async def list_escalations(hour_id: int | None = None) -> list[dict[str, Any]]:
        """List escalations, optionally filtered by hour."""
        return study_orchestrator.get_escalations(hour_id)

    @router.post("/escalations")
    async def create_escalation(
        hour_id: int = 0,
        source_agent: str = "manual",
        reason: str = "",
        severity: str = "high",
    ) -> dict[str, Any]:
        """Manually raise an escalation."""
        if not reason:
            raise HTTPException(status_code=422, detail="reason is required")
        return study_orchestrator.escalate(hour_id, source_agent, reason, severity)

    @router.post("/assets/rank")
    async def rank_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rank a list of pipeline assets by composite score."""
        return portfolio_agent.rank_assets(assets)

else:
    router = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Standalone helpers (usable without FastAPI)
# ---------------------------------------------------------------------------
def make_decision(hour_id: int = 0, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Standalone wrapper around the portfolio decision logic."""
    decision = portfolio_agent.evaluate_portfolio(hour_id, context)
    return _decision_to_dict(decision)


def _decision_to_dict(decision: SponsorDecision) -> dict[str, Any]:
    """Convert a SponsorDecision to a plain dict."""
    if hasattr(decision, "model_dump"):
        return decision.model_dump()  # type: ignore[union-attr]
    return {
        "decision_id": decision.decision_id,
        "hour_id": decision.hour_id,
        "agent": decision.agent,
        "outcome": decision.outcome,
        "rationale": decision.rationale,
        "confidence": decision.confidence,
        "timestamp": decision.timestamp,
        "evidence_refs": decision.evidence_refs,
    }
