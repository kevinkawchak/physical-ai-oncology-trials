"""Portfolio agent -- programme prioritisation and go/no-go decisions.

Evaluates pipeline assets, weighs clinical, regulatory, and commercial
factors, and issues go / no-go / conditional / escalate decisions that
flow into the study orchestrator.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sponsor_server.models import (
    DecisionOutcome,
    SponsorDecision,
    _now_iso,
    source_data_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights used by the portfolio model
# ---------------------------------------------------------------------------
SCORING_WEIGHTS: dict[str, float] = {
    "unmet_medical_need": 0.25,
    "probability_of_success": 0.20,
    "competitive_landscape": 0.15,
    "regulatory_pathway": 0.15,
    "safety_profile": 0.15,
    "commercial_potential": 0.10,
}

GO_THRESHOLD = 0.65
CONDITIONAL_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate_portfolio(hour_id: int, context: dict[str, Any] | None = None) -> SponsorDecision:
    """Run the portfolio scoring model and return a sponsor decision.

    Parameters
    ----------
    hour_id:
        Simulation hour (0-23).
    context:
        Optional dict of external signals (enrollment pace, safety flags, etc.).

    Returns
    -------
    SponsorDecision
    """
    ctx = context or {}
    scores = _compute_scores(hour_id, ctx)
    composite = sum(SCORING_WEIGHTS[k] * scores.get(k, 0.5) for k in SCORING_WEIGHTS)

    if composite >= GO_THRESHOLD:
        outcome = DecisionOutcome.GO
    elif composite >= CONDITIONAL_THRESHOLD:
        outcome = DecisionOutcome.CONDITIONAL
    else:
        outcome = DecisionOutcome.NO_GO

    # If a safety escalation is flagged in context, override to ESCALATE.
    if ctx.get("safety_escalation"):
        outcome = DecisionOutcome.ESCALATE

    decision = SponsorDecision(
        decision_id=f"PF-{uuid.uuid4().hex[:8]}",
        hour_id=hour_id,
        agent="portfolio_agent",
        outcome=outcome.value,
        rationale=_build_rationale(composite, scores, outcome),
        confidence=round(composite, 4),
        timestamp=_now_iso(),
        evidence_refs=[source_data_path(hour_id)],
    )
    logger.info("Portfolio decision hour=%d outcome=%s confidence=%.3f", hour_id, outcome.value, composite)
    return decision


def rank_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return *assets* sorted by descending composite score.

    Each asset dict is expected to carry keys matching ``SCORING_WEIGHTS``.
    A ``_composite`` key is added to every returned dict.
    """
    for asset in assets:
        asset["_composite"] = sum(SCORING_WEIGHTS[k] * asset.get(k, 0.5) for k in SCORING_WEIGHTS)
    return sorted(assets, key=lambda a: a["_composite"], reverse=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _compute_scores(hour_id: int, ctx: dict[str, Any]) -> dict[str, float]:
    """Derive per-dimension scores from simulation context.

    In a full deployment these would come from real-time data feeds;
    here we use deterministic defaults modulated by hour and context.
    """
    base: dict[str, float] = {k: 0.7 for k in SCORING_WEIGHTS}

    # Enrollment pace affects probability of success
    enrollment_rate = ctx.get("enrollment_rate", 1.0)
    base["probability_of_success"] = min(1.0, 0.5 + 0.2 * enrollment_rate)

    # Late hours imply more data -> higher confidence
    hour_factor = min(hour_id / 23.0, 1.0)
    base["regulatory_pathway"] = 0.6 + 0.2 * hour_factor

    # Safety flags depress the safety-profile dimension
    if ctx.get("ae_count", 0) > 3:
        base["safety_profile"] = max(0.2, base["safety_profile"] - 0.1 * ctx["ae_count"])

    return base


def _build_rationale(composite: float, scores: dict[str, float], outcome: DecisionOutcome) -> str:
    """Produce a human-readable rationale string."""
    top_driver = max(scores, key=lambda k: SCORING_WEIGHTS[k] * scores[k])
    return (
        f"Composite score {composite:.3f} -> {outcome.value}. "
        f"Leading driver: {top_driver} ({scores[top_driver]:.2f})."
    )
