from __future__ import annotations

"""Five-level escalation model for autonomous sponsor decisions.

Maps a decision context (confidence score, decision type, safety
implications) to one of five escalation levels, each with its own
approval requirements and timeout semantics.

Levels
------
1. Informational   - Log and continue; no approval needed.
2. Confirmation    - Pause for peer-agent validation; 5-min timeout.
3. Clinical        - Clinician approval required; 30-min timeout.
4. Regulatory      - Sponsor-of-record signature; 4-hour timeout.
5. Safety-critical - Mandatory human review; no timeout (blocks).
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EscalationLevel(IntEnum):
    INFORMATIONAL = 1
    CONFIRMATION = 2
    CLINICAL = 3
    REGULATORY = 4
    SAFETY_CRITICAL = 5


class DecisionType(str):
    """Lightweight marker—values are free-form strings."""


# ---------------------------------------------------------------------------
# Configuration per level
# ---------------------------------------------------------------------------

LEVEL_CONFIG: dict[EscalationLevel, dict[str, object]] = {
    EscalationLevel.INFORMATIONAL: {
        "label": "Informational",
        "required_approvals": [],
        "timeout_seconds": None,
        "action": "Log and continue",
    },
    EscalationLevel.CONFIRMATION: {
        "label": "Confirmation",
        "required_approvals": ["peer_agent"],
        "timeout_seconds": 300,
        "action": "Pause, peer validation, 5-min timeout",
    },
    EscalationLevel.CLINICAL: {
        "label": "Clinical",
        "required_approvals": ["clinician"],
        "timeout_seconds": 1800,
        "action": "Clinician approval, 30-min timeout",
    },
    EscalationLevel.REGULATORY: {
        "label": "Regulatory",
        "required_approvals": ["sponsor_of_record"],
        "timeout_seconds": 14400,
        "action": "Sponsor-of-record signature, 4-hour timeout",
    },
    EscalationLevel.SAFETY_CRITICAL: {
        "label": "Safety-critical",
        "required_approvals": ["human_reviewer", "safety_officer"],
        "timeout_seconds": None,
        "action": "Mandatory human review, no timeout",
    },
}


# ---------------------------------------------------------------------------
# Decision context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionContext:
    """Input bundle for the escalation engine."""

    decision_id: str
    decision_type: str
    confidence_score: float  # 0.0 – 1.0
    safety_relevant: bool
    regulatory_relevant: bool
    patient_impact: bool
    source_agent: str
    description: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        decision_type: str,
        confidence_score: float,
        source_agent: str,
        *,
        safety_relevant: bool = False,
        regulatory_relevant: bool = False,
        patient_impact: bool = False,
        description: str = "",
        metadata: dict[str, object] | None = None,
    ) -> DecisionContext:
        return cls(
            decision_id=uuid.uuid4().hex[:12],
            decision_type=decision_type,
            confidence_score=max(0.0, min(1.0, confidence_score)),
            safety_relevant=safety_relevant,
            regulatory_relevant=regulatory_relevant,
            patient_impact=patient_impact,
            source_agent=source_agent,
            description=description,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Escalation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EscalationResult:
    """Output of the escalation engine for a single decision."""

    decision_id: str
    level: EscalationLevel
    level_label: str
    required_approvals: list[str]
    timeout_seconds: int | None
    action: str
    rationale: str
    evaluated_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "level": self.level.value,
            "level_label": self.level_label,
            "required_approvals": self.required_approvals,
            "timeout_seconds": self.timeout_seconds,
            "action": self.action,
            "rationale": self.rationale,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Escalation rules
# ---------------------------------------------------------------------------

# Confidence thresholds (lower confidence => higher escalation)
_CONF_HIGH = 0.90
_CONF_MEDIUM = 0.70
_CONF_LOW = 0.50

# Decision types that always elevate
_SAFETY_TYPES = frozenset(
    {
        "sae_review",
        "dose_modification",
        "study_halt",
        "safety_signal",
        "robot_halt",
        "emergency_unblind",
    }
)

_REGULATORY_TYPES = frozenset(
    {
        "ind_amendment",
        "protocol_amendment",
        "dmc_response",
        "regulatory_filing",
        "annual_report",
    }
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EscalationEngine:
    """Determine the escalation level for a given decision context."""

    def __init__(self) -> None:
        self._history: list[EscalationResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, ctx: DecisionContext) -> EscalationResult:
        """Return the escalation result for *ctx*."""
        level = self._compute_level(ctx)
        cfg = LEVEL_CONFIG[level]
        rationale = self._build_rationale(ctx, level)

        result = EscalationResult(
            decision_id=ctx.decision_id,
            level=level,
            level_label=str(cfg["label"]),
            required_approvals=list(cfg["required_approvals"]),  # type: ignore[arg-type]
            timeout_seconds=cfg["timeout_seconds"],  # type: ignore[arg-type]
            action=str(cfg["action"]),
            rationale=rationale,
            evaluated_at=datetime.now(timezone.utc),
        )
        self._history.append(result)
        logger.info(
            "Decision %s escalated to L%d (%s): %s",
            ctx.decision_id,
            level.value,
            result.level_label,
            rationale,
        )
        return result

    def history(self, limit: int = 50) -> list[EscalationResult]:
        return list(reversed(self._history[-limit:]))

    def level_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {lvl.name: 0 for lvl in EscalationLevel}
        for r in self._history:
            counts[r.level.name] += 1
        return counts

    # ------------------------------------------------------------------
    # Level computation
    # ------------------------------------------------------------------

    def _compute_level(self, ctx: DecisionContext) -> EscalationLevel:
        # Safety-critical decisions always go to L5
        if ctx.decision_type in _SAFETY_TYPES and ctx.safety_relevant:
            return EscalationLevel.SAFETY_CRITICAL

        # Very low confidence + patient impact => L5
        if ctx.confidence_score < 0.30 and ctx.patient_impact:
            return EscalationLevel.SAFETY_CRITICAL

        # Regulatory decisions => at least L4
        if ctx.decision_type in _REGULATORY_TYPES or ctx.regulatory_relevant:
            if ctx.confidence_score < _CONF_MEDIUM:
                return EscalationLevel.SAFETY_CRITICAL
            return EscalationLevel.REGULATORY

        # Patient-impacting decisions => at least L3
        if ctx.patient_impact:
            if ctx.confidence_score < _CONF_LOW:
                return EscalationLevel.REGULATORY
            if ctx.confidence_score < _CONF_HIGH:
                return EscalationLevel.CLINICAL
            return EscalationLevel.CONFIRMATION

        # Safety-relevant but not in the critical type set
        if ctx.safety_relevant:
            if ctx.confidence_score < _CONF_MEDIUM:
                return EscalationLevel.CLINICAL
            return EscalationLevel.CONFIRMATION

        # Routine decisions
        if ctx.confidence_score >= _CONF_HIGH:
            return EscalationLevel.INFORMATIONAL
        if ctx.confidence_score >= _CONF_MEDIUM:
            return EscalationLevel.CONFIRMATION
        return EscalationLevel.CLINICAL

    # ------------------------------------------------------------------
    # Rationale builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rationale(ctx: DecisionContext, level: EscalationLevel) -> str:
        parts: list[str] = []
        parts.append(f"type={ctx.decision_type}")
        parts.append(f"confidence={ctx.confidence_score:.2f}")

        flags: list[str] = []
        if ctx.safety_relevant:
            flags.append("safety")
        if ctx.regulatory_relevant:
            flags.append("regulatory")
        if ctx.patient_impact:
            flags.append("patient-impact")
        if flags:
            parts.append("flags=" + "+".join(flags))

        parts.append(f"=> L{level.value} ({level.name})")
        return "; ".join(parts)
