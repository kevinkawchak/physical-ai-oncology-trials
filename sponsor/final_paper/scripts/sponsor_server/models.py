"""Pydantic v2 data models for the sponsor simulation.

Falls back to ``dataclasses`` when Pydantic is not installed so
the module can be imported in lightweight / offline environments.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Try Pydantic v2; fall back to dataclasses
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field

    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False

    # Minimal shims so the rest of the file can use the same class bodies.
    class BaseModel:  # type: ignore[no-redef]
        """Dataclass-backed stand-in when Pydantic is absent."""

        model_config: dict[str, Any] = {}

        def model_dump(self) -> dict[str, Any]:
            """Return fields as a plain dict."""
            return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]

    def Field(default: Any = None, **_kwargs: Any) -> Any:  # type: ignore[no-redef]  # noqa: N802
        """No-op replacement for ``pydantic.Field``."""
        return default


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class DecisionOutcome(str, enum.Enum):
    """Possible sponsor-level decision outcomes."""

    GO = "go"
    NO_GO = "no_go"
    CONDITIONAL = "conditional"
    ESCALATE = "escalate"


class SafetySeverity(str, enum.Enum):
    """CTCAE-aligned severity grades."""

    GRADE_1 = "grade_1"
    GRADE_2 = "grade_2"
    GRADE_3 = "grade_3"
    GRADE_4 = "grade_4"
    GRADE_5 = "grade_5"


class RobotCategory(str, enum.Enum):
    """Ten robot categories from the Physical-AI Oncology framework."""

    SURGICAL_RESECTION = "surgical_resection"
    BIOPSY_SAMPLING = "biopsy_sampling"
    RADIATION_DELIVERY = "radiation_delivery"
    CATHETER_PORT_PLACEMENT = "catheter_port_placement"
    IV_PREPARATION = "iv_preparation"
    PHARMACY_DISPENSING = "pharmacy_dispensing"
    PATIENT_TRANSPORT = "patient_transport"
    REHABILITATION = "rehabilitation"
    DISINFECTION_STERILISATION = "disinfection_sterilisation"
    TELEPRESENCE_MONITORING = "telepresence_monitoring"


class GateStatus(str, enum.Enum):
    """Four-gate safety protocol status values."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    BYPASSED_EMERGENCY = "bypassed_emergency"


class SimulationPhase(str, enum.Enum):
    """Coarse phase within the 24-hour simulation."""

    STARTUP = "startup"
    ENROLLMENT = "enrollment"
    TREATMENT = "treatment"
    MONITORING = "monitoring"
    CLOSE_OUT = "close_out"


# ---------------------------------------------------------------------------
# Helper to optionally wrap classes as dataclasses
# ---------------------------------------------------------------------------
def _maybe_dataclass(cls: type) -> type:
    """Apply ``@dataclass`` only when Pydantic is absent."""
    if not _HAS_PYDANTIC:
        return dataclass(cls)
    return cls


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------
@_maybe_dataclass
class SponsorDecision(BaseModel):
    """A sponsor-level go / no-go / escalate decision."""

    if _HAS_PYDANTIC:
        model_config: dict[str, Any] = {"extra": "forbid"}

    decision_id: str = Field(default="")
    hour_id: int = Field(default=0)
    agent: str = Field(default="portfolio_agent")
    outcome: str = Field(default=DecisionOutcome.GO.value)
    rationale: str = Field(default="")
    confidence: float = Field(default=1.0)
    timestamp: str = Field(default="")
    evidence_refs: list[str] = Field(default_factory=list)


@_maybe_dataclass
class HourlyDirective(BaseModel):
    """Instructions issued to agents for a specific simulation hour."""

    if _HAS_PYDANTIC:
        model_config: dict[str, Any] = {"extra": "forbid"}

    hour_id: int = Field(default=0)
    phase: str = Field(default=SimulationPhase.STARTUP.value)
    directives: list[str] = Field(default_factory=list)
    target_agents: list[str] = Field(default_factory=list)
    priority: int = Field(default=1)
    source_data_path: str = Field(default="")
    timestamp: str = Field(default="")


@_maybe_dataclass
class RobotAuthorization(BaseModel):
    """Authorisation record for a physical robot action."""

    if _HAS_PYDANTIC:
        model_config: dict[str, Any] = {"extra": "forbid"}

    authorization_id: str = Field(default="")
    robot_category: str = Field(default=RobotCategory.SURGICAL_RESECTION.value)
    procedure_id: str = Field(default="")
    patient_id: str = Field(default="")
    site_id: str = Field(default="")
    gate_1_operator: str = Field(default=GateStatus.PENDING.value)
    gate_2_supervisor: str = Field(default=GateStatus.PENDING.value)
    gate_3_sponsor: str = Field(default=GateStatus.PENDING.value)
    gate_4_regulatory: str = Field(default=GateStatus.PENDING.value)
    overall_status: str = Field(default="pending")
    issued_by: str = Field(default="robot_execution_gateway")
    timestamp: str = Field(default="")


@_maybe_dataclass
class SafetyEscalation(BaseModel):
    """Safety signal or adverse-event escalation record."""

    if _HAS_PYDANTIC:
        model_config: dict[str, Any] = {"extra": "forbid"}

    escalation_id: str = Field(default="")
    hour_id: int = Field(default=0)
    ae_term: str = Field(default="")
    severity: str = Field(default=SafetySeverity.GRADE_1.value)
    seriousness: bool = Field(default=False)
    causality: str = Field(default="unrelated")
    action_taken: str = Field(default="")
    e2b_submitted: bool = Field(default=False)
    reporter_agent: str = Field(default="safety_agent")
    timestamp: str = Field(default="")


@_maybe_dataclass
class AgentMetrics(BaseModel):
    """Per-agent performance metrics for a simulation hour."""

    if _HAS_PYDANTIC:
        model_config: dict[str, Any] = {"extra": "forbid"}

    agent_name: str = Field(default="")
    hour_id: int = Field(default=0)
    tasks_received: int = Field(default=0)
    tasks_completed: int = Field(default=0)
    escalations_sent: int = Field(default=0)
    mean_latency_ms: float = Field(default=0.0)
    errors: int = Field(default=0)


@_maybe_dataclass
class SimulationSummary(BaseModel):
    """End-of-simulation summary across all 24 hours."""

    if _HAS_PYDANTIC:
        model_config: dict[str, Any] = {"extra": "forbid"}

    trial_id: str = Field(default="")
    total_hours: int = Field(default=24)
    decisions_made: int = Field(default=0)
    safety_escalations: int = Field(default=0)
    robot_authorizations: int = Field(default=0)
    patients_enrolled: int = Field(default=0)
    sites_active: int = Field(default=0)
    agent_metrics: list[dict[str, Any]] = Field(default_factory=list)
    phase_transitions: list[dict[str, Any]] = Field(default_factory=list)
    final_status: str = Field(default="completed")
    created_at: str = Field(default="")


@_maybe_dataclass
class HourStatus(BaseModel):
    """Status snapshot for a single simulation hour."""

    if _HAS_PYDANTIC:
        model_config: dict[str, Any] = {"extra": "forbid"}

    hour_id: int = Field(default=0)
    phase: str = Field(default=SimulationPhase.STARTUP.value)
    active_agents: list[str] = Field(default_factory=list)
    pending_tasks: int = Field(default=0)
    completed_tasks: int = Field(default=0)
    open_escalations: int = Field(default=0)
    robot_actions_pending: int = Field(default=0)
    timestamp: str = Field(default="")


@_maybe_dataclass
class TrialEvent(BaseModel):
    """Generic event emitted during the simulation."""

    if _HAS_PYDANTIC:
        model_config: dict[str, Any] = {"extra": "forbid"}

    event_id: str = Field(default="")
    hour_id: int = Field(default=0)
    event_type: str = Field(default="info")
    source_agent: str = Field(default="")
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default="")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.utcnow().isoformat() + "Z"


def source_data_path(hour: int) -> str:
    """Return the conventional data directory for a given hour."""
    return f"new-trial/hour-{hour:02d}"


# Keep ``field`` reachable for submodules that import from here.
_field_ref = field  # noqa: F841 – re-export guard

__all__ = [
    "DecisionOutcome",
    "SafetySeverity",
    "RobotCategory",
    "GateStatus",
    "SimulationPhase",
    "SponsorDecision",
    "HourlyDirective",
    "RobotAuthorization",
    "SafetyEscalation",
    "AgentMetrics",
    "SimulationSummary",
    "HourStatus",
    "TrialEvent",
    "source_data_path",
]
