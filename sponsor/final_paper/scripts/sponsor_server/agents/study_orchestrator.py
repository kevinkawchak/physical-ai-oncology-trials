"""Study orchestrator -- master task routing, escalation, and coordination.

The study orchestrator is the central hub that receives hourly data from
``new-trial/hour-00`` through ``hour-23``, distributes tasks to the
twelve specialised agents, aggregates results, and manages escalation
paths when agents disagree or thresholds are breached.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sponsor_server.agents import AGENT_NAMES
from sponsor_server.models import (
    AgentMetrics,
    HourlyDirective,
    HourStatus,
    SimulationPhase,
    SimulationSummary,
    TrialEvent,
    _now_iso,
    source_data_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Escalation thresholds
# ---------------------------------------------------------------------------
ESCALATION_LATENCY_MS: float = 5000.0
ESCALATION_ERROR_RATE: float = 0.10

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
_hour_statuses: dict[int, HourStatus] = {}
_events: list[TrialEvent] = []
_metrics: dict[str, list[AgentMetrics]] = {name: [] for name in AGENT_NAMES}
_escalations: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ingest_hour(hour_id: int, payload: dict[str, Any] | None = None) -> HourlyDirective:
    """Ingest data for a simulation hour and produce a master directive.

    Parameters
    ----------
    hour_id:
        Simulation hour (0-23).
    payload:
        Optional pre-parsed hour data (patient records, robot logs, etc.).

    Returns
    -------
    HourlyDirective distributed to all agents.
    """
    phase = _resolve_phase(hour_id)
    target_agents = _select_agents(phase)
    directives = _build_directives(phase, hour_id, payload or {})

    directive = HourlyDirective(
        hour_id=hour_id,
        phase=phase.value,
        directives=directives,
        target_agents=target_agents,
        priority=_compute_priority(phase, payload or {}),
        source_data_path=source_data_path(hour_id),
        timestamp=_now_iso(),
    )

    # Initialise hour status
    status = HourStatus(
        hour_id=hour_id,
        phase=phase.value,
        active_agents=target_agents,
        pending_tasks=len(directives),
        completed_tasks=0,
        open_escalations=0,
        robot_actions_pending=0,
        timestamp=_now_iso(),
    )
    _hour_statuses[hour_id] = status

    _emit_event(hour_id, "hour_ingested", "study_orchestrator", {
        "phase": phase.value,
        "agents_activated": len(target_agents),
        "directives_count": len(directives),
    })

    logger.info(
        "Hour %d ingested: phase=%s agents=%d directives=%d",
        hour_id,
        phase.value,
        len(target_agents),
        len(directives),
    )
    return directive


def route_task(hour_id: int, agent_name: str, task: dict[str, Any]) -> dict[str, Any]:
    """Route a single task to a named agent.

    Returns
    -------
    dict with routing confirmation and tracking ID.
    """
    task_id = f"T-{uuid.uuid4().hex[:8]}"
    result: dict[str, Any] = {
        "task_id": task_id,
        "hour_id": hour_id,
        "agent": agent_name,
        "task": task,
        "status": "routed",
        "timestamp": _now_iso(),
    }
    _emit_event(hour_id, "task_routed", "study_orchestrator", {"task_id": task_id, "agent": agent_name})
    logger.info("Task %s routed to %s", task_id, agent_name)
    return result


def escalate(hour_id: int, source_agent: str, reason: str, severity: str = "high") -> dict[str, Any]:
    """Record an escalation from an agent.

    Escalations are forwarded to the portfolio agent and clinical
    accountability agent for review.
    """
    esc: dict[str, Any] = {
        "escalation_id": f"ESC-{uuid.uuid4().hex[:8]}",
        "hour_id": hour_id,
        "source_agent": source_agent,
        "reason": reason,
        "severity": severity,
        "reviewers": ["portfolio_agent", "clinical_accountability_agent"],
        "status": "open",
        "created_at": _now_iso(),
    }
    _escalations.append(esc)

    # Update hour status
    hour_status = _hour_statuses.get(hour_id)
    if hour_status is not None:
        object.__setattr__(hour_status, "open_escalations", hour_status.open_escalations + 1)

    _emit_event(hour_id, "escalation_raised", source_agent, {"escalation_id": esc["escalation_id"]})
    logger.warning("Escalation from %s at hour %d: %s", source_agent, hour_id, reason)
    return esc


def record_agent_metrics(
    agent_name: str,
    hour_id: int,
    tasks_received: int = 0,
    tasks_completed: int = 0,
    escalations_sent: int = 0,
    mean_latency_ms: float = 0.0,
    errors: int = 0,
) -> AgentMetrics:
    """Record performance metrics for an agent at a given hour."""
    metrics = AgentMetrics(
        agent_name=agent_name,
        hour_id=hour_id,
        tasks_received=tasks_received,
        tasks_completed=tasks_completed,
        escalations_sent=escalations_sent,
        mean_latency_ms=mean_latency_ms,
        errors=errors,
    )
    _metrics.setdefault(agent_name, []).append(metrics)

    # Check for auto-escalation triggers
    if mean_latency_ms > ESCALATION_LATENCY_MS:
        escalate(hour_id, agent_name, f"Latency {mean_latency_ms:.0f}ms exceeds threshold")
    if tasks_received > 0 and (errors / tasks_received) > ESCALATION_ERROR_RATE:
        escalate(hour_id, agent_name, f"Error rate {errors}/{tasks_received} exceeds threshold")

    return metrics


def get_hour_status(hour_id: int) -> HourStatus | None:
    """Return the status snapshot for a simulation hour."""
    return _hour_statuses.get(hour_id)


def get_escalations(hour_id: int | None = None) -> list[dict[str, Any]]:
    """Return escalations, optionally filtered by hour."""
    if hour_id is not None:
        return [e for e in _escalations if e["hour_id"] == hour_id]
    return list(_escalations)


def build_simulation_summary(trial_id: str = "SIM-001") -> SimulationSummary:
    """Compile an end-of-simulation summary across all 24 hours.

    Aggregates decisions, safety events, robot authorisations, and
    per-agent metrics into a single ``SimulationSummary``.
    """
    all_metrics: list[dict[str, Any]] = []
    for agent_name, metrics_list in _metrics.items():
        if metrics_list:
            total_tasks = sum(m.tasks_completed for m in metrics_list)
            total_errors = sum(m.errors for m in metrics_list)
            all_metrics.append({
                "agent": agent_name,
                "total_tasks_completed": total_tasks,
                "total_errors": total_errors,
                "hours_active": len(metrics_list),
            })

    phase_transitions: list[dict[str, Any]] = []
    prev_phase = ""
    for h in sorted(_hour_statuses.keys()):
        status = _hour_statuses[h]
        if status.phase != prev_phase:
            phase_transitions.append({"hour": h, "phase": status.phase})
            prev_phase = status.phase

    return SimulationSummary(
        trial_id=trial_id,
        total_hours=len(_hour_statuses),
        decisions_made=len(_events),
        safety_escalations=sum(1 for e in _escalations if e.get("source_agent") == "safety_agent"),
        robot_authorizations=0,
        patients_enrolled=0,
        sites_active=0,
        agent_metrics=all_metrics,
        phase_transitions=phase_transitions,
        final_status="completed" if len(_hour_statuses) >= 24 else "in_progress",
        created_at=_now_iso(),
    )


def get_events(hour_id: int | None = None) -> list[TrialEvent]:
    """Return simulation events, optionally filtered by hour."""
    if hour_id is not None:
        return [e for e in _events if e.hour_id == hour_id]
    return list(_events)


def reset_state() -> None:
    """Clear all orchestrator state."""
    _hour_statuses.clear()
    _events.clear()
    _escalations.clear()
    for key in _metrics:
        _metrics[key] = []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_phase(hour_id: int) -> SimulationPhase:
    """Map simulation hour to trial phase."""
    if hour_id <= 2:
        return SimulationPhase.STARTUP
    if hour_id <= 8:
        return SimulationPhase.ENROLLMENT
    if hour_id <= 16:
        return SimulationPhase.TREATMENT
    if hour_id <= 21:
        return SimulationPhase.MONITORING
    return SimulationPhase.CLOSE_OUT


def _select_agents(phase: SimulationPhase) -> list[str]:
    """Choose which agents are active in a given phase."""
    always_active = ["study_orchestrator", "safety_agent", "data_biostats_agent"]
    phase_agents: dict[SimulationPhase, list[str]] = {
        SimulationPhase.STARTUP: ["clinops_agent", "regulatory_agent", "supply_agent", "site_gateway"],
        SimulationPhase.ENROLLMENT: ["clinops_agent", "site_gateway", "supply_agent", "quality_agent"],
        SimulationPhase.TREATMENT: [
            "clinops_agent", "robot_execution_gateway", "supply_agent",
            "clinical_accountability_agent",
        ],
        SimulationPhase.MONITORING: ["clinops_agent", "quality_agent", "regulatory_agent"],
        SimulationPhase.CLOSE_OUT: [
            "regulatory_agent", "quality_agent", "portfolio_agent", "asset_lead_agent",
        ],
    }
    selected = list(always_active)
    for agent in phase_agents.get(phase, []):
        if agent not in selected:
            selected.append(agent)
    return selected


def _build_directives(phase: SimulationPhase, hour_id: int, payload: dict[str, Any]) -> list[str]:
    """Produce directive strings for the master hourly directive."""
    base = [f"Load source data from {source_data_path(hour_id)}"]

    if phase == SimulationPhase.STARTUP:
        base.extend(["Verify protocol activation", "Confirm IRB approval status"])
    elif phase == SimulationPhase.ENROLLMENT:
        base.extend(["Process screening queue", "Update randomisation system"])
    elif phase == SimulationPhase.TREATMENT:
        base.extend(["Execute treatment schedule", "Monitor robot procedures"])
    elif phase == SimulationPhase.MONITORING:
        base.extend(["Run data quality checks", "Generate monitoring reports"])
    elif phase == SimulationPhase.CLOSE_OUT:
        base.extend(["Finalise database lock", "Prepare regulatory submission package"])

    if payload.get("safety_alert"):
        base.append("PRIORITY: Process safety alert")

    return base


def _compute_priority(phase: SimulationPhase, payload: dict[str, Any]) -> int:
    """Assign a priority score (1=low, 5=critical)."""
    base_priority: dict[SimulationPhase, int] = {
        SimulationPhase.STARTUP: 2,
        SimulationPhase.ENROLLMENT: 3,
        SimulationPhase.TREATMENT: 3,
        SimulationPhase.MONITORING: 2,
        SimulationPhase.CLOSE_OUT: 2,
    }
    priority = base_priority.get(phase, 2)
    if payload.get("safety_alert"):
        priority = 5
    return priority


def _emit_event(hour_id: int, event_type: str, source_agent: str, payload: dict[str, Any]) -> None:
    """Create and store a TrialEvent."""
    event = TrialEvent(
        event_id=f"EV-{uuid.uuid4().hex[:8]}",
        hour_id=hour_id,
        event_type=event_type,
        source_agent=source_agent,
        payload=payload,
        timestamp=_now_iso(),
    )
    _events.append(event)
