from __future__ import annotations

"""Coordination protocols for autonomous sponsor simulation."""

from .agent_event_bus import AgentEventBus, Event, EventType, Severity
from .escalation_engine import EscalationEngine, EscalationLevel, EscalationResult
from .gate_transition_manager import GateOutcome, GateTransitionManager

__all__ = [
    "AgentEventBus",
    "EscalationEngine",
    "EscalationLevel",
    "EscalationResult",
    "Event",
    "EventType",
    "GateOutcome",
    "GateTransitionManager",
    "Severity",
]
