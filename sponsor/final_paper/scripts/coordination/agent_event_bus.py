from __future__ import annotations

"""Publish-subscribe event bus for inter-agent communication.

Provides typed, severity-graded event routing between autonomous sponsor
agents.  Each event carries a timestamp, source agent identifier, event
type, severity (1-5), and an arbitrary payload dictionary.  Agents
subscribe to one or more event types and receive only matching events.
"""

import logging
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Canonical event categories routed by the bus."""

    GATE_TRANSITION = "GATE_TRANSITION"
    SAFETY_ALERT = "SAFETY_ALERT"
    ENROLLMENT_UPDATE = "ENROLLMENT_UPDATE"
    ROBOT_STATUS = "ROBOT_STATUS"
    DATA_QUALITY = "DATA_QUALITY"
    REGULATORY_DEADLINE = "REGULATORY_DEADLINE"
    VENDOR_PERFORMANCE = "VENDOR_PERFORMANCE"


class Severity(int, Enum):
    """Five-level severity scale (1 = lowest, 5 = highest)."""

    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5


@dataclass(frozen=True)
class Event:
    """Immutable event envelope transmitted over the bus."""

    event_id: str
    timestamp: datetime
    source_agent: str
    event_type: EventType
    severity: Severity
    payload: dict[str, object]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        source_agent: str,
        event_type: EventType,
        severity: Severity,
        payload: dict[str, object] | None = None,
    ) -> Event:
        """Build an event with auto-generated id and UTC timestamp."""
        return cls(
            event_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc),
            source_agent=source_agent,
            event_type=event_type,
            severity=severity,
            payload=payload or {},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "source_agent": self.source_agent,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "payload": self.payload,
        }


# Type alias for subscriber callbacks
EventHandler = Callable[[Event], None]


# ---------------------------------------------------------------------------
# Subscription record
# ---------------------------------------------------------------------------


@dataclass
class _Subscription:
    subscriber_id: str
    agent_name: str
    event_types: set[EventType]
    min_severity: Severity
    handler: EventHandler


# ---------------------------------------------------------------------------
# Bus implementation
# ---------------------------------------------------------------------------


@dataclass
class AgentEventBus:
    """Thread-safe publish-subscribe event bus.

    Usage::

        bus = AgentEventBus()
        bus.subscribe("enrollment_agent", {EventType.ENROLLMENT_UPDATE}, handler)
        bus.publish(Event.create("gate_agent", EventType.ENROLLMENT_UPDATE,
                                 Severity.INFO, {"enrolled": 42}))
    """

    _subscriptions: dict[EventType, list[_Subscription]] = field(
        default_factory=lambda: defaultdict(list),
    )
    _event_log: list[Event] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _max_log_size: int = 10_000

    # ------------------------------------------------------------------
    # Subscribe / unsubscribe
    # ------------------------------------------------------------------

    def subscribe(
        self,
        agent_name: str,
        event_types: set[EventType],
        handler: EventHandler,
        min_severity: Severity = Severity.DEBUG,
    ) -> str:
        """Register *handler* to receive events of the given types.

        Returns a subscription id that can be used to unsubscribe.
        """
        sub_id = uuid.uuid4().hex[:10]
        sub = _Subscription(
            subscriber_id=sub_id,
            agent_name=agent_name,
            event_types=event_types,
            min_severity=min_severity,
            handler=handler,
        )
        with self._lock:
            for et in event_types:
                self._subscriptions[et].append(sub)
        logger.info(
            "Agent '%s' subscribed to %s (min_severity=%s) [%s]",
            agent_name,
            sorted(e.value for e in event_types),
            min_severity.name,
            sub_id,
        )
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription by id.  Returns True if found."""
        removed = False
        with self._lock:
            for et in list(self._subscriptions):
                before = len(self._subscriptions[et])
                self._subscriptions[et] = [
                    s for s in self._subscriptions[et] if s.subscriber_id != subscription_id
                ]
                if len(self._subscriptions[et]) < before:
                    removed = True
        if removed:
            logger.info("Subscription %s removed", subscription_id)
        return removed

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> int:
        """Dispatch *event* to all matching subscribers.

        Returns the number of handlers that were invoked.
        """
        self._record(event)

        with self._lock:
            targets = list(self._subscriptions.get(event.event_type, []))

        dispatched = 0
        for sub in targets:
            if event.severity.value < sub.min_severity.value:
                continue
            try:
                sub.handler(event)
                dispatched += 1
            except Exception:
                logger.exception(
                    "Handler for '%s' raised on event %s",
                    sub.agent_name,
                    event.event_id,
                )
        logger.debug(
            "Event %s (%s, severity=%d) dispatched to %d handler(s)",
            event.event_id,
            event.event_type.value,
            event.severity.value,
            dispatched,
        )
        return dispatched

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def recent_events(
        self,
        event_type: EventType | None = None,
        min_severity: Severity = Severity.DEBUG,
        limit: int = 50,
    ) -> list[Event]:
        """Return the most recent events matching the optional filters."""
        with self._lock:
            source = list(reversed(self._event_log))
        results: list[Event] = []
        for ev in source:
            if event_type is not None and ev.event_type != event_type:
                continue
            if ev.severity.value < min_severity.value:
                continue
            results.append(ev)
            if len(results) >= limit:
                break
        return results

    def subscriber_count(self, event_type: EventType) -> int:
        with self._lock:
            return len(self._subscriptions.get(event_type, []))

    def total_events_published(self) -> int:
        with self._lock:
            return len(self._event_log)

    def severity_counts(self) -> dict[str, int]:
        """Return event counts grouped by severity name."""
        counts: dict[str, int] = {s.name: 0 for s in Severity}
        with self._lock:
            for ev in self._event_log:
                counts[ev.severity.name] += 1
        return counts

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record(self, event: Event) -> None:
        with self._lock:
            self._event_log.append(event)
            if len(self._event_log) > self._max_log_size:
                self._event_log = self._event_log[-self._max_log_size:]
