"""Regulatory agent -- IND lifecycle management, filings, and compliance calendar.

Tracks regulatory milestones across the 24-hour simulation, manages the
IND application lifecycle, generates filing stubs, and maintains a
compliance calendar so deadlines are never missed.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sponsor_server.models import (
    _now_iso,
    source_data_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IND lifecycle states
# ---------------------------------------------------------------------------
IND_STATES: list[str] = [
    "pre_ind",
    "ind_submitted",
    "ind_active",
    "ind_amendment",
    "ind_safety_report",
    "ind_annual_report",
    "ind_withdrawn",
]

# ---------------------------------------------------------------------------
# Compliance calendar entries (hour-based in the simulation)
# ---------------------------------------------------------------------------
_DEFAULT_MILESTONES: list[dict[str, Any]] = [
    {"milestone": "pre_ind_meeting", "due_hour": 1, "status": "pending"},
    {"milestone": "ind_submission", "due_hour": 3, "status": "pending"},
    {"milestone": "30_day_safety_review", "due_hour": 6, "status": "pending"},
    {"milestone": "protocol_amendment_check", "due_hour": 12, "status": "pending"},
    {"milestone": "annual_report_stub", "due_hour": 20, "status": "pending"},
    {"milestone": "study_closeout_filing", "due_hour": 23, "status": "pending"},
]

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
_ind_state: dict[str, Any] = {"status": "pre_ind", "history": []}
_compliance_calendar: list[dict[str, Any]] = []
_filings: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def initialise_compliance_calendar() -> list[dict[str, Any]]:
    """Populate the compliance calendar with default milestones.

    Calling this again resets the calendar to its initial state.
    """
    _compliance_calendar.clear()
    for m in _DEFAULT_MILESTONES:
        _compliance_calendar.append({**m, "id": f"CM-{uuid.uuid4().hex[:6]}"})
    logger.info("Compliance calendar initialised with %d milestones", len(_compliance_calendar))
    return list(_compliance_calendar)


def advance_ind_state(new_state: str, hour_id: int, notes: str = "") -> dict[str, Any]:
    """Transition the IND to a new lifecycle state.

    Parameters
    ----------
    new_state:
        Target state (must be in ``IND_STATES``).
    hour_id:
        Simulation hour of the transition.
    notes:
        Free-text annotation.

    Returns
    -------
    dict with previous state, new state, and timestamp.
    """
    previous = _ind_state["status"]
    if new_state not in IND_STATES:
        logger.error("Invalid IND state: %s", new_state)
        return {"error": f"Invalid state: {new_state}", "valid_states": IND_STATES}

    _ind_state["status"] = new_state
    transition: dict[str, Any] = {
        "from": previous,
        "to": new_state,
        "hour_id": hour_id,
        "notes": notes,
        "timestamp": _now_iso(),
    }
    _ind_state["history"].append(transition)
    logger.info("IND transition: %s -> %s at hour %d", previous, new_state, hour_id)
    return transition


def check_compliance(hour_id: int) -> list[dict[str, Any]]:
    """Return milestones that are due or overdue at *hour_id*.

    Each returned milestone dict includes a ``status`` field that is
    updated to ``overdue`` if the due hour has passed without completion.
    """
    due: list[dict[str, Any]] = []
    for m in _compliance_calendar:
        if m["due_hour"] <= hour_id and m["status"] == "pending":
            m["status"] = "overdue" if m["due_hour"] < hour_id else "due_now"
            due.append(m)
    if due:
        logger.warning("%d compliance milestones due/overdue at hour %d", len(due), hour_id)
    return due


def create_filing(
    filing_type: str,
    hour_id: int,
    content_summary: str = "",
) -> dict[str, Any]:
    """Generate a regulatory filing stub.

    Parameters
    ----------
    filing_type:
        E.g. ``ind_submission``, ``safety_report``, ``protocol_amendment``.
    hour_id:
        Simulation hour of filing generation.
    content_summary:
        Brief description of filing contents.

    Returns
    -------
    dict representing the filing record.
    """
    filing: dict[str, Any] = {
        "filing_id": f"FIL-{uuid.uuid4().hex[:8]}",
        "type": filing_type,
        "hour_id": hour_id,
        "content_summary": content_summary,
        "source": source_data_path(hour_id),
        "status": "draft",
        "created_at": _now_iso(),
    }
    _filings.append(filing)
    logger.info("Filing created: type=%s id=%s", filing_type, filing["filing_id"])
    return filing


def get_ind_status() -> dict[str, Any]:
    """Return the current IND state and transition history."""
    return dict(_ind_state)


def get_filings() -> list[dict[str, Any]]:
    """Return all filing records."""
    return list(_filings)


def complete_milestone(milestone_id: str) -> dict[str, Any] | None:
    """Mark a compliance milestone as completed.

    Returns the updated milestone or ``None`` if not found.
    """
    for m in _compliance_calendar:
        if m.get("id") == milestone_id:
            m["status"] = "completed"
            m["completed_at"] = _now_iso()
            logger.info("Milestone completed: %s", milestone_id)
            return m
    return None


def reset_state() -> None:
    """Clear all in-memory regulatory state."""
    _ind_state["status"] = "pre_ind"
    _ind_state["history"] = []
    _compliance_calendar.clear()
    _filings.clear()
