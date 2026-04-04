"""Clinical operations agent -- site startup, enrollment tracking, and monitoring.

Manages the operational lifecycle of trial sites within the 24-hour
simulation: activating sites, tracking patient enrollment against
targets, scheduling monitoring visits, and flagging deviations.
"""

from __future__ import annotations

import logging
from typing import Any

from sponsor_server.models import (
    HourlyDirective,
    SimulationPhase,
    _now_iso,
    source_data_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_ENROLLMENT_PER_HOUR: int = 2
MONITORING_INTERVAL_HOURS: int = 4

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
_sites: dict[str, dict[str, Any]] = {}
_enrollment_log: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def activate_site(site_id: str, hour_id: int, investigator: str = "PI-TBD") -> dict[str, Any]:
    """Register and activate a trial site.

    Parameters
    ----------
    site_id:
        Unique site identifier.
    hour_id:
        Hour at which the site goes live.
    investigator:
        Name or code for the principal investigator.

    Returns
    -------
    dict with site activation details.
    """
    record: dict[str, Any] = {
        "site_id": site_id,
        "activated_hour": hour_id,
        "investigator": investigator,
        "status": "active",
        "patients_enrolled": 0,
        "monitoring_visits": [],
        "activated_at": _now_iso(),
    }
    _sites[site_id] = record
    logger.info("Site activated: %s at hour %d", site_id, hour_id)
    return record


def record_enrollment(site_id: str, patient_id: str, hour_id: int) -> dict[str, Any]:
    """Record a patient enrollment event at a site.

    Returns
    -------
    dict with enrollment confirmation and running totals.
    """
    site = _sites.get(site_id, {})
    if site:
        site["patients_enrolled"] = site.get("patients_enrolled", 0) + 1

    entry: dict[str, Any] = {
        "site_id": site_id,
        "patient_id": patient_id,
        "hour_id": hour_id,
        "timestamp": _now_iso(),
    }
    _enrollment_log.append(entry)
    logger.info("Enrollment: patient=%s site=%s hour=%d", patient_id, site_id, hour_id)
    return {
        "enrolled": True,
        "site_total": site.get("patients_enrolled", 1),
        "global_total": len(_enrollment_log),
        "source": source_data_path(hour_id),
    }


def get_enrollment_status(hour_id: int) -> dict[str, Any]:
    """Summarise enrollment progress through *hour_id*.

    Compares actual enrollment to the linear target and flags
    whether the study is on-track, ahead, or behind.
    """
    enrolled_through = [e for e in _enrollment_log if e["hour_id"] <= hour_id]
    target = (hour_id + 1) * TARGET_ENROLLMENT_PER_HOUR
    actual = len(enrolled_through)
    pace = "on_track" if actual >= target else ("behind" if actual < target * 0.8 else "slightly_behind")
    return {
        "through_hour": hour_id,
        "target": target,
        "actual": actual,
        "pace": pace,
        "sites_active": sum(1 for s in _sites.values() if s.get("status") == "active"),
    }


def schedule_monitoring(site_id: str, hour_id: int) -> dict[str, Any]:
    """Schedule or record a monitoring visit for a site.

    Monitoring visits happen every ``MONITORING_INTERVAL_HOURS``.
    """
    visit: dict[str, Any] = {
        "site_id": site_id,
        "scheduled_hour": hour_id,
        "type": "routine" if hour_id % MONITORING_INTERVAL_HOURS == 0 else "triggered",
        "status": "scheduled",
        "timestamp": _now_iso(),
    }
    site = _sites.get(site_id)
    if site is not None:
        site.setdefault("monitoring_visits", []).append(visit)
    logger.info("Monitoring visit scheduled: site=%s hour=%d type=%s", site_id, hour_id, visit["type"])
    return visit


def build_hourly_directive(hour_id: int) -> HourlyDirective:
    """Generate the ClinOps portion of the hourly directive.

    Determines the simulation phase and builds directives targeted at
    operations-related agents.
    """
    phase = _resolve_phase(hour_id)
    directives = _phase_directives(phase, hour_id)
    return HourlyDirective(
        hour_id=hour_id,
        phase=phase.value,
        directives=directives,
        target_agents=["clinops_agent", "site_gateway", "supply_agent"],
        priority=2 if phase == SimulationPhase.ENROLLMENT else 1,
        source_data_path=source_data_path(hour_id),
        timestamp=_now_iso(),
    )


def get_sites() -> dict[str, dict[str, Any]]:
    """Return the current sites registry (read-only copy)."""
    return dict(_sites)


def reset_state() -> None:
    """Clear in-memory state (useful for testing)."""
    _sites.clear()
    _enrollment_log.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_phase(hour_id: int) -> SimulationPhase:
    """Map a simulation hour to the coarse trial phase."""
    if hour_id <= 2:
        return SimulationPhase.STARTUP
    if hour_id <= 8:
        return SimulationPhase.ENROLLMENT
    if hour_id <= 16:
        return SimulationPhase.TREATMENT
    if hour_id <= 21:
        return SimulationPhase.MONITORING
    return SimulationPhase.CLOSE_OUT


def _phase_directives(phase: SimulationPhase, hour_id: int) -> list[str]:
    """Return a list of directive strings appropriate for the phase."""
    base: list[str] = [f"Ingest data from {source_data_path(hour_id)}"]
    phase_map: dict[SimulationPhase, list[str]] = {
        SimulationPhase.STARTUP: ["Verify site readiness", "Confirm drug supply at depot"],
        SimulationPhase.ENROLLMENT: ["Screen and enroll patients", "Update IWRS allocation"],
        SimulationPhase.TREATMENT: ["Administer study drug", "Collect PK/PD samples"],
        SimulationPhase.MONITORING: ["Conduct monitoring visit", "Reconcile source data"],
        SimulationPhase.CLOSE_OUT: ["Lock database queries", "Archive site files"],
    }
    base.extend(phase_map.get(phase, []))
    return base
