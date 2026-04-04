"""Operations router -- clinical operations and site management endpoints.

Exposes the clinops agent's functionality: site activation, enrollment
tracking, monitoring scheduling, and hourly directive generation.
"""
from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, HTTPException

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False

from sponsor_server.agents import clinops_agent, study_orchestrator
from sponsor_server.models import HourlyDirective, HourStatus

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if _HAS_FASTAPI:
    router = APIRouter(prefix="/api/v1/sponsor", tags=["operations"])

    @router.post("/hour/{hour_id}/direct", response_model=None)
    async def direct_hour(hour_id: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Ingest hour data and issue a master directive."""
        if hour_id < 0 or hour_id > 23:
            raise HTTPException(status_code=422, detail="hour_id must be 0-23")
        directive = study_orchestrator.ingest_hour(hour_id, payload)
        return _directive_to_dict(directive)

    @router.get("/hour/{hour_id}/status", response_model=None)
    async def get_hour_status(hour_id: int) -> dict[str, Any]:
        """Retrieve status snapshot for a simulation hour."""
        status = study_orchestrator.get_hour_status(hour_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"No data for hour {hour_id}")
        return _status_to_dict(status)

    @router.post("/sites/activate")
    async def activate_site(site_id: str, hour_id: int = 0, investigator: str = "PI-TBD") -> dict[str, Any]:
        """Activate a new trial site."""
        return clinops_agent.activate_site(site_id, hour_id, investigator)

    @router.post("/sites/{site_id}/enroll")
    async def enroll_patient(site_id: str, patient_id: str, hour_id: int = 0) -> dict[str, Any]:
        """Record a patient enrollment at a site."""
        return clinops_agent.record_enrollment(site_id, patient_id, hour_id)

    @router.get("/enrollment/status")
    async def enrollment_status(hour_id: int = 0) -> dict[str, Any]:
        """Get enrollment progress summary."""
        return clinops_agent.get_enrollment_status(hour_id)

    @router.post("/sites/{site_id}/monitoring")
    async def schedule_monitoring(site_id: str, hour_id: int = 0) -> dict[str, Any]:
        """Schedule a monitoring visit for a site."""
        return clinops_agent.schedule_monitoring(site_id, hour_id)

else:
    router = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------
def direct_hour_standalone(hour_id: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Issue an hourly directive without the server running."""
    directive = study_orchestrator.ingest_hour(hour_id, payload)
    return _directive_to_dict(directive)


def get_hour_status_standalone(hour_id: int) -> dict[str, Any] | None:
    """Retrieve hour status without the server running."""
    status = study_orchestrator.get_hour_status(hour_id)
    if status is None:
        return None
    return _status_to_dict(status)


def _directive_to_dict(directive: HourlyDirective) -> dict[str, Any]:
    """Convert an HourlyDirective to a plain dict."""
    if hasattr(directive, "model_dump"):
        return directive.model_dump()  # type: ignore[union-attr]
    return {
        "hour_id": directive.hour_id,
        "phase": directive.phase,
        "directives": directive.directives,
        "target_agents": directive.target_agents,
        "priority": directive.priority,
        "source_data_path": directive.source_data_path,
        "timestamp": directive.timestamp,
    }


def _status_to_dict(status: HourStatus) -> dict[str, Any]:
    """Convert an HourStatus to a plain dict."""
    if hasattr(status, "model_dump"):
        return status.model_dump()  # type: ignore[union-attr]
    return {
        "hour_id": status.hour_id,
        "phase": status.phase,
        "active_agents": status.active_agents,
        "pending_tasks": status.pending_tasks,
        "completed_tasks": status.completed_tasks,
        "open_escalations": status.open_escalations,
        "robot_actions_pending": status.robot_actions_pending,
        "timestamp": status.timestamp,
    }
