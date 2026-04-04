"""Robotics router -- robot authorisation and 4-gate safety protocol endpoints.

Exposes the robot execution gateway's authorisation workflow: requesting
authorisation, advancing individual gates, querying status, and managing
global regulatory holds.
"""
from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, HTTPException

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False

from sponsor_server.agents import robot_gateway
from sponsor_server.models import RobotAuthorization

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if _HAS_FASTAPI:
    router = APIRouter(prefix="/api/v1/sponsor/robot", tags=["robotics"])

    @router.post("/authorize", response_model=None)
    async def authorize_robot(
        robot_category: str = "telepresence_monitoring",
        procedure_id: str = "",
        patient_id: str = "",
        site_id: str = "",
    ) -> dict[str, Any]:
        """Request a new robot authorisation (all gates start pending)."""
        auth = robot_gateway.request_authorization(
            robot_category=robot_category,
            procedure_id=procedure_id,
            patient_id=patient_id,
            site_id=site_id,
        )
        return _auth_to_dict(auth)

    @router.post("/authorize/{authorization_id}/gate")
    async def advance_gate(
        authorization_id: str,
        gate_name: str = "gate_1_operator",
        status: str = "passed",
    ) -> dict[str, Any]:
        """Advance a specific gate on an authorisation."""
        auth = robot_gateway.advance_gate(authorization_id, gate_name, status)
        if auth is None:
            raise HTTPException(status_code=404, detail=f"Authorization {authorization_id} not found")
        return _auth_to_dict(auth)

    @router.get("/authorize/{authorization_id}")
    async def get_authorization(authorization_id: str) -> dict[str, Any]:
        """Retrieve a single authorisation by ID."""
        auth = robot_gateway.get_authorization(authorization_id)
        if auth is None:
            raise HTTPException(status_code=404, detail=f"Authorization {authorization_id} not found")
        return _auth_to_dict(auth)

    @router.get("/authorizations")
    async def list_authorizations(status_filter: str | None = None) -> list[dict[str, Any]]:
        """List all authorisations, optionally filtered by status."""
        auths = robot_gateway.list_authorizations(status_filter)
        return [_auth_to_dict(a) for a in auths]

    @router.post("/hold")
    async def set_hold(active: bool = True) -> dict[str, Any]:
        """Activate or deactivate a global regulatory hold."""
        return robot_gateway.set_regulatory_hold(active)

    @router.get("/categories")
    async def list_categories() -> list[str]:
        """Return the ten robot category identifiers."""
        return robot_gateway.get_category_list()

else:
    router = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------
def authorize_standalone(
    robot_category: str = "telepresence_monitoring",
    procedure_id: str = "",
    patient_id: str = "",
    site_id: str = "",
) -> dict[str, Any]:
    """Request a robot authorisation without the server running."""
    auth = robot_gateway.request_authorization(robot_category, procedure_id, patient_id, site_id)
    return _auth_to_dict(auth)


def _auth_to_dict(auth: RobotAuthorization) -> dict[str, Any]:
    """Convert a RobotAuthorization to a plain dict."""
    if hasattr(auth, "model_dump"):
        return auth.model_dump()  # type: ignore[union-attr]
    return {
        "authorization_id": auth.authorization_id,
        "robot_category": auth.robot_category,
        "procedure_id": auth.procedure_id,
        "patient_id": auth.patient_id,
        "site_id": auth.site_id,
        "gate_1_operator": auth.gate_1_operator,
        "gate_2_supervisor": auth.gate_2_supervisor,
        "gate_3_sponsor": auth.gate_3_sponsor,
        "gate_4_regulatory": auth.gate_4_regulatory,
        "overall_status": auth.overall_status,
        "issued_by": auth.issued_by,
        "timestamp": auth.timestamp,
    }
