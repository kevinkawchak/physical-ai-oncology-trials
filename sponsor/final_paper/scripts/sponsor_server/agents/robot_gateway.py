"""Robot execution gateway -- robot authorisation with 4-gate safety protocol.

Implements the four-gate authorisation model for physical robot actions
in the oncology trial simulation:

  Gate 1 - Operator confirmation (site-level)
  Gate 2 - Supervisor approval (clinical accountability agent)
  Gate 3 - Sponsor clearance (study orchestrator / safety agent)
  Gate 4 - Regulatory hold check (regulatory agent)

All ten robot categories from the Physical-AI Oncology framework are
supported.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sponsor_server.models import (
    GateStatus,
    RobotAuthorization,
    RobotCategory,
    _now_iso,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------
GATE_NAMES: list[str] = [
    "gate_1_operator",
    "gate_2_supervisor",
    "gate_3_sponsor",
    "gate_4_regulatory",
]

# Categories that require ALL four gates (high-risk)
HIGH_RISK_CATEGORIES: set[str] = {
    RobotCategory.SURGICAL_RESECTION.value,
    RobotCategory.RADIATION_DELIVERY.value,
    RobotCategory.CATHETER_PORT_PLACEMENT.value,
    RobotCategory.BIOPSY_SAMPLING.value,
}

# Categories where gate 4 (regulatory) can be auto-passed
LOW_RISK_CATEGORIES: set[str] = {
    RobotCategory.PATIENT_TRANSPORT.value,
    RobotCategory.DISINFECTION_STERILISATION.value,
    RobotCategory.TELEPRESENCE_MONITORING.value,
}

# ---------------------------------------------------------------------------
# In-memory authorisation store
# ---------------------------------------------------------------------------
_authorizations: dict[str, RobotAuthorization] = {}
_hold_active: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def request_authorization(
    robot_category: str,
    procedure_id: str,
    patient_id: str,
    site_id: str,
) -> RobotAuthorization:
    """Create a new robot authorisation request.

    The request starts with all gates in ``pending`` status.  Low-risk
    categories have gate 4 auto-passed.

    Parameters
    ----------
    robot_category:
        One of the ten ``RobotCategory`` values.
    procedure_id:
        Identifier for the specific procedure.
    patient_id:
        Anonymised patient identifier.
    site_id:
        Trial site where the robot action will occur.

    Returns
    -------
    RobotAuthorization with initial gate statuses.
    """
    auth_id = f"RA-{uuid.uuid4().hex[:8]}"

    gate_4_initial = GateStatus.PENDING.value
    if robot_category in LOW_RISK_CATEGORIES:
        gate_4_initial = GateStatus.PASSED.value

    auth = RobotAuthorization(
        authorization_id=auth_id,
        robot_category=robot_category,
        procedure_id=procedure_id,
        patient_id=patient_id,
        site_id=site_id,
        gate_1_operator=GateStatus.PENDING.value,
        gate_2_supervisor=GateStatus.PENDING.value,
        gate_3_sponsor=GateStatus.PENDING.value,
        gate_4_regulatory=gate_4_initial,
        overall_status="pending",
        issued_by="robot_execution_gateway",
        timestamp=_now_iso(),
    )
    _authorizations[auth_id] = auth
    logger.info(
        "Authorization requested: id=%s category=%s procedure=%s",
        auth_id,
        robot_category,
        procedure_id,
    )
    return auth


def advance_gate(authorization_id: str, gate_name: str, status: str) -> RobotAuthorization | None:
    """Advance a specific gate on an existing authorisation.

    Parameters
    ----------
    authorization_id:
        ID returned by ``request_authorization``.
    gate_name:
        One of ``gate_1_operator``, ``gate_2_supervisor``, etc.
    status:
        New status for the gate (``passed``, ``failed``, ``bypassed_emergency``).

    Returns
    -------
    Updated ``RobotAuthorization`` or ``None`` if not found.
    """
    auth = _authorizations.get(authorization_id)
    if auth is None:
        logger.error("Authorization not found: %s", authorization_id)
        return None

    if gate_name not in GATE_NAMES:
        logger.error("Invalid gate name: %s", gate_name)
        return None

    # Enforce sequential gating for high-risk categories
    if auth.robot_category in HIGH_RISK_CATEGORIES:
        gate_index = GATE_NAMES.index(gate_name)
        for prior_gate in GATE_NAMES[:gate_index]:
            prior_value = getattr(auth, prior_gate)
            if prior_value not in (GateStatus.PASSED.value, GateStatus.BYPASSED_EMERGENCY.value):
                logger.warning(
                    "Cannot advance %s: prior gate %s is %s",
                    gate_name,
                    prior_gate,
                    prior_value,
                )
                return auth

    # Apply a global regulatory hold
    if _hold_active and gate_name == "gate_4_regulatory":
        status = GateStatus.FAILED.value

    # Update the gate -- we use object.__setattr__ for Pydantic frozen compat
    object.__setattr__(auth, gate_name, status)
    _update_overall_status(auth)
    logger.info(
        "Gate advanced: auth=%s gate=%s -> %s overall=%s",
        authorization_id, gate_name, status, auth.overall_status,
    )
    return auth


def set_regulatory_hold(active: bool) -> dict[str, Any]:
    """Activate or deactivate a global regulatory hold.

    When active, no authorisation can pass gate 4.
    """
    global _hold_active  # noqa: PLW0603
    _hold_active = active
    logger.warning("Regulatory hold %s", "ACTIVATED" if active else "deactivated")
    return {"hold_active": _hold_active, "timestamp": _now_iso()}


def get_authorization(authorization_id: str) -> RobotAuthorization | None:
    """Retrieve an authorisation by ID."""
    return _authorizations.get(authorization_id)


def list_authorizations(status_filter: str | None = None) -> list[RobotAuthorization]:
    """List all authorisations, optionally filtered by overall status."""
    auths = list(_authorizations.values())
    if status_filter is not None:
        auths = [a for a in auths if a.overall_status == status_filter]
    return auths


def get_category_list() -> list[str]:
    """Return the ten robot category values."""
    return [c.value for c in RobotCategory]


def reset_state() -> None:
    """Clear in-memory state."""
    global _hold_active  # noqa: PLW0603
    _authorizations.clear()
    _hold_active = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _update_overall_status(auth: RobotAuthorization) -> None:
    """Recompute overall status from individual gate values."""
    gates = [getattr(auth, g) for g in GATE_NAMES]
    if any(g == GateStatus.FAILED.value for g in gates):
        new_status = "rejected"
    elif all(g in (GateStatus.PASSED.value, GateStatus.BYPASSED_EMERGENCY.value) for g in gates):
        new_status = "authorized"
    else:
        new_status = "pending"
    object.__setattr__(auth, "overall_status", new_status)
