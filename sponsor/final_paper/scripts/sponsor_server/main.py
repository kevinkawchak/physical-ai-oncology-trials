"""FastAPI application for the autonomous sponsor simulation.

Aggregates routers from the governance, operations, safety, and robotics
domains into a single ASGI application.  When FastAPI is not installed
the module exposes the same functionality through plain-function wrappers
so the logic can be exercised in standalone / test mode.

Endpoints
---------
- POST /api/v1/sponsor/decision
- POST /api/v1/sponsor/hour/{hour_id}/direct
- GET  /api/v1/sponsor/hour/{hour_id}/status
- POST /api/v1/sponsor/robot/authorize
- GET  /api/v1/sponsor/simulation/summary
"""
from __future__ import annotations

import logging
from typing import Any

from sponsor_server.agents import AGENT_NAMES, study_orchestrator
from sponsor_server.models import SimulationSummary
from sponsor_server.routers import governance, operations, robotics, safety

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application (conditional)
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False

if _HAS_FASTAPI:
    app = FastAPI(
        title="Sponsor Simulation Server",
        description=(
            "Autonomous sponsor simulation for physical-AI oncology trials. "
            "Coordinates 12 agents across 24 simulated hours with data from "
            "new-trial/hour-00 through hour-23."
        ),
        version="0.1.0",
    )

    # Include domain routers
    if governance.router is not None:
        app.include_router(governance.router)
    if operations.router is not None:
        app.include_router(operations.router)
    if safety.router is not None:
        app.include_router(safety.router)
    if robotics.router is not None:
        app.include_router(robotics.router)

    # ------------------------------------------------------------------
    # Top-level convenience endpoints (delegate to routers / orchestrator)
    # ------------------------------------------------------------------
    @app.get("/api/v1/sponsor/simulation/summary", tags=["simulation"])
    async def simulation_summary(trial_id: str = "SIM-001") -> dict[str, Any]:
        """Compile the end-of-simulation summary."""
        summary = study_orchestrator.build_simulation_summary(trial_id)
        return _summary_to_dict(summary)

    @app.get("/api/v1/sponsor/agents", tags=["simulation"])
    async def list_agents() -> list[str]:
        """Return the canonical list of 12 agent names."""
        return list(AGENT_NAMES)

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

else:
    app = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Standalone API (usable without starting the server)
# ---------------------------------------------------------------------------
def get_simulation_summary(trial_id: str = "SIM-001") -> dict[str, Any]:
    """Build the simulation summary as a plain dict."""
    summary = study_orchestrator.build_simulation_summary(trial_id)
    return _summary_to_dict(summary)


def make_decision(hour_id: int = 0, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Issue a governance decision (delegates to the governance router)."""
    return governance.make_decision(hour_id, context)


def direct_hour(hour_id: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ingest hour data and produce a directive (delegates to operations)."""
    return operations.direct_hour_standalone(hour_id, payload)


def get_hour_status(hour_id: int) -> dict[str, Any] | None:
    """Retrieve status for a simulation hour."""
    return operations.get_hour_status_standalone(hour_id)


def authorize_robot(
    robot_category: str = "telepresence_monitoring",
    procedure_id: str = "",
    patient_id: str = "",
    site_id: str = "",
) -> dict[str, Any]:
    """Request a robot authorisation (delegates to robotics router)."""
    return robotics.authorize_standalone(robot_category, procedure_id, patient_id, site_id)


def classify_ae(
    hour_id: int = 0,
    ae_term: str = "",
    grade: int = 1,
    serious: bool = False,
    causality: str = "unrelated",
    patient_id: str = "",
) -> dict[str, Any]:
    """Classify an adverse event (delegates to safety router)."""
    return safety.classify_ae_standalone(
        hour_id=hour_id,
        ae_term=ae_term,
        grade=grade,
        serious=serious,
        causality=causality,
        patient_id=patient_id,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _summary_to_dict(summary: SimulationSummary) -> dict[str, Any]:
    """Convert a SimulationSummary to a plain dict."""
    if hasattr(summary, "model_dump"):
        return summary.model_dump()  # type: ignore[union-attr]
    return {
        "trial_id": summary.trial_id,
        "total_hours": summary.total_hours,
        "decisions_made": summary.decisions_made,
        "safety_escalations": summary.safety_escalations,
        "robot_authorizations": summary.robot_authorizations,
        "patients_enrolled": summary.patients_enrolled,
        "sites_active": summary.sites_active,
        "agent_metrics": summary.agent_metrics,
        "phase_transitions": summary.phase_transitions,
        "final_status": summary.final_status,
        "created_at": summary.created_at,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    if _HAS_FASTAPI:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        logger.warning("FastAPI not installed -- running standalone smoke test")
        print("Agents:", AGENT_NAMES)
        result = direct_hour(0)
        print("Hour 0 directive:", result)
        print("Summary:", get_simulation_summary())
