"""Router sub-package for sponsor server FastAPI endpoints.

Routers are organised by domain:
  - governance  : sponsor-level decision and programme governance
  - operations  : clinical operations and site management
  - safety      : adverse-event handling, escalation, E2B
  - robotics    : robot authorisation and 4-gate safety protocol
"""
from __future__ import annotations

from sponsor_server.routers import governance, operations, robotics, safety

__all__: list[str] = ["governance", "operations", "robotics", "safety"]
