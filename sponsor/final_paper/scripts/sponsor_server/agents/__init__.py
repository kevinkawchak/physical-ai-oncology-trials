"""Agent modules for the autonomous sponsor simulation.

Twelve agents collaborate to run the 24-hour simulated trial:

  1. portfolio_agent        - Programme prioritisation, go/no-go
  2. asset_lead_agent       - Compound-level strategy
  3. clinical_accountability_agent - Medical oversight
  4. study_orchestrator      - Master task routing and escalation
  5. clinops_agent           - Site startup, enrollment, monitoring
  6. safety_agent            - AE classification, E2B, safety signals
  7. regulatory_agent        - IND lifecycle, filings, compliance
  8. quality_agent           - GCP/GMP quality system
  9. supply_agent            - Drug supply and logistics
 10. data_biostats_agent     - Data management and biostatistics
 11. site_gateway            - Site-level coordination
 12. robot_execution_gateway - Physical robot authorisation
"""

from __future__ import annotations

AGENT_NAMES: list[str] = [
    "portfolio_agent",
    "asset_lead_agent",
    "clinical_accountability_agent",
    "study_orchestrator",
    "clinops_agent",
    "safety_agent",
    "regulatory_agent",
    "quality_agent",
    "supply_agent",
    "data_biostats_agent",
    "site_gateway",
    "robot_execution_gateway",
]

__all__ = ["AGENT_NAMES"]
