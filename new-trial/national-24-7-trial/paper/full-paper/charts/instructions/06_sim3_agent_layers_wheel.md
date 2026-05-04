# 06 - Simulation 3 Agent Layers Wheel (53 Core Agents in 4 Layers)

## Purpose

Replace the `tab:sim3-layers` table in Section 3.3 (Results, Simulation 3)
with a wheel diagram (donut plus inner labels) showing the four agent
layers, their counts, and a sample of named agents per layer.

## Source Paper Section

`sections/results.tex` lines 369 to 386 (Table sim3-layers).

## Image Properties

- Filename: `images/06_sim3_agent_layers_wheel.png`
- DPI: 300
- Size: 9 inches wide by 7 inches tall
- Background: white (#FFFFFF)
- Palette: governance navy (#1F4E79), study execution teal (#2C7A7A),
  site/robotics gold (#B45424), trust deep purple (#6A4C8C). All wedge fills
  light, with darker rims.

## Layout

- Center wheel: donut with four wedges proportional to agent count
  (governance 6, study execution 12, site/robotics 18, trust 17 - total 53).
- Inside the wedge, the count is rendered in large bold; outside the wedge,
  the layer name is rendered with two example agent names beneath.
- Right side: a short legend listing the primary responsibility of each
  layer plus a "Notable Hour-12 Load" annotation per layer.
- Header: "53 Core Sponsor Agents Organized into Four Layers."

## Layer Data

- Governance (6 agents): portfolio_agent, asset_lead_agent,
  clinical_accountability_agent. Responsibility: regulatory and oversight
  decisions. Hour-12 load: 8 tasks, 1 escalation.
- Study Execution (12 agents): study_orchestrator, clinops_agent,
  regulatory_agent, quality_agent. Responsibility: day-to-day study
  conduct. Hour-12 load: 12 tasks.
- Site/Robotics (18 agents): site_gateway, robot_execution_gateway,
  supply_agent. Responsibility: site coordination and robotic ops. Hour-12
  load: 6 tasks.
- Trust (17 agents): data_biostats_agent, safety_agent, audit_trail_manager.
  Responsibility: safety and integrity verification. Hour-12 load: 8 tasks.

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- Constrained layout, black text on light fills.

## Suggested Caption

Figure 6: 53 core sponsor agents organized into governance, study execution,
site / robotics, and trust layers.
