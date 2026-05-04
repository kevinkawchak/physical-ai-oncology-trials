# 26 - Robot Authorization Flowchart (NEW)

## Purpose

Add a NEW robot authorization decision pipeline flowchart in Section 3.3
(Results, Simulation 3) that traces an incoming robot authorization
request through robot_execution_gateway, clinical_accountability_agent,
safety_agent, and the audit_trail_manager, with the typical Sim 4 day-1
volume of 168 robot authorizations and 6 borderline cases annotated.

## Source Paper Section

`sections/results.tex` Section 3.4 (Sim 4 day 1 robot_execution_gateway
clears 168 authorizations of which 6 borderline) and Section 3.3 (Sim 3
agent layers).

## Image Properties

- Filename: `images/26_robot_authorization_flowchart.png`
- DPI: 300
- Size: 10 inches wide by 6 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: gateway navy (#1F4E79), accountability gold (#B45424), safety
  green (#2C7A4D), audit purple (#6A4C8C), arrows dark slate.

## Layout

- Left start node: "Site emits robot authorization request."
- Pipeline (left to right):
  1. robot_execution_gateway: routine clearance.
  2. Decision diamond: borderline?
     - No: directly to clearance bus (top arrow).
     - Yes: forwarded to clinical_accountability_agent.
  3. clinical_accountability_agent: review or approve.
  4. Decision diamond: AE-related?
     - Yes: forwarded to safety_agent.
     - No: clearance bus.
  5. safety_agent: emit signal to FDA real-time API.
  6. audit_trail_manager: record decision.
- Right end node: "Clearance bus to site (authorized) or escalation queue
  (held)."
- Annotations: "Sim 4 Day 1: 168 routine clearances, 6 borderline forwards
  reviewed, 0 escalations to safety_agent at hour 23."
- Header: "Robot Authorization Decision Pipeline (Simulation 3 plus
  Simulation 4)."

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- All arrows clearly oriented and labeled.

## Suggested Caption

Figure 26: Robot authorization decision pipeline across robot_execution_
gateway, clinical_accountability_agent, safety_agent, and audit_trail_
manager with Sim 4 Day 1 volume of 168 authorizations annotated.
