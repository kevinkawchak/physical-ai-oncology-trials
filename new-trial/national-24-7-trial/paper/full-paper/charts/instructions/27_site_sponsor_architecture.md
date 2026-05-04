# 27 - Site Sponsor Architecture (NEW, Full Page)

## Purpose

Add a NEW full-page architecture diagram in Section 2.2 (Methods,
Simulation Type) that visualizes the four-simulation set as two pairs:
Sims 1 and 2 on the site side and Sims 3 and 4 on the sponsor side, with
explicit signal flow connecting the site stream into the sponsor stream
and onward to the FDA RTCT real-time API through Paradigm Health.

## Source Paper Section

`sections/methods.tex` Section 2.2 (simulation type) and
`sections/discussion.tex` Section 4.1 (FDA RTCT comparison).

## Image Properties

- Filename: `images/27_site_sponsor_architecture.png`
- DPI: 300
- Size: 11 inches wide by 8.5 inches tall (US letter landscape, full page)
- Background: white (#FFFFFF)
- Palette: site cards navy (#1F4E79), sponsor cards gold (#B45424),
  Paradigm Health card teal (#2C7A7A), FDA card dark green (#1F4E2C). All
  fills light, dark text.

## Layout

- Three-column layout left to right.
- Left column: "Sites - Signal Origination."
  - Box: "Sim 1 - Continuous National 24/7 RTCT" (4 sites, 116 robots,
    minute-resolution narrative).
  - Box: "Sim 2 - Single-Patient 10-Stage Journey" (PAT-2026-0042, 1,120
    days).
- Center column: "Paradigm Health Aggregator." Two arrows incoming from
  the left site boxes.
- Right column: "Sponsors - Decision Origination."
  - Box: "Sim 3 - 24-Hour Autonomous Sponsor" (53 agents, 4 layers, 24
    JSON outputs).
  - Box: "Sim 4 - 168-Hour 7-Day Sponsor Extension" (168 scripts, 7 days,
    Core i5-6200U local verification).
- Top spanning the full width: "FDA RTCT Real-Time API" with arrows down
  from Paradigm Health and from each sponsor box.
- Below the diagram: a short note "Sites generate the patient and robot
  signal stream; sponsors generate the governance and regulatory decision
  stream; the FDA RTCT pilot ingests both."
- Header: "Site / Sponsor / FDA RTCT Architecture - Four Simulations Plus
  Two Connection Buses."

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- Black text on light fills.

## Suggested Caption

Figure 27: Site versus sponsor architecture connecting Sims 1, 2 (sites)
to Sims 3, 4 (sponsor) and onward through Paradigm Health to the FDA RTCT
real-time API.
