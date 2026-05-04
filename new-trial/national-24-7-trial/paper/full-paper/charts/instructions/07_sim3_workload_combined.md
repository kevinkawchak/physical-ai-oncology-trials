# 07 - Simulation 3 Combined Workload (Hours 00, 12, 23) - Full Page

## Purpose

Per the project brief, combine the three consecutive Sim 3 hourly agent
workload tables (hour 00, hour 12, hour 23) on page 16 into a single
full-page comparison figure. The combined figure removes redundant table
headings and lets a reader compare task, decision, and escalation counts
across the three representative hours at a glance.

## Source Paper Section

`sections/results.tex` lines 397 to 483 (the three hour-00 / hour-12 /
hour-23 verbatim agent workload tables).

## Image Properties

- Filename: `images/07_sim3_workload_combined.png`
- DPI: 300
- Size: 8.5 inches wide by 11 inches tall (US letter portrait, full page)
- Background: white (#FFFFFF)
- Palette: hour-00 navy (#1F4E79), hour-12 gold (#B45424), hour-23 teal
  (#2C7A7A). Escalation pip in red (#C9302C).

## Layout

- Top header: "Simulation 3 Sponsor Agent Workload - Hour 00, Hour 12, and
  Hour 23 of the 24-Hour Run."
- Three sub-grouped horizontal bar groups, one per hour, each on its own
  panel. Within each panel:
  - Y axis: 12 named agents (portfolio_agent, asset_lead_agent,
    clinical_accountability_agent, study_orchestrator, clinops_agent,
    safety_agent, regulatory_agent, quality_agent, supply_agent,
    data_biostats_agent, site_gateway, robot_execution_gateway).
  - X axis: task count from 0 to 6.
  - Stacked or grouped bars showing tasks plus decisions, with a small red
    pip on the agent rows that recorded an escalation that hour.
- Right of each panel: total summary box (TOTAL Tasks, Decisions,
  Escalations).
- Bottom totals strip: a horizontal mini-summary comparing the three hours
  (hour-00 24/18/0, hour-12 34/29/1, hour-23 24/18/0).

## Workload Data

### Hour 00 (overnight, 2 patients in scope)

| Agent                          | Tasks | Decisions | Escalations |
| ------------------------------ | ----- | --------- | ----------- |
| portfolio_agent                | 2     | 1         | 0           |
| asset_lead_agent               | 3     | 2         | 0           |
| clinical_accountability_agent  | 1     | 1         | 0           |
| study_orchestrator             | 2     | 2         | 0           |
| clinops_agent                  | 3     | 2         | 0           |
| safety_agent                   | 1     | 1         | 0           |
| regulatory_agent               | 2     | 1         | 0           |
| quality_agent                  | 3     | 2         | 0           |
| supply_agent                   | 1     | 1         | 0           |
| data_biostats_agent            | 2     | 2         | 0           |
| site_gateway                   | 3     | 2         | 0           |
| robot_execution_gateway        | 1     | 1         | 0           |
| TOTAL                          | 24    | 18        | 0           |

### Hour 12 (mid-day peak, 5 patients in scope)

| Agent                          | Tasks | Decisions | Escalations |
| ------------------------------ | ----- | --------- | ----------- |
| portfolio_agent                | 3     | 1         | 0           |
| asset_lead_agent               | 1     | 1         | 0           |
| clinical_accountability_agent  | 4     | 3         | 1           |
| study_orchestrator             | 2     | 2         | 0           |
| clinops_agent                  | 5     | 5         | 0           |
| safety_agent                   | 3     | 3         | 0           |
| regulatory_agent               | 1     | 1         | 0           |
| quality_agent                  | 4     | 4         | 0           |
| supply_agent                   | 2     | 1         | 0           |
| data_biostats_agent            | 5     | 5         | 0           |
| site_gateway                   | 3     | 2         | 0           |
| robot_execution_gateway        | 1     | 1         | 0           |
| TOTAL                          | 34    | 29        | 1           |

### Hour 23 (overnight, 2 patients in scope, day-end audit)

| Agent                          | Tasks | Decisions | Escalations |
| ------------------------------ | ----- | --------- | ----------- |
| portfolio_agent                | 1     | 1         | 0           |
| asset_lead_agent               | 2     | 2         | 0           |
| clinical_accountability_agent  | 3     | 2         | 0           |
| study_orchestrator             | 1     | 1         | 0           |
| clinops_agent                  | 2     | 1         | 0           |
| safety_agent                   | 3     | 2         | 0           |
| regulatory_agent               | 1     | 1         | 0           |
| quality_agent                  | 2     | 2         | 0           |
| supply_agent                   | 3     | 2         | 0           |
| data_biostats_agent            | 1     | 1         | 0           |
| site_gateway                   | 2     | 1         | 0           |
| robot_execution_gateway        | 3     | 2         | 0           |
| TOTAL                          | 24    | 18        | 0           |

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- Black text on light fill, no dark mode.
- Constrained layout, no manual repositioning required.

## Suggested Caption

Figure 7: Sponsor agent workload across hours 00, 12, and 23 of the 24-hour
autonomous run, combining three consecutive paper tables into a single
comparison.
