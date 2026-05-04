# 08 - Simulation 4 Daily Metrics Dashboard - Full Page

## Purpose

Replace the dense `tab:sim4-daily` 7-row daily metrics table in Section 3.4
(Results, Simulation 4) with a full-page dashboard that shows the daily
patient counts, PSL trajectory, daily decision load, and notable events at
a glance.

## Source Paper Section

`sections/results.tex` lines 519 to 538 (Table sim4-daily).

## Image Properties

- Filename: `images/08_sim4_daily_metrics_dashboard.png`
- DPI: 300
- Size: 11 inches wide by 8.5 inches tall (US letter landscape, full page)
- Background: white (#FFFFFF)
- Palette: navy primary (#1F4E79), green accent (#2C7A4D), gold accent
  (#B45424), pale gray fills (#F0F0F0).

## Layout

- Top header: "Simulation 4 - 168 Hour 7-Day Sponsor Extension Daily
  Metrics."
- Top row of three KPI cards (week totals): 2,016 sponsor decisions, 1,336
  patients processed, 125 escalations.
- Middle panel left: a clustered bar chart showing daily patient counts
  alongside daily decision load (two y axes).
- Middle panel right: a line chart showing PSL trajectory from 63.4 to 70.0
  across the 7 days with start and end markers and the +6.6 delta
  annotated.
- Bottom panel: a 7-row notes strip that lists each day's notable event
  (Day 3 safety day with 22 escalations, Day 5 analysis day biostats peak,
  etc.).

## Daily Metrics Data

| Day   | Hour Range | Patients | PSL Start | PSL End | Daily Decisions | Notable Event              |
| ----- | ---------- | -------- | --------- | ------- | --------------- | -------------------------- |
| Day 1 | H000-H023  | 168      | 63.4      | 64.8    | 288             | Initialization cohort      |
| Day 2 | H024-H047  | 195      | 64.8      | 65.9    | 285             | Enrollment ramp            |
| Day 3 | H048-H071  | 218      | 66.0      | 67.0    | 290             | Safety day, 22 escalations |
| Day 4 | H072-H095  | 200      | 67.0      | 67.9    | 290             | Mid-week steady state      |
| Day 5 | H096-H119  | 195      | 68.0      | 68.8    | 295             | Analysis day, biostats peak|
| Day 6 | H120-H143  | 210      | 68.8      | 69.5    | 290             | Pre-closeout audit         |
| Day 7 | H144-H167  | 150      | 69.5      | 70.0    | 278             | Closeout, audit signoff    |

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- Black text on light fill.
- Constrained layout.

## Suggested Caption

Figure 8: 168-hour 7-day sponsor extension with patient counts, PSL
trajectory, and daily decision load.
