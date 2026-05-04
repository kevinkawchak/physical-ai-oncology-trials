# 21 - PSL Trajectory across the 168-Hour Run (NEW)

## Purpose

Add a NEW PSL trajectory chart in Section 3.4 (Results, Simulation 4) that
visualizes the PSL score climbing from a start of 63.4 to an end of 70.0
across the 168 hour run, with each day boundary annotated and notable
events called out.

## Source Paper Section

`sections/results.tex` Section 3.4 (Sim 4 daily metrics) and the
`tab:sim4-daily` PSL Start - End column.

## Image Properties

- Filename: `images/21_psl_trajectory.png`
- DPI: 300
- Size: 10 inches wide by 5.5 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: trajectory line forest green (#2C7A4D), day boundary vertical
  lines light gray (#D8D8D8), event annotation amber (#B45424), shaded
  band beneath line at light green tint (#E5F0E8).

## Layout

- X axis: hour 0 through hour 167 with day-boundary vertical gridlines at
  hour 24, 48, 72, 96, 120, 144.
- Y axis: PSL score on a 60 to 75 range.
- A single smooth line interpolated through 8 anchor points (start of each
  day plus end of week) to render the climb from 63.4 to 70.0.
- Anchor markers labeled with day index and PSL value.
- Annotations:
  - Day 3 hour 60: "Safety day, 22 escalations" with arrow.
  - Day 5 hour 108: "Analysis day, biostats peak (7 interim analyses)."
  - Day 7 hour 167: "Closeout, audit signoff (final PSL 70.0)."
- Header: "Simulation 4 PSL Trajectory: 63.4 to 70.0 (+6.6) across 168
  Hours."

## PSL Anchor Data

- Hour 0:   63.4
- Hour 23:  64.8
- Hour 47:  65.9
- Hour 71:  67.0
- Hour 95:  67.9
- Hour 119: 68.8
- Hour 143: 69.5
- Hour 167: 70.0

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.

## Suggested Caption

Figure 21: Patient Safety Liveness trajectory across the 168-hour 7-day
sponsor run, climbing from 63.4 to 70.0 (+6.6) with day-3 safety, day-5
analysis, and day-7 closeout annotations.
