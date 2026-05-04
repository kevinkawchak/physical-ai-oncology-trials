# 04 - Simulation 2 Patient Journey Timeline (1,120 Days)

## Purpose

Replace the PAT-2026-0042 1,120 day journey ASCII timeline in Section 3.2
(Results, Simulation 2) with a clean stage-by-day timeline figure that
shows the ten stages mapped to their day windows and cumulative cost
trajectory.

## Source Paper Section

`sections/results.tex` lines 320 to 337 (the journey timeline verbatim
block) plus the journey table on lines 281 to 303.

## Image Properties

- Filename: `images/04_sim2_journey_timeline.png`
- DPI: 300
- Size: 10 inches wide by 5.5 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: stage-color encoded bars - cool teal (#2C7A7A) for prep stages,
  green (#2C7A4D) for active treatment, gold (#B45424) for surveillance,
  navy (#1F4E79) for closeout. Cost overlay line in dark accent (#7A1F1F).

## Layout

- Y axis (left): 10 stage rows from Stage 1 Prescreening through Stage 10
  Closeout.
- X axis: Day -45 through Day 1095 (logarithmic-friendly broken into pre-day
  zero and post-day zero with a vertical gridline at Day 0 marking surgery).
- Stage bars sized to their day window and labeled with USL score where
  defined.
- Right Y axis: cumulative cost in million USD with a single overlaid line
  starting at $0 and ending at $0.91 million per patient run with $1.30
  million baseline reference line.
- Header: "Patient PAT-2026-0042 - 10 Stage Journey Timeline (1,120 days)."
- Footer annotation: "Cumulative cost $0.91M against $1.30M baseline,
  savings $0.39M (30%)."

## Stage Data

- Stage 1 Prescreening: Day -45 (single day point).
- Stage 2 Enrollment: Day -30.
- Stage 3 Digital Twin: Day -15.
- Stage 4 Robot Qualification: Day -7, USL 86.0 baseline.
- Stage 5 Surgery: Day 0, USL 87.5 procedural.
- Stage 6 Recovery: Day 1 to 30, USL 87.0.
- Stage 7 Immunotherapy: Day 31 to 720 (35 cycles pembrolizumab), USL 88.75.
- Stage 8 Federated Learning: Day 90 to 720.
- Stage 9 Surveillance: Day 1 to 1095, USL 88.5.
- Stage 10 Closeout: Day 1095 (single point).

## Style Rules

- Single dashes only.
- Section sign U+00A7 where the source uses SS.
- Black text on light fills, no dark mode.

## Suggested Caption

Figure 4: Patient PAT-2026-0042 ten-stage 1,120-day journey from prescreening
through closeout with cumulative cost trajectory.
