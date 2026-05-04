# 23 - Commit Cadence Timeline (NEW)

## Purpose

Add a NEW commit cadence comparison timeline in Section 6 (Conclusions)
that contrasts the 1 commit per hour cadence of the four LLM simulations
with the typical retrospective per-cohort or per-paper release of the
supervised baselines (Manz 2020, SHIELD-RT, SCORPIO, PROGPATH, AIM-LCpro,
Huang 2025).

## Source Paper Section

`sections/conclusions.tex` Section 6 (persistent themes - hourly commit
cadence) and `sections/introduction.tex` Section 1.2 (AI baseline release
cadences).

## Image Properties

- Filename: `images/23_commit_cadence_timeline.png`
- DPI: 300
- Size: 10 inches wide by 5 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: simulation green (#2C7A4D) for the high-frequency commit dots,
  baseline gray (#7C7C7C) for the low-frequency release dots, light grid
  (#E6E6E6).

## Layout

- Two horizontal timeline rows.
- Top row: "Four LLM Simulations - 1 Commit per Hour." Dense green dots,
  one per simulated hour, across a 1 week window (168 hours) showing the
  Sim 4 cadence specifically.
- Bottom row: "Supervised Baselines - 1 Release per Cohort or per Paper."
  Sparse gray dots placed approximately at the months when each baseline
  released its primary paper (Manz 2020, SHIELD-RT 2020, SCORPIO 2025,
  PROGPATH 2025, AIM-LCpro 2025, Huang 2025).
- Above each row: a label with the cumulative artifact count.
- Header: "Hourly Commit Cadence vs Per-Cohort Release Cadence."
- Bottom band: "168 commits in 7 days from Sim 4; 6 papers in 5 years from
  the supervised baseline set."

## Cadence Data

- Top row: 168 dots evenly spaced at 1 per hour spanning hour 0 through
  hour 167. Day boundaries at hour 24, 48, 72, 96, 120, 144 with thin
  vertical gridlines.
- Bottom row: 6 dots placed at the years 2020 (Manz, SHIELD-RT), 2025
  (SCORPIO, PROGPATH, AIM-LCpro, Huang 2025) on a separate 2020-2025
  timescale. Use a small inset axis or a separate row with its own X
  range.

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.

## Suggested Caption

Figure 23: One commit per hour cadence (168 commits in 7 days from Sim 4)
versus retrospective supervised model release cycles (6 papers in 5 years).
