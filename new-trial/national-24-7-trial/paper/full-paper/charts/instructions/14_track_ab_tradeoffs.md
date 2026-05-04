# 14 - Track A vs Track B Trade-offs

## Purpose

Replace the `tab:future-tracks` table in Section 5.2 (Limitations and
Future Work) with a balanced two-column trade-off comparison chart between
Track A (single big model) and Track B (big model plus small local
agents).

## Source Paper Section

`sections/limitations_future.tex` lines 111 to 130 (tab:future-tracks).

## Image Properties

- Filename: `images/14_track_ab_tradeoffs.png`
- DPI: 300
- Size: 10 inches wide by 6 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: Track A blue (#1F4E79), Track B teal (#2C7A7A), header dark
  navy.

## Layout

- Top header: "Track A vs Track B for a Future RTCT-Aligned Oncology Trial
  AI."
- Subheader: "Single big model performing all tasks vs single big model
  that creates smaller agents performing smaller tasks locally."
- Seven horizontal rows, one per property, with two side cards per row.
- Bottom band: a research-question callout that reads "Both tracks should
  be prototyped before a real-patient trial is designed; the comparison
  itself becomes a research output."

## Property Data

| Property             | Track A: Big Single Model            | Track B: Big Model + Small Agents      |
| -------------------- | ------------------------------------ | -------------------------------------- |
| Architecture         | 1 model, 1 inference loop            | 1 orchestrator plus N specialist agents|
| Inference cost       | Higher per decision                  | Lower per site, higher overall design  |
| Reg. accountability  | Single surface                       | Distributed; each agent has its own floor|
| PHI handling         | Cloud transit required               | Local agents handle PHI on-site        |
| Hardware floor       | Cloud (data-center)                  | Site (Core i5-6200U / 4 GB feasible)   |
| Update cycle         | Single model upgrade                 | Per-agent upgrade, with orchestrator swap|
| Extension target     | Simulation 4 + clinical link         | Simulation 1 site network + per-task agents|

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.

## Suggested Caption

Figure 14: Track A single big model versus Track B big model plus small
local agents for RTCT deployment.
