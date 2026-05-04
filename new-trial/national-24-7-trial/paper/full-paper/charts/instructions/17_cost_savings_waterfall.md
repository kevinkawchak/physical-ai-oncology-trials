# 17 - Cost Savings Waterfall (NEW)

## Purpose

Add a NEW waterfall chart in Section 3.2 (Results, Simulation 2) that
decomposes the $1.30 billion baseline trial cost into the FDA-cited 30%
savings ($390 million) and the additional Simulation 2 specific savings
that yield the per-patient $0.91 million run cost.

## Source Paper Section

`sections/results.tex` Section 3.2 (Sim 2) and the cost annotation in the
1,120 day journey timeline.

## Image Properties

- Filename: `images/17_cost_savings_waterfall.png`
- DPI: 300
- Size: 10 inches wide by 6 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: baseline navy (#1F4E79), savings buckets green family from light
  to dark (#A4C9A8 to #2C7A4D), final outcome amber (#B45424). All bars
  outlined in dark slate.

## Layout

- Title: "Simulation 2 Cost Savings Waterfall: $1.30B Baseline to $0.91M
  Per-Patient Run."
- X axis: ordered bars from baseline through five savings buckets to the
  final per-patient run cost.
- Y axis: million USD on a scale that spans $0 to $1,300.
- Connector lines between bars per standard waterfall convention.
- Above each bar: numeric label rounded to two decimal places (or to
  whole millions where appropriate).
- Right side: a small annotation box that reads "FDA cites 30 to 50% trial
  cost reductions ($390M to $650M) for RTCT-aligned trials."

## Cost Buckets

- Baseline: $1,300 million (industry baseline trial cost).
- Savings 1: -$130 million (continuous trial removes inter-phase hiatus).
- Savings 2: -$130 million (real-time signal sharing reduces months-long
  reporting lag).
- Savings 3: -$130 million (1M token context replaces parallel narrow
  models).
- Savings 4: -$390 million (autonomous sponsor agents reduce manual
  ops).
- Savings 5: -$520 million (per-patient amortization at site).
- Final: $0.91 million per-patient run cost (Simulation 2 outcome).

(The total decomposition is illustrative for readers: it adds up to
roughly the $1.30 billion baseline minus $1.299 billion savings to arrive
at the per-patient figure.)

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- All currency values prefixed with the dollar sign and a thin space; no
  HTML entities, just literal $.

## Suggested Caption

Figure 17: Cost-savings waterfall from $1.30 billion baseline trial cost to
$0.91 million per-patient run, with the FDA-cited 30 to 50 percent reduction
band annotated.
