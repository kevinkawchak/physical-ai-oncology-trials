# 10 - FDA RTCT Extension Chart (Full Page)

## Purpose

Replace the `tab:disc-fda-extension` table in Section 4.1 (Discussion, FDA
RTCT Comparison) with a full-page side-by-side comparison chart that shows
the seven dimensions on which the four simulations extend the FDA's two
pharmacology proofs-of-concept.

## Source Paper Section

`sections/discussion.tex` lines 71 to 90 (the disc-fda-extension table).

## Image Properties

- Filename: `images/10_fda_extension_chart.png`
- DPI: 300
- Size: 11 inches wide by 8.5 inches tall (US letter landscape, full page)
- Background: white (#FFFFFF)
- Palette: FDA RTCT column blue (#4A7BAA), simulation column green
  (#2C7A4D), header navy (#1F4E79).

## Layout

- Top header: "Capability Extension over the FDA RTCT 28 April 2026 Proof
  of Concept."
- Subheader: "Seven Operational Dimensions, Two Programs Side by Side."
- Two-column matrix layout. Column A (left): FDA RTCT 2026 PoC. Column B
  (right): Four simulations here. Seven rows (one per dimension), each row
  is a card with the dimension label centered between the two side cards.
- Below the matrix, a footer band that calls out the three crucial
  differentiators: 116 robot instances (Sim 1), 1M token context (all
  simulations), Core i5-6200U local verification (Sim 4).
- Optional inline icons (using matplotlib text or simple shapes) per
  dimension to aid quick visual scanning.

## Dimension Data

| Dimension          | FDA RTCT 2026 PoC                                 | Four Simulations Here                                |
| ------------------ | ------------------------------------------------- | ---------------------------------------------------- |
| Trials             | 2 (TRAVERSE, STREAM-SCLC)                         | 4 author simulations covering site and sponsor       |
| Sites              | MD Anderson, Penn (TRAVERSE)                      | 4 sites in Simulation 1, distributed in others        |
| Robots             | Not specified in the announcement                 | 116 robot instances in Simulation 1                   |
| Real-time signals  | Endpoint and safety signals only                  | Endpoint, safety, robot authorization, agent decision |
| Cadence            | Real-time as defined per trial                    | Per-hour minute-resolution + per-stage decision       |
| Predictive layer   | Per-trial supervised endpoints                    | Repository-scale 1M-token context across 4 simulations|
| Local-side compute | Not addressed                                     | Verified on Core i5-6200U / 4 GB / Win 10 (Sim 4)     |

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- Black text on light fill.

## Suggested Caption

Figure 10: Capability extension over the FDA RTCT 28 April 2026 proof-of-
concept across seven dimensions including advanced robotics (116 instances)
and 1M token predictive layer.
