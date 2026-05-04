# 20 - Financial Assessment Dashboard (NEW, Full Page)

## Purpose

Add a NEW full-page financial assessment dashboard in Section 4.3
(Discussion, Significance) that visualizes the FDA $390 million to $650
million per-trial cost reduction band, the per-stage savings driver, and
the per-simulation cost-to-reproduce summary across the four author
simulations.

## Source Paper Section

`sections/results.tex` Section 3.2 (Sim 2 cost) and
`sections/discussion.tex` Section 4.3 (significance).

## Image Properties

- Filename: `images/20_financial_assessment_dashboard.png`
- DPI: 300
- Size: 11 inches wide by 8.5 inches tall (US letter landscape, full page)
- Background: white (#FFFFFF)
- Palette: navy primary (#1F4E79), green positive (#2C7A4D), gold neutral
  (#B45424), light fills #F5F5F5.

## Layout

- Top row: three KPI cards.
  1. "Baseline Trial Cost: $1.30 billion."
  2. "FDA-cited Reduction Band: $390M - $650M (30 to 50 percent)."
  3. "Per-Patient Run (Sim 2): $0.91M."
- Middle left: a bar chart titled "Per-Stage Savings Drivers." Stages 1
  through 10 on the X axis, savings in million USD on the Y axis.
- Middle right: a horizontal bar chart titled "Per-Simulation Cost to
  Reproduce." Sim 1 cloud only, Sim 2 cloud plus light local, Sim 3 cloud
  only, Sim 4 cloud plus Core i5-6200U laptop. Costs are illustrative.
- Bottom band: a single takeaway sentence "Track B (cloud plus small local
  agents) is the architecture that maps the FDA reduction band to per-site
  hardware floors."

## Per-Stage Savings (Million USD per Patient Run)

| Stage | Savings |
| ----- | ------- |
| 1     | 0.02    |
| 2     | 0.03    |
| 3     | 0.04    |
| 4     | 0.05    |
| 5     | 0.07    |
| 6     | 0.06    |
| 7     | 0.08    |
| 8     | 0.02    |
| 9     | 0.02    |
| 10    | 0.01    |

## Per-Simulation Cost to Reproduce (Illustrative, Million USD)

- Sim 1 (cloud only, 56h): 0.020
- Sim 2 (cloud plus light local): 0.012
- Sim 3 (cloud only, 24h): 0.016
- Sim 4 (cloud plus Core i5-6200U laptop): 0.008

## Style Rules

- Single dashes only, no em dashes.
- Section sign U+00A7 where source uses SS.
- Black text on light fill.

## Suggested Caption

Figure 20: Financial assessment dashboard for the FDA $390 million to $650
million per-trial cost reduction with per-stage savings drivers and per-
simulation cost to reproduce.
