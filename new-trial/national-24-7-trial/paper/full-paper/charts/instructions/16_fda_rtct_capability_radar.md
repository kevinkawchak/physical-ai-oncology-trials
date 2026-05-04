# 16 - FDA RTCT Capability Radar (NEW, Full Page)

## Purpose

Add a NEW capability radar chart in Section 4.1 (Discussion, FDA RTCT) that
visualizes the seven-axis comparison of FDA RTCT 28 April 2026
proof-of-concept versus the four author simulations. This figure does not
exist in the v3.6.0 paper and was requested for the v3.7.0 chart pack.

## Source Paper Section

`sections/discussion.tex` Section 4.1 (intro discussion of FDA RTCT) and
`sections/introduction.tex` Section 1.1 (FDA announcement).

## Image Properties

- Filename: `images/16_fda_rtct_capability_radar.png`
- DPI: 300
- Size: 9 inches wide by 9 inches tall (square, full page on portrait
  letter with margins)
- Background: white (#FFFFFF)
- Palette: FDA RTCT polygon blue (#4A7BAA) at alpha 0.35; four-simulation
  polygon green (#2C7A4D) at alpha 0.45; gridlines light gray.

## Layout

- Centered radar chart with 7 axes evenly spaced at 360 / 7 degrees:
  1. Trial count
  2. Site count
  3. Robotic integration
  4. Real-time signal types
  5. Cadence depth
  6. Predictive context size
  7. Local-side compute
- Each axis scaled 0 to 5 with 0 at the center.
- Two overlaid filled polygons for the FDA RTCT and the four author
  simulations.
- Header: "FDA RTCT 28 April 2026 vs Four LLM Simulations - Capability
  Radar."
- Right-side legend with two named entries.
- Bottom band: numeric callouts for the three biggest deltas - robotic
  integration (FDA 0 vs Sim 5), predictive context size (FDA 1 vs Sim 5),
  local-side compute (FDA 0 vs Sim 5).

## Score Data

| Axis                      | FDA RTCT 2026 PoC | Four LLM Simulations |
| ------------------------- | ----------------- | -------------------- |
| Trial count               | 2                 | 4                    |
| Site count                | 2                 | 4                    |
| Robotic integration       | 0                 | 5                    |
| Real-time signal types    | 2                 | 4                    |
| Cadence depth             | 3                 | 5                    |
| Predictive context size   | 1                 | 5                    |
| Local-side compute        | 0                 | 5                    |

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- Black text on light fill.
- Filled polygons with alpha so the overlap area is visible.

## Suggested Caption

Figure 16: Capability radar comparing FDA RTCT pharmacology proof-of-concept
to the four LLM simulations across seven axes including robotics integration
and 1M token predictive context.
