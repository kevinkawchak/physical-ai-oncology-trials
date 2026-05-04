# 24 - Safety vs Efficacy Quadrant (NEW, Full Page)

## Purpose

Add a NEW 2x2 quadrant chart in Section 4.3 (Discussion, Significance for
Patient Safety and Efficacy) that positions the four author simulations
and the named supervised baselines on a safety-prediction axis (X) and an
efficacy-prediction axis (Y).

## Source Paper Section

`sections/discussion.tex` Section 4.3 (significance for patient safety
and efficacy).

## Image Properties

- Filename: `images/24_safety_efficacy_quadrant.png`
- DPI: 300
- Size: 9 inches wide by 9 inches tall (square, full page on portrait
  letter)
- Background: white (#FFFFFF)
- Palette: simulation markers green (#2C7A4D), supervised baseline markers
  gray (#7C7C7C), foundation models markers purple (#6A4C8C). Quadrant
  background tints very light (alpha 0.10) per quadrant.

## Layout

- X axis (Safety prediction depth): 0 (low) to 5 (high).
- Y axis (Efficacy prediction depth): 0 (low) to 5 (high).
- Four quadrant labels:
  - Top right: "High safety + High efficacy" (target zone).
  - Top left: "Low safety + High efficacy."
  - Bottom right: "High safety + Low efficacy."
  - Bottom left: "Low safety + Low efficacy."
- Markers placed per the data table; each marker labeled with the model or
  simulation name. Markers in the four-simulation set are square; named
  supervised baselines are circle; multimodal foundation models are
  triangle.
- Header: "Safety vs Efficacy Prediction Depth: Four LLM Simulations vs
  Supervised and Foundation Baselines."

## Marker Data

| Name                | X (Safety) | Y (Efficacy) | Marker Type |
| ------------------- | ---------- | ------------ | ----------- |
| Manz 2020           | 3.5        | 1.5          | circle      |
| SHIELD-RT 2020      | 4.0        | 2.0          | circle      |
| SCORPIO 2025        | 2.5        | 3.5          | circle      |
| PROGPATH 2025       | 2.0        | 4.0          | triangle    |
| AIM-LCpro 2025      | 2.5        | 4.0          | triangle    |
| Huang 2025 null     | 1.5        | 1.5          | circle      |
| Sim 1 (site)        | 4.5        | 4.0          | square      |
| Sim 2 (site)        | 4.0        | 4.5          | square      |
| Sim 3 (sponsor)     | 4.5        | 4.5          | square      |
| Sim 4 (sponsor)     | 5.0        | 4.5          | square      |

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- Each marker label placed near the marker without overlap; use small
  offset and dashed leader line where needed.

## Suggested Caption

Figure 24: Safety versus efficacy prediction depth quadrant positioning the
four simulations against named supervised and multimodal foundation model
baselines.
