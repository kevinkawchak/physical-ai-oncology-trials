# 05 - Simulation 2 Stage Table with USL Trajectory

## Purpose

Replace the dense `tab:sim2-stages` 10-row table in Section 3.2 (Results,
Simulation 2) with a hybrid table-plus-USL-trajectory figure. The figure
preserves the regulatory mapping content while making the USL trajectory
and the Robot column visually scannable.

## Source Paper Section

`sections/results.tex` lines 281 to 303 (Table sim2-stages).

## Image Properties

- Filename: `images/05_sim2_stage_usl_table.png`
- DPI: 300
- Size: 9.5 inches wide by 5 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: header navy (#1F4E79) on white text, alternating row fills
  off-white (#F5F5F5) and white. USL trajectory line accent green (#2C7A4D).

## Layout

- Left two-thirds: a clean table with five columns (Stage number, Stage Name,
  Description, Robot, USL Score). Rows alternate fill color for scannability.
- Right one-third: a small USL trajectory chart showing the 6 stages with
  defined USL (Stages 4 through 7 plus 9) on a 0 to 100 scale with the
  baseline 86.0 marked and the peak 88.75 highlighted.
- Header: "Simulation 2 Ten-Stage Journey for Patient PAT-2026-0042. USL =
  Usability Safety Level (0 to 100)."
- Footer note: "Stages 1, 2, 3, 8, and 10 have no robot or USL value."

## Stage Data

| Stage | Name              | Description                          | Robot           | USL    |
| ----- | ----------------- | ------------------------------------ | --------------- | ------ |
| 1     | Prescreening      | EHR scan, eligibility check          | n/a             | n/a    |
| 2     | Enrollment        | Consent, regulatory snapshot         | n/a             | n/a    |
| 3     | Digital twin init | Patient digital twin instantiation   | n/a             | n/a    |
| 4     | Robot qualification| da Vinci Xi qualification at site   | da Vinci Xi     | 86.0   |
| 5     | Robotic surgery   | Lobectomy, R0, neg margins, 168 min  | da Vinci Xi     | 87.5   |
| 6     | Post-op recovery  | Cardio, respiratory, pain monitor    | monitor cobot   | 87.0   |
| 7     | Immunotherapy     | 35 cycles pembrolizumab over 24 mo   | Franka Emika    | 88.75  |
| 8     | Federated learning| Cross-site model update              | n/a             | n/a    |
| 9     | Surveillance      | 36 mo survival monitoring + DT       | n/a             | 88.5   |
| 10    | Closeout          | Final report, archive, audit         | n/a             | n/a    |

## Style Rules

- Single dashes only, no em or en dashes.
- Section sign U+00A7 where source uses SS.
- Constrained layout, black text on light fill.

## Suggested Caption

Figure 5: Ten-stage trial path with robotic platform assignment and Usability
Safety Level scores across the 1,120 day patient journey.
