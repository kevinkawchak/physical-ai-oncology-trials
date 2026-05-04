# 18 - Patient Safety Pipeline Funnel (NEW, Full Page)

## Purpose

Add a NEW funnel chart in Section 4.3 (Discussion, Significance for Patient
Safety and Efficacy) that visualizes the patient safety pipeline from
prescreening through 36 month surveillance and closeout, using the
Simulation 2 ten-stage journey as the canonical site-side flow.

## Source Paper Section

`sections/results.tex` Section 3.2 (Sim 2 stages) and
`sections/discussion.tex` Section 4.3 (significance).

## Image Properties

- Filename: `images/18_patient_safety_funnel.png`
- DPI: 300
- Size: 8.5 inches wide by 11 inches tall (US letter portrait, full page)
- Background: white (#FFFFFF)
- Palette: graduated green family from light pale (#E5F0E8) at the top to
  deep forest (#1F4E2C) at the bottom; safety annotation accent red
  (#C9302C) for the AE filter; closeout gold (#B45424) accent.

## Layout

- Centered vertical funnel with ten trapezoid layers, each layer wider on
  top and narrower toward the bottom (or use a horizontal funnel if it
  reads better).
- Each layer labeled with stage number and stage name plus the surviving
  cohort count for a hypothetical 1,000-patient cohort that flows through
  the funnel.
- Right side: per-stage safety filter description (eligibility filter at
  prescreening, consent filter at enrollment, twin verification at twin
  init, qualification at robot qual, intra-op safety filter at surgery,
  recovery monitoring filter, immunotherapy AE filter, federation privacy
  filter, surveillance filter, closeout filter).
- Bottom band: a takeaway sentence that reads "1M token context permits
  per-stage safety filtering at minute resolution; supervised models
  filter at per-visit resolution."
- Header: "Patient Safety Pipeline Funnel - 10 Stages with Per-Stage
  Filter."

## Funnel Cohort Data (Hypothetical 1,000-Patient Cohort)

| Stage | Stage Name           | Cohort Surviving Filter |
| ----- | -------------------- | ----------------------- |
| 1     | Prescreening         | 1,000                   |
| 2     | Enrollment           | 720                     |
| 3     | Digital twin init    | 720                     |
| 4     | Robot qualification  | 700                     |
| 5     | Surgery              | 690                     |
| 6     | Post-op recovery     | 680                     |
| 7     | Immunotherapy        | 650                     |
| 8     | Federated learning   | 645                     |
| 9     | Surveillance         | 630                     |
| 10    | Closeout             | 615                     |

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- Black text on light fill (white text only on the deeper green levels with
  high enough contrast to remain legible).

## Suggested Caption

Figure 18: Patient safety pipeline funnel from prescreening through 36 month
surveillance and closeout, illustrating the per-stage filtering rate for a
hypothetical 1,000-patient cohort.
