# 02 - Simulation 1 Hour 12 Patient Flow Gantt

## Purpose

Replace the hour-12 patient flow ASCII block in Section 3.1 (Results,
Simulation 1) with a Gantt-style minute-resolution chart showing the eight
named patients across the four sites for the 12:00 to 12:59 UTC slot.

## Source Paper Section

`sections/results.tex` lines 157 to 176 (the hour-12 patient flow verbatim
block).

## Image Properties

- Filename: `images/02_sim1_patient_flow.png`
- DPI: 300
- Size: 10 inches wide by 5 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: site-color encoded blocks - SITE-A #1F4E79, SITE-B #2C7A4D,
  SITE-C #B45424, SITE-D #6A4C8C
- Recovery cooling color #B6CFE6 pale blue

## Layout

- Y axis: 8 patient identifiers (PAT-CONT-0051A through PAT-CONT-0058A) with
  the procedure label on the right of the bar.
- X axis: minute 00 through minute 59 with major tick every 10 minutes.
- Stage rectangles per patient mapped to procedure (BX, RT, ABL, CMP, IMG)
  plus a separate cooling and recovery (REC) light-fill rectangle on the
  same row.
- Site is encoded by row color and a left-side site stripe.
- Header: "Hour 12 Patient Flow (12:00-12:59 UTC)."
- Footer note: 5 arrivals, 4 departures (PAT-CONT-0002, PAT-CONT-0007,
  PAT-CONT-0018A, PAT-CONT-0030A), 1 AE-002 resolved at 12:30.

## Patient Data

- PAT-CONT-0051A SITE-B HCC ablation: minute 00 to 25 ABL, 26 to 35 cool, 36
  to 59 REC.
- PAT-CONT-0052A SITE-A GBM RT10 wrap: minute 00 to 14 RT, exit.
- PAT-CONT-0053A SITE-C mantle BX: minute 00 to 18 BX, 19 to 59 REC.
- PAT-CONT-0054A SITE-A NSCLC RT3: minute 08 to 34 (CHK plus POS plus RT
  plus exit).
- PAT-CONT-0055A SITE-B sarcoma BX: minute 20 to 33 BX, 34 to 59 REC.
- PAT-CONT-0056A SITE-C SCLC RT2: minute 35 to 58 RT.
- PAT-CONT-0057A SITE-D pediatric visit: minute 42 to 55 CMP.
- PAT-CONT-0058A SITE-A HCC img: minute 50 to 59+ IMG.

## Style Rules

- Single dashes only.
- Black text on light fill.
- Constrained layout.

## Suggested Caption

Figure 2: Hour 12 minute-resolution patient flow Gantt across the four-site
continuous RTCT network.
