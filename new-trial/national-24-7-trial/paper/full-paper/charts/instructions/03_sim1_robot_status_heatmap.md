# 03 - Simulation 1 Hour 23 Robot Status Heatmap

## Purpose

Replace the hour-23 robot status timeline ASCII block in Section 3.1
(Results, Simulation 1) with a minute-resolution heatmap showing the four
sites and their 116 robot instances. The heatmap highlights the dominant
overnight standby pattern and the few non-standby events (passive
monitoring on COMPN-03 at SITE-A and COMPN-02 at SITE-D, plus the short
imaging session on IMAGE-01 at SITE-D).

## Source Paper Section

`sections/results.tex` lines 189 to 215 (the hour-23 robot status verbatim
block).

## Image Properties

- Filename: `images/03_sim1_robot_status_heatmap.png`
- DPI: 300
- Size: 10 inches wide by 5.5 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: standby (SS) light gray (#E6E6E6), monitoring (MM) blue
  (#4A7BAA), active (AA) green (#2C7A4D), idle dot (..) white with light
  border, error red (#C9302C) reserved for legend only since hour 23 has
  zero errors.

## Layout

- Y axis: rows for SITE-A COMPN-03, SITE-A Other 28, SITE-B All 29, SITE-C
  All 29, SITE-D IMAGE-01, SITE-D COMPN-02, SITE-D Other 27 (7 rows total
  representing the 116 robot instances).
- X axis: minute 00 through minute 59 with major tick every 5 minutes.
- Color cells encode the per-minute robot status using the palette above.
- Header: "Hour 23 Robot Status Timeline (23:00-23:59 UTC), Day 1 Closing."
- Right-side legend with the four states (Standby, Monitoring, Active,
  Idle).
- Footer: hour-23 active robot-min 18, utilization 0.26%, peak 1, errors 0,
  day-1 cumulative active robot-min ~2,500.

## Robot Data Encoding

- SITE-A COMPN-03: MM for every minute 00 through 59.
- SITE-A Other 28 robots: SS for every minute.
- SITE-B All 29 robots: SS for every minute.
- SITE-C All 29 robots: SS for every minute.
- SITE-D IMAGE-01: idle 00 to 29, AA at 30 through 44, idle 45 to 59.
- SITE-D COMPN-02: MM for every minute.
- SITE-D Other 27 robots: SS for every minute.

## Style Rules

- All section signs as Unicode U+00A7 (none expected here, applied as a
  general rule).
- Single dashes only.
- Black text on light fill, no dark mode.

## Suggested Caption

Figure 3: Hour 23 day-1 closing robot status heatmap across 116 instances at
four trial sites with overnight standby dominant and 18 active robot-minutes.
