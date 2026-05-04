# 01 - Simulation 1 Site Network Diagram (Hour 00 Cold Start)

## Purpose

Replace the hour-00 facility ASCII block and Table sim1-hour00-network in
Section 3.1 (Results, Simulation 1) with a single full-page network diagram
that shows the four trial sites, the 116 robot inventory split by station
type, and the Paradigm Health to FDA real-time API channel.

## Source Paper Section

`new-trial/national-24-7-trial/paper/full-paper/sections/results.tex` lines
31 through 105 (Table 1 and the verbatim hour-00 facility block).

## Image Properties

- Filename: `images/01_sim1_site_network.png`
- DPI: 300
- Size: 11 inches wide by 8.5 inches tall (US letter landscape, full page)
- Background: white (#FFFFFF)
- Palette: blue family (#1F4E79 dark navy primary, #4A7BAA mid blue, #B6CFE6
  pale blue, #C9302C accent red for the active session and FDA channel)
- All text dark slate (#1A1A1A) on light fill

## Layout

- Title bar at top: "National 24/7 Continuous RTCT - Hour 00 Network Status,
  00:00-00:59 UTC."
- Four site cards in a 2 by 2 grid for SITE-A Houston, SITE-B Philadelphia,
  SITE-C Boston, SITE-D Texas Medical Center. Each card lists patients on
  site, active procedures, robot inventory by station type (KIOSK, SURG,
  COBOT, RTPOS, TRACK, NEEDLE, COMPN, HUMAN, IMAGE), and the per-site signal
  link latency.
- Below the grid, a single horizontal Paradigm Health Aggregator card showing
  the four channels (TRAVERSE, STREAM-SCLC, TRAVERSE-PED, NETWORK-OBS) all
  UP, with the median 13 second FDA acknowledgement latency annotation.
- Footer: total network summary (15 patients, 4 active procedures including 3
  done plus 1 ongoing, 116 robot instances, 4 RTCT endpoints).

## Site Card Data

- SITE-A Houston: 6 on site, 2 active at 00:09 and 00:42, 29 robot instances,
  COBOT-01 active and RTPOS-02 plus TRACK-02 active, signal link UP at 142 ms.
- SITE-B Philadelphia: 4 on site, 1 active at 00:57, 29 robot instances,
  COBOT-02 active, signal link UP at 168 ms.
- SITE-C Boston: 3 on site, 1 active at 00:23, 29 robot instances, IMAGE-01
  active, signal link UP at 201 ms.
- SITE-D Texas Medical Center: 2 on site, 0 active, 29 robot instances,
  NEEDLE-02 calibration scheduled 00:06 to 00:25, COMPN-02 pediatric passive
  monitoring, signal link UP at 119 ms.

## Style Rules

- No em dashes, en dashes, or double dashes anywhere.
- Single dashes for ranges and separators.
- All section symbols use the Unicode section sign (U+00A7) when present.
- Constrained layout, no manual positioning required by the author.
- Black text only on light fills; never light text on dark.

## Suggested Caption

Figure 1: Hour 00 cold start network across SITE-A, SITE-B, SITE-C, and
SITE-D with 116 robot inventory.
