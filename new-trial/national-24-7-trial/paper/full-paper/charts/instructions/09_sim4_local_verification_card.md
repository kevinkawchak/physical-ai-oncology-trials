# 09 - Simulation 4 Local Verification Card (Core i5-6200U / 4 GB)

## Purpose

Replace the local verification ASCII block in Section 3.4 (Results,
Simulation 4) with a clean three-pane verification card that documents
the hardware, the mitigations applied, and the run results.

## Source Paper Section

`sections/results.tex` lines 618 to 641 (the local verification verbatim
block).

## Image Properties

- Filename: `images/09_sim4_local_verification_card.png`
- DPI: 300
- Size: 9.5 inches wide by 5.5 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: navy primary (#1F4E79), gray section fills (#F0F0F0), green
  accent (#2C7A4D) for pass marks, gold accent (#B45424) for the partial
  pass mark on hour 134.

## Layout

- Top title: "Simulation 4 Local Verification - Core i5-6200U / 4 GB RAM /
  Windows 10 Pro."
- Three vertically stacked panels:
  1. Hardware panel: CPU (Intel Core i5-6200U at 2.30 GHz, 2 cores 4 threads,
     2015), RAM (4 GB DDR4), Storage (256 GB SSD), OS (Windows 10 Pro
     22H2), Python (3.10.12 CPython, no extensions).
  2. Mitigations panel: thermal throttling watchdog at 85 C poll every 30
     seconds, Windows Update set to manual and deferred, antivirus
     exclusion on the 168_hours/ directory, pagefile sized to 4096 MB on
     SSD.
  3. Results panel: 168/168 hourly scripts started, 167/168 completed end
     to end, sponsor_168h_summary.json mirrors cloud output within 1%
     tolerance on patient counts, identical on PSL trajectory.
- Right edge: a small overall verdict banner reading "Partial pass: 167/168
  scripts completed; hour 134 paused for thermal throttling and resumed
  cleanly after intervention."

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS (e.g., temperature reference uses
  Celsius written out).
- No degree symbol issues; render "85 C" with single space.

## Suggested Caption

Figure 9: Local verification card for the Core i5-6200U 4 GB Windows 10 Pro
reproduction of Simulation 4 with thermal throttling mitigation and 167/168
hourly scripts completed.
