# 30 - RTCT Signal Flow (NEW)

## Purpose

Add a NEW RTCT signal flow chart in Section 4.1 (Discussion, FDA RTCT
Comparison) that traces the real-time signal stream from sites and
sponsors through Paradigm Health to the FDA RTCT API, with the per-channel
endpoint breakdown.

## Source Paper Section

`sections/results.tex` Section 3.1 (Sim 1 hour-00 facility lists the four
TRAVERSE plus STREAM-SCLC plus TRAVERSE-PED plus NETWORK-OBS channels) and
`sections/discussion.tex` Section 4.1.

## Image Properties

- Filename: `images/30_rtct_signal_flow.png`
- DPI: 300
- Size: 10 inches wide by 6 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: site stream blue (#1F4E79), sponsor stream gold (#B45424),
  Paradigm Health teal (#2C7A7A), FDA dark green (#1F4E2C), arrow lines
  dark slate.

## Layout

- Left column: Site signal sources.
  - SITE-A (Houston) - 29 robot instances.
  - SITE-B (Philadelphia) - 29 robot instances.
  - SITE-C (Boston) - 29 robot instances.
  - SITE-D (Texas Med Ctr) - 29 robot instances.
- Center: a vertical Paradigm Health bus with four labeled channels:
  TRAVERSE, STREAM-SCLC, TRAVERSE-PED, NETWORK-OBS.
- Right column: FDA RTCT API node with the median 13 second
  acknowledgement latency annotation.
- Optional overlay arrows from Sim 3 plus Sim 4 sponsor side directly to
  the Paradigm Health bus and onward to the FDA API.
- Header: "RTCT Signal Flow - Sites Plus Sponsors Through Paradigm Health
  to FDA RTCT API."

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- All four channel labels visible without crowding.
- Black text on light fills.

## Suggested Caption

Figure 30: Real-time signal flow from sites and sponsors through Paradigm
Health to the FDA RTCT API across the four TRAVERSE, STREAM-SCLC, TRAVERSE-
PED, and NETWORK-OBS endpoint channels.
