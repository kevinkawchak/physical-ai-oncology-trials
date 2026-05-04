# 12 - Cloud vs Local Compute Trade-offs Chart

## Purpose

Replace the cloud vs local ASCII trade-off block in Section 4.4
(Discussion, Cloud Versus Local Compute Trade-Offs) with a clean
side-by-side comparison chart showing the seven properties and their
relative scores.

## Source Paper Section

`sections/discussion.tex` lines 247 to 261 (the cloud-vs-local verbatim
block).

## Image Properties

- Filename: `images/12_cloud_local_tradeoffs.png`
- DPI: 300
- Size: 10 inches wide by 6 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: cloud blue (#1F4E79), local plus cloud green (#2C7A4D), section
  divider gray (#D8D8D8).

## Layout

- Top header: "Cloud-Only vs Cloud-Plus-Local Verification for RTCT-Aligned
  Oncology Trial AI."
- Seven horizontal rows, one per property. Each row has the property name
  on the left and two parallel pill-shaped cells on the right (cloud-only
  in blue, cloud-plus-local in green) with the assessment text inside.
- Right edge: a small summary card reading "Sims 1, 2, 3 cloud only; Sim 4
  cloud plus local on Core i5-6200U."
- Bottom: a single takeaway sentence "Hybrid cloud plus local is the
  natural endpoint for the FDA RTCT pilot."

## Property Data

| Property             | Cloud Compute Only       | Cloud + Local Verification    |
| -------------------- | ------------------------ | ----------------------------- |
| Wall-clock speed     | Highest                  | Cloud high, local slower      |
| 1M token context     | Yes (Claude Opus)        | Yes (cloud) / partial (local) |
| Reproducibility      | Cloud commit log         | Cloud + local rerun parity    |
| Hardware floor       | Data-center scale        | i5-6200U / 4 GB demonstrated  |
| PHI / data security  | Cloud transit risk       | Local processing + cloud meta |
| Audit trail          | GitHub commits           | GitHub commits + local hash   |
| Site adoption cost   | Cloud subscription       | Off-the-shelf laptop          |

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.

## Suggested Caption

Figure 12: Cloud-only versus cloud-plus-local-verification trade-offs across
seven operational properties.
