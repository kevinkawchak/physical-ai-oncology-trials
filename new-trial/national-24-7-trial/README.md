# National 24/7 Continuous Real-Time Clinical Trial Simulation

Released on 1 May 2026
CEO Kevin Kawchak, ChemicalQDevice

[![Release](https://img.shields.io/badge/Release-v3.4.2-brightgreen.svg)]()
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19194724-blue)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial/national-24-7-trial)

The original CFR documents are in the public domain. The original ICH document
is copyrighted and may be used, reproduced, incorporated into other works,
adapted, modified, translated or distributed under a public license. This
current work is not endorsed or sponsored by CFR, ICH, or FDA; and was adapted
using Claude Code Opus 4.7.

## Overview

This directory contains a **continuous, real-time, never-ending Physical AI
oncology clinical trial simulation** built in direct response to the FDA's
April 28, 2026 announcement on Real-Time Clinical Trials (RTCT) and the
agency's stated path toward continuous trials. Unlike the prior 24-hour
on-demand simulation in `new-trial/`, this simulation:

- Runs **indefinitely**, hour by hour, with no pre-defined stop
- Outputs **real-time commits** to GitHub on a 1-hour cadence (24 commits/day)
- Maintains **minute-level resolution** for every hour generated
- Streams safety signals and endpoints in a format compatible with the FDA's
  RTCT signal-sharing framework demonstrated by AstraZeneca / Paradigm Health

When the user runs out of tokens, the simulation pauses; it can be resumed by
appending the next hour using the same 7-file-per-hour format.

## FDA Source

See `FDA-April-2026/FDA_RealTime_Clinical_Trials.md` for the FDA news release
that motivated this simulation. Key elements implemented here:

- Real-time signal reporting (vs. retrospective batch reporting)
- Continuous trials (no hiatus between phases)
- Safety endpoint streaming with FDA-validated technical framework
- Multi-site readiness (national platform integration)

## Format Per Hour (7 Files)

This format **exactly matches** the file count and structure of
`new-trial/hour-XX/`:

| # | File | Type | Purpose |
|---|------|------|---------|
| 1 | `hour_XX_simulation.md` | markdown | Master simulation log, minute-by-minute |
| 2 | `hour_XX_robot_logs.md` | markdown | Per-robot telemetry across 10 robot types |
| 3 | `hour_XX_patient_records.md` | markdown | Patient vitals, arrivals, departures |
| 4 | `hour_XX_psl_scores.md` | markdown | PSL scoring snapshot for the hour |
| 5 | `hour_XX_diagram_facility.txt` | txt diagram | Facility status ASCII layout |
| 6 | `hour_XX_diagram_patient_flow.txt` | txt diagram | Patient flow timeline |
| 7 | `hour_XX_diagram_robot_status.txt` | txt diagram | Robot status timeline |

When the simulation is terminated by the user, a `final-commit/` folder with
the same 6-file termination summary is generated:
`final_24h_summary.md`, `final_error_review.md`, `final_psl_cumulative.md`,
`final_diagram_facility.txt`, `final_diagram_patient_flow.txt`,
`final_diagram_robot_status.txt`.

## Directory Structure

```
new-trial/national-24-7-trial/
  README.md                         - This file
  FDA-April-2026/
    FDA_RealTime_Clinical_Trials.md - Source FDA news release (28 Apr 2026)
    README.md                       - FDA folder description
  hour-00/                          - Hour 00 (7 files, minute resolution)
  hour-01/                          - Hour 01 (7 files, minute resolution)
  ...                               - Continues indefinitely
  hour-NN/                          - Last hour generated before token exhaust
  final-commit/                     - Generated only at user-initiated stop
```

## Continuous Trial Model

```
+--------------------------------------------------------------+
|                    CONTINUOUS RTCT LOOP                      |
|                                                              |
|   [Patient arrival] -> [Robot orchestration] -> [Procedure]  |
|         ^                                              |     |
|         |                                              v     |
|   [Real-time     <- [FDA signal stream] <- [Safety signals]  |
|    re-enrollment]    via Paradigm Health                     |
|                      framework                               |
|                                                              |
|   1-minute resolution | 1-hour commit cadence | indefinite   |
+--------------------------------------------------------------+
```

## Real-Time Commit Cadence

- **24 commits per day** (1 commit per simulated hour)
- Each commit contains exactly 7 new files in `hour-XX/`
- Commits are made on branch `claude/add-fda-clinical-trial-CVStP`
- Simulation hours are ordered: hour-00, hour-01, ... hour-99, hour-100, ...
  After hour-23 the count continues (hour-24, hour-25, ...) reflecting the
  cumulative continuous nature of the trial.

## Sites and Robots

This simulation extends the 10-robot, 29-instance facility from `new-trial/`
to a **national multi-site network**. Site identifiers:

| Site | Location | Role |
|------|----------|------|
| SITE-A | Houston (MD Anderson partnership analog) | Primary, mantle cell lymphoma analog |
| SITE-B | Philadelphia (UPenn analog) | Secondary, mantle cell lymphoma analog |
| SITE-C | Boston | Lead site for STREAM-SCLC analog (limited-stage SCLC) |
| SITE-D | Houston/Texas Medical Center | Pediatric oncology continuous trial |

All sites stream signals to a central `Paradigm Health`-style aggregator,
which validates and pushes endpoints to FDA in real time.

## PSL Framework

PSL (Physical AI Standard Level) is unchanged from `new-trial/psl_framework.md`:

- Dimension A (Omniscient): ICH E6(R3) - data governance, complete knowledge
- Dimension B (Omnipresent): 21 CFR Part 50 - informed consent, ubiquitous presence
- Dimension C (Omnipotent): 21 CFR Part 312 - IND, complete capability

For continuous trials, PSL is computed every hour and trended across hours.
A new derived metric, **Continuity-PSL (C-PSL)**, equals the rolling 24-hour
mean of cumulative site PSL.

## Regulatory Foundation

- ICH E6(R3) Adaption (DOI: 10.5281/zenodo.18973368)
- 21 CFR Part 50 Adaption (DOI: 10.5281/zenodo.19040707)
- 21 CFR Part 312 Adaption (DOI: 10.5281/zenodo.19057628)
- FDA RTCT announcement, 28 April 2026 (see FDA-April-2026/)

## Citation

Kawchak, K. (2026). National 24/7 Continuous Real-Time Physical AI Oncology
Clinical Trial Simulation. physical-ai-oncology-trials v3.4.2.
DOI: 10.5281/zenodo.18445179
