# National 24/7 Continuous Real-Time Clinical Trial Simulation

Released on 1 May 2026
CEO Kevin Kawchak, ChemicalQDevice

[![Release](https://img.shields.io/badge/Release-v3.6.0-brightgreen.svg)]()
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19194724-blue)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial/national-24-7-trial)
[![Full Paper DOI](https://img.shields.io/badge/Paper_DOI-10.5281%2Fzenodo.19994945-blue)](https://doi.org/10.5281/zenodo.19994945)

Note: Additional national-24-7-trial/extra-hours/ directory available for hours 56-83 approximated diagrams and markdowns. These are excluded from the polished full-paper text per Methods due to extended AI run time during cloud generation.

The polished 70+ page LaTeX manuscript is at `paper/full-paper/` (v3.6.0). The earlier template skeleton with bracketed processing instructions remains untouched at `paper/` (v3.5.0).

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
  Background-A/                     - Deep research chunk set A (v3.4.2)
    chunk_01_baseline_and_short_horizon.md
    chunk_02_multimodal_and_limitations.md
    chunk_03_bibtex.md              - 17 BibTeX references
    README.md                       - Chunk navigation guide
  Background-B/                     - Deep research chunk set B (v3.4.2)
    chunk_01_baseline_and_prediction_domains.md
    chunk_02_response_metrics_conclusions.md
    chunk_03_bibtex_references.md   - 9 BibTeX references
    README.md                       - Chunk navigation guide
  hour-00/                          - Hour 00 (7 files, minute resolution)
  hour-01/                          - Hour 01 (7 files, minute resolution)
  ...                               - Continues through hour-55
  hour-55/                          - Last full minute-resolution hour
  extra-hours/                      - hour-56 through hour-83 (approximated)
  paper/                            - ★ Paper Template (v3.5.0)
    main.tex                        - Document skeleton + global formatting brief
    new_paper.sty                   - Style file (arxiv-derived, CC BY 4.0)
    references.bib                  - 35 entries with DOIs and clickable URLs
    orcid_icon.png                  - Title-page ORCID hyperlink asset
    README.md                       - Paper-template documentation
    LaTeX_Source_Files.zip          - Overleaf-ready archive of all paper files
    sections/
      abstract.tex                  - Bracketed instructions for the abstract
      introduction.tex              - Bracketed instructions for the introduction
      methods.tex                   - Final prose (no instructions)
      results.tex                   - Bracketed instructions for the four sims
      discussion.tex                - Bracketed instructions for the discussion
      limitations_future.tex        - Bracketed instructions for limits + future
      conclusions.tex               - Bracketed instructions for conclusions
      back_matter.tex               - Acknowledgments, ethics, rights, citation
```

## Paper Template (v3.5.0)

The `paper/` directory contains the LaTeX skeleton for the manuscript
"Accelerated Patient Prediction in Physical AI Oncology Clinical Trials:
Four Comprehensive LLM Simulations" (Kawchak K., 2026, DOI:
[10.5281/zenodo.19994945](https://doi.org/10.5281/zenodo.19994945)). The
template carries bracketed processing instructions naming the exact files
to read for each section (FDA-April-2026/, Background-A/, Background-B/,
the four simulation directories under new-trial/, patient-journey/, and
sponsor/final_paper/) and the comprehensive ASCII diagrams to embed
verbatim. The next Claude Code Opus 4.7 Max generation pass will populate
the bracketed instructions into final prose to produce a 70+ page PDF.

| Simulation | Path | Headline output |
|------------|------|-----------------|
| 1. Continuous RTCT | `hour-00/` through `extra-hours/hour-83/` | 84 hrs x 7 files, 4 sites, 116 robots |
| 2. Single-patient journey | `patient-journey/stage_01_*` to `stage_10_*` | 10 stages, 1120 days, PAT-2026-0042 |
| 3. 24-hour sponsor | `sponsor/final_paper/scripts/` | 24 .py + 24 JSON + 75 ASCII + 53 agents |
| 4. 168-hour 7-day | `sponsor/final_paper/168_hours/day_01/` to `day_07/` | 168 .py, 168 JSON, 525 ASCII, 7 PRs |

## Continuous Trial Model

```
+----------------------------------------------------------------+
|                    CONTINUOUS RTCT LOOP                        |
|                                                                |
|   [Patient arrival] -> [Robot orchestration] -> [Procedure]    |
|         ^                                              |       |
|         |                                              v       | 
|   [Real-time     <- [FDA signal stream] <- [Safety signals]    |
|    re-enrollment]    via Paradigm Health                       |
|                      framework                                 |
|                                                                |
|   1-minute resolution | 1-hour commit cadence | indefinite     |
+----------------------------------------------------------------+
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
Clinical Trial Simulation. physical-ai-oncology-trials v3.5.0.
DOI: 10.5281/zenodo.18445179

For the v3.5.0 paper template specifically, cite:
Kawchak, K. (2026). Accelerated Patient Prediction in Physical AI Oncology
Clinical Trials: Four Comprehensive LLM Simulations. Zenodo.
DOI: [10.5281/zenodo.19994945](https://doi.org/10.5281/zenodo.19994945)
