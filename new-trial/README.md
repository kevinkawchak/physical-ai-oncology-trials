# 24-Hour On-Demand Physical AI Oncology Clinical Trial Simulation

Released on 23 March 2026
CEO Kevin Kawchak, ChemicalQDevice

[![Release](https://img.shields.io/badge/Release-v2.8.0-brightgreen.svg)]()

The original CFR documents are in the public domain. The original ICH document
is copyrighted and may be used, reproduced, incorporated into other works,
adapted, modified, translated or distributed under a public license. This
current work is not endorsed or sponsored by CFR, ICH, or FDA; and was adapted
using Claude Code Opus 4.6.

## Overview

This directory contains a complete 24-hour simulation of an on-demand,
patient-centric Physical AI oncology clinical trial at a single site. The
simulation runs with 1-minute resolution (1,440 total minutes) and
demonstrates how 10 robot types serve 150-180 patients across 15 cancer
types in a fully autonomous, 24/7 facility.

The simulation introduces the Physical AI Standard Level (PSL) framework,
which evaluates each robot type on three regulatory dimensions: Omniscient
(ICH E6(R3)), Omnipresent (21 CFR Part 50), and Omnipotent (21 CFR Part 312).

## Key Results

- Total patients served: 168 unique patients across 24 hours
- Cancer types treated: 15 simultaneously
- Robot types active: 10 (29 total robot instances)
- Cumulative site PSL: 63.4 to 64.8 (Advanced Site)
- Peak throughput: Hour 09 with 15 new arrivals
- Average wait time: 12 minutes (arrival to procedure start)
- Adverse events: 7 total (4.2% rate), all managed successfully
- Facility uptime: 99.7% across all robot types

## Regulatory Foundation

Three adapted regulatory frameworks govern this simulation:

- (a) ICH E6(R3) Adaption (DOI: 10.5281/zenodo.18973368) - System
  qualification, data governance, monitoring, and essential records
- (b) 21 CFR Part 50 Adaption (DOI: 10.5281/zenodo.19040707) - Informed
  consent, pediatric safeguards, and pre-procedure safety matrix
- (c) 21 CFR Part 312 Adaption (DOI: 10.5281/zenodo.19057628) - IND
  requirements, safety reporting, and expanded access

## Complementary References

- USL Framework (Kawchak, 2026; DOI: 10.5281/zenodo.18778220) - Robot
  technical interoperability scoring across 4 dimensions
- Patient Journey (Kawchak, 2026; DOI: 10.5281/zenodo.19119939) - Single-
  patient autonomous trial journey demonstration

## Directory Structure

```
new-trial/
  README.md                    - This file
  psl_framework.md             - PSL scoring framework definition
  site_specification.md        - Facility and staffing specifications
  format_comparison.md         - On-demand vs. traditional trial comparison
  prompts.md                   - v2.8.0 development prompt
  hour-00/                     - Hour 00: 00:00-00:59 (low volume, overnight)
    hour_00_simulation.md      - Master simulation log
    hour_00_robot_logs.md      - Per-robot telemetry and status
    hour_00_patient_records.md - Patient vitals and records
    hour_00_psl_scores.md      - PSL scores for all 10 robot types
    hour_00_diagram_facility.txt    - Facility layout diagram
    hour_00_diagram_patient_flow.txt - Patient flow diagram
    hour_00_diagram_robot_status.txt - Robot status timeline
  hour-01/ through hour-23/   - Same 7 files per hour
  final-commit/                - Error review and 24-hour summaries
    final_error_review.md      - Consistency check across all hours
    final_24h_summary.md       - Complete 24-hour performance summary
    final_psl_cumulative.md    - PSL trajectory and analysis
    final_diagram_facility.txt - End-of-day facility status
    final_diagram_patient_flow.txt  - 24-hour patient flow summary
    final_diagram_robot_status.txt  - 24-hour robot utilization heatmap
```

## The 10 Robot Types

| # | Robot Type | Instances | Cancer Types |
|---|-----------|-----------|-------------|
| 1 | Surgical Robots | 3 suites | Mediastinal tumors |
| 2 | Cobots | 4 stations | Soft-tissue sarcoma |
| 3 | RT Positioning Robots | 3 vaults | Brain tumors |
| 4 | Needle-Placement Systems | 2 suites | Parotid tumors |
| 5 | Social Companion Robots | 5 stations | Pediatric leukemia |
| 6 | Humanoids | 3 stations | Pediatric osteosarcoma |
| 7 | RT Motion-Tracking Robots | 3 vaults | Lung tumors |
| 8 | Imaging Assistant Robots | 4 bays | Liver tumors |
| 9 | Steerable Needle Robots | 2 suites | Liver ablation |
| 10 | Rehab Exoskeletons | 3 bays | Femur osteosarcoma |

## PSL Framework Summary

PSL scores range from 0.0 to 10.0 per robot across three dimensions:

- Dimension A (Omniscient): Based on ICH E6(R3) - complete knowledge
- Dimension B (Omnipresent): Based on 21 CFR Part 50 - ubiquitous presence
- Dimension C (Omnipotent): Based on 21 CFR Part 312 - complete capability

Cumulative Site PSL ranges from 0.0 to 100.0 (sum of all 10 robot PSL scores).

See psl_framework.md for complete framework definition.

## Patient Population

- Total 24-hour patients: 168
- Patient ID format: PAT-ODMND-XXXX
- Age range: 4-78 years
- Cancer types: 15 simultaneously
- Scheduling: Patient-chosen, real-time booking
- Peak hours: 08:00-12:00 and 16:00-19:00

## Citation

Kawchak, K. (2026). 24-Hour On-Demand Physical AI Oncology Clinical Trial
Simulation. physical-ai-oncology-trials v2.8.0.
DOI: 10.5281/zenodo.18445179
