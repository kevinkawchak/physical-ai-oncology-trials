# Patient-Robot Instructions: Physical AI Oncology Trials

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18810541-blue)](https://doi.org/10.5281/zenodo.18810541)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Pages](https://img.shields.io/badge/Pages-10-green.svg)]()

## Overview

This directory documents the **10-page patient-facing instructional PDF** for physical AI oncology clinical trials. Each page is a self-contained instruction sheet for one robot type, designed so that an upcoming patient can visualize, read, and feel comfortable regarding how to correctly interact with a specific robot.

In v2.0.0, the paper, LaTeX source files, and images have been relocated to external hosting to reduce repository size. All materials are accessible via the hyperlinks below.

**Author:** Kevin Kawchak, CEO ChemicalQDevice
**ORCID:** [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)
**Date:** March 2, 2026
**AI Model:** Claude Code Opus 4.6
**DOI:** [10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)

## Paper

The compiled 10-page PDF is available on Zenodo:

> **[Patient-Robot Instructions: Physical AI Oncology Trials (PDF)](https://doi.org/10.5281/zenodo.18810541)**

## LaTeX Source Files

The LaTeX source code (.tex, .sty, .bib, README) is available on Zenodo:

> **[LaTeX Source Code (Zenodo)](https://doi.org/10.5281/zenodo.18810541)**


## Images

All 10 patient-robot instruction images are available on Google Drive:

> **[Patient-Robot Instruction Images (Google Drive)](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax)**

## v2.0.0 Changes

- **Repository size reduction**: @kevinkawchak relocated paper PDFs, LaTeX source files, and images from v1.9.0 and v1.9.1 into Google Drive to reduce repository size
- **Hyperlink-only references**: Paper, LaTeX source files, and images are now referenced via hyperlinks only (no binary files in the repository)
- **Consolidated documentation**: All materials accessible through Zenodo DOI and Google Drive links
- **Major release**: v2.0.0 marks the second major release milestone for the repository

## Prior Versions

- **v1.9.1** (March 1, 2026): New images, streamlined 3-step instructions, corrected URLs. See [`research/v1.9.1/`](research/v1.9.1/) for archived generation scripts.
- **v1.9.0** (February 28, 2026): Original 10-page PDF with Cairo black-and-white illustrations. See [`research/v1.9.0/`](research/v1.9.0/) for archived generation scripts.

## Robot Types (10 Pages)

| Page | Robot Type | Abbreviated Title | Cancer Type | Est. Time |
|------|-----------|-------------------|-------------|-----------|
| 1 | Surgical Robots | Surgical Robots | Prostate cancer | 90--180 min |
| 2 | Cobots (Collaborative Robots) | Cobots | Breast cancer | 10--25 min |
| 3 | Radiotherapy Patient-Positioning Robots | RT Positioning Robots | Lung cancer | 15--30 min |
| 4 | Robotic Needle-Placement Systems | Needle-Placement Robots | Liver cancer | 20--45 min |
| 5 | Social Companion Robots | Companion Robots | Pediatric leukemia | 10--20 min |
| 6 | Humanoids | Humanoids | Pediatric bone cancer | 15--25 min |
| 7 | Radiotherapy Motion-Management / Tracking Robots | RT Motion-Tracking Robots | Pancreatic cancer | 10--20 min |
| 8 | Imaging Assistant Robots | Imaging Robots | Thyroid cancer | 10--20 min |
| 9 | Steerable Needle / Needle-Steering Robots | Steerable Needle Robots | Kidney cancer | 30--60 min |
| 10 | Rehabilitation Exoskeletons / Robotic Gait Trainers | Rehab Exoskeletons | Bone cancer post-surgery | 15--30 min |

## Page Layout

Each page follows a consistent format:

- **(a) Header:** Kevin Kawchak, CEO ChemicalQDevice, ORCID, email (top bar)
- **(b) Title:** "Patient-Robot Instructions: AI Oncology Trials - [Abbreviated Robot Type]" (bar below)
- **(c) Image:** Illustration from Google Drive, occupying the largest portion of the page
- **(d) Dashed bar, full robot type name (centered), bold bar**
- **(e) Instructions:** 1 introductory sentence + 3-item numbered list:
  1. What to do upon entering the room (hands, body position, timing)
  2. What to do during the interaction (quantitative: minutes, forces, distances)
  3. What to do to conclude the session (recovery steps, timing)
- **(f) Footer:** Sources (abbreviated clickable links) + "For Demonstration Purposes Only", date, DOI, Claude Code Opus 4.6, page number

## Directory Structure

```
patients/
├── README.md                   # This file (v2.0.0)
├── research/                   # Archived generation scripts
│   ├── v1.9.1/
│   │   ├── generate_pdf.py     # reportlab + Pillow PDF generator
│   │   ├── paper/README        # Paper access (Drive link)
│   │   └── images/README.md    # Image access (Drive link)
│   └── v1.9.0/
│       ├── README.md           # v1.9.0 overview
│       ├── generate_illustrations.py  # Cairo illustration generator
│       ├── generate_pdf.py     # Cairo-based PDF generator
│       ├── paper/README        # Paper access (Drive link)
│       ├── svg/README.md       # SVG files (Drive link)
│       ├── pdf/README.md       # PDF files (Drive link)
│       └── png/README.md       # PNG files (Drive link)
└── prompts/
    └── prompts.md              # Development prompt archive
```

## References

The bibliography (`references.bib`, available on [Zenodo](https://doi.org/10.5281/zenodo.18810541)) contains 28 references covering:
- Surgical robotics (Intuitive Surgical, Sheetz et al., Moglia et al.)
- Collaborative robots (Franka Robotics, Lamon et al., Haddadin & Croft)
- Radiotherapy positioning (Accuray, Hoisak & Pawlicki, Verellen et al.)
- Needle placement (Walsh et al., Li et al.)
- Companion robots (SoftBank, Logan et al., Rhee et al.)
- Humanoid robots (Boston Dynamics, Darvish et al.)
- Motion tracking (Varian, Bertholet et al., Keall et al.)
- Imaging robots (von Haxthausen et al., Jiang et al.)
- Steerable needles (Cowan et al., Reed et al.)
- Rehabilitation exoskeletons (Molteni et al., Ekso Bionics)
- ISO standards (ISO 15223-1, ISO 20417, ISO 7010)

## License

- **Paper and images:** CC BY 4.0
- **Generation scripts:** MIT (same as repository)
