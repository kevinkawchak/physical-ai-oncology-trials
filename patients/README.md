# Patient-Robot Instructions: Physical AI Oncology Trials

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18810541-blue)](https://doi.org/10.5281/zenodo.18810541)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Pages](https://img.shields.io/badge/Pages-10-green.svg)]()

## Overview

This directory contains a **10-page patient-facing instructional PDF** for physical AI oncology clinical trials. Each page is a self-contained instruction sheet for one robot type, designed so that an upcoming patient can visualize, read, and feel comfortable regarding how to correctly interact with a specific robot. The v1.9.1 update introduces new images, a streamlined 3-step instruction format with quantitative interaction data, corrected URLs, and a reorganized file structure.

**Author:** Kevin Kawchak, CEO ChemicalQDevice
**ORCID:** [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)
**Date:** March 1, 2026
**AI Model:** Claude Code Opus 4.6
**DOI:** [10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)

## What Changed in v1.9.1

- **New images** from [Google Drive](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax) numbered 1--10, occupying the largest portion of each page
- **Streamlined instructions**: 1 introductory sentence + 3-item numbered list (entering, interacting, concluding) with quantitative data (minutes, mm, N)
- **Title updated** to "Patient-Robot Instructions: AI Oncology Trials - [Robot Type]" with abbreviated names when needed
- **Fixed all URLs** in bibliography and source links (corrected domains for Intuitive, Franka, Accuray, SoftBank, Boston Dynamics, Varian, Ekso Bionics)
- **Single DOI** (10.5281/zenodo.18810541) throughout
- **"For Demonstration Purposes Only"** added to each page
- **Abbreviated clickable source links** (e.g., "Intuitive Surgical" instead of full URL)
- **v1.9.0 materials** moved to `patients/research/` for archival
- **Three PDF versions**: full-size, 10 MB, 5 MB
- **Each robot paired with a specific cancer type** for patient context

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

## Page Layout (v1.9.1)

Each page follows a consistent format:

- **(a) Header:** Kevin Kawchak, CEO ChemicalQDevice, ORCID, email (top bar)
- **(b) Title:** "Patient-Robot Instructions: AI Oncology Trials - [Abbreviated Robot Type]" (bar below)
- **(c) Image:** New illustration from Google Drive, occupying the largest portion of the page
- **(d) Dashed bar, full robot type name (centered), bold bar**
- **(e) Instructions:** 1 introductory sentence + 3-item numbered list:
  1. What to do upon entering the room (hands, body position, timing)
  2. What to do during the interaction (quantitative: minutes, forces, distances)
  3. What to do to conclude the session (recovery steps, timing)
- **(f) Footer:** Sources (abbreviated clickable links) + "For Demonstration Purposes Only", date, DOI, Claude Code Opus 4.6, page number

## Directory Structure

```
patients/
├── README.md                   # This file
├── generate_pdf.py             # PDF generator (reportlab + Pillow)
│
├── paper/                      # ★ Main paper output (v1.9.1)
│   ├── Patient-Robot Instructions: Physical AI Oncology Trials.pdf
│   ├── Patient-Robot Instructions: Physical AI Oncology Trials (10MB).pdf
│   ├── Patient-Robot Instructions: Physical AI Oncology Trials (5MB).pdf
│   ├── Latex Source Code.zip   # .tex, .sty, .bib, README
│   ├── patient_robot_instructions.tex
│   ├── patient_robot_instructions.sty
│   ├── references.bib          # BibTeX bibliography (28 references)
│   └── README
│
├── images/                     # ★ New images (v1.9.1)
│   ├── README.md               # Image access and Google Drive link
│   ├── 1.png ... 10.png        # Numbered images for each page
│
├── research/                   # ★ Archived v1.9.0 materials
│   └── v1.9.0/
│       ├── README.md
│       ├── generate_illustrations.py
│       ├── generate_pdf.py
│       ├── paper/              # Original v1.9.0 paper files
│       ├── svg/                # 10 SVG vector illustrations
│       ├── pdf/                # 10 PDF vector illustrations
│       └── png/                # 10 PNG raster illustrations
│
└── prompts/
    └── prompts.md              # Development prompt archive
```

## Regenerating the PDF

```bash
# Install dependencies
pip install reportlab Pillow

# Generate all three PDF versions
python3 patients/generate_pdf.py
```

## References

The bibliography (`references.bib`) contains 28 references covering:
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
