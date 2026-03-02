# Patient-Robot Instructions: Physical AI Oncology Trials


[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Pages](https://img.shields.io/badge/Pages-10-green.svg)]()

## Overview

This directory contains a **10-page patient-facing instructional PDF** with professional black-and-white portrait illustrations. Each page is a self-contained instructional sheet for one robot type used in physical AI oncology clinical trials. The document is designed so that an upcoming patient can visualize, read, and feel comfortable regarding how to correctly interact with a specific type of robot for their upcoming trial.

**Author:** Kevin Kawchak, CEO ChemicalQDevice
**ORCID:** [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)
**Date:** February 28, 2026
**AI Model:** Claude Code Opus 4.6


## Robot Types (10 Pages)

Ordered by frequency of use in physical AI oncology trials:

| Page | Robot Type | Patient | Setting |
|------|-----------|---------|---------|
| 1 | **Surgical Robots** | Adult | Operating room |
| 2 | **Cobots** (Collaborative Robots) | Adult | Exam/procedure room |
| 3 | **Radiotherapy Patient-Positioning Robots** | Adult | Treatment vault |
| 4 | **Robotic Needle-Placement Systems** | Adult | Interventional suite |
| 5 | **Social Companion Robots** | **Pediatric** | Activity room |
| 6 | **Humanoids** | **Pediatric** | Activity room |
| 7 | **Radiotherapy Motion-Management / Tracking Robots** | Adult | Treatment vault |
| 8 | **Imaging Assistant Robots** | Adult | Imaging room |
| 9 | **Steerable Needle / Needle-Steering Robots** | Adult | Interventional suite |
| 10 | **Rehabilitation Exoskeletons / Robotic Gait Trainers** | Adult | Rehabilitation gym |

**Selection:** Top 10 selected from 13 candidates based on relevance to physical AI oncology trials. Must-include categories: Cobots, Surgical Robots, Humanoids. Excluded: Telepresence robots, Autonomous hospital transport robots (AMRs), UV disinfection robots (limited direct patient interaction).

**Pediatric Pages:** Pages 5 and 6 feature children of suitable age for pediatric oncology trials, matched to robots appropriate for their size (Social Companion Robots and Humanoids).

## Page Layout

Each page follows a consistent format:

- **(a) Header:** Kevin Kawchak, CEO ChemicalQDevice, ORCID, email
- **(b) Title:** "Patient-Robot Instructions: Physical AI Oncology Trials - [Robot Type]"
- **(c) Illustration:** Prominent black-and-white portrait image showing one patient and one robot interacting in the most likely clinical scenario
- **(d) Instructions:** Five numbered sections with bullet points:
  1. Preparation at Home
  2. Entering the Room (what to do when alone with the robot)
  3. During the Interaction (with quantitative data: minutes, distances, forces)
  4. Concluding the Session
  5. At Home & Follow-Up
- **(e) Footer:** Date, DOI, Claude Code Opus 4.6, page number, truncated source links

## Patient Diversity

Illustrations depict diverse patients across the 10 pages through varied:
- Hair styles (short, long, curly/afro, bald/stubble, headscarf, textured, ponytail, pigtails, short messy)
- Body types and postures
- Age groups (8 adults, 2 pediatric)

## ISO Standards Referenced

Symbols and safety pictograms follow established international standards:

| Standard | Description | Usage |
|----------|-------------|-------|
| **ISO 15223-1:2021** | Medical device symbols | Keep-still indicator on non-invasive pages |
| **ISO 20417:2021** | Medical device information | Information labeling guidance |
| **ISO 7000:2019** | Graphical symbols for equipment | Equipment status indicators |
| **IEC 60417:2023** | Graphical symbols for equipment | Equipment interface symbols |
| **ISO 7010:2019** | Safety signs (W003 radiation) | Radiotherapy pages |
| **ISO 3864-1:2011** | Safety colours and safety signs | Warning triangle on surgical/needle pages |

## Illustration Details

Each illustration is rendered using Python Cairo for high-quality vector graphics:

- **Style:** Professional black-and-white line art (black lines on white background)
- **Content:** One human patient + one robot per page (no doctors or nurses)
- **Robots:** Each robot is drawn to be universally recognizable as that specific type, without manufacturer logos
- **Patients:** Diverse in appearance; pediatric patients on pages 5--6 are appropriately sized relative to their robots
- **Resolution:** SVG/PDF (vector, resolution-independent), PNG (3600×4000 pixels, 2× scale)
- **Consistency:** All illustrations share the same border style, title placement, floor line, label position, and ISO symbol placement

## Regenerating Files

```bash
# Generate individual illustrations (SVG, PDF, PNG)
python3 patients/generate_illustrations.py

# Generate combined 10-page PDF
python3 patients/generate_pdf.py
```

**Dependencies:** `pycairo` (Cairo bindings for Python)

```bash
pip install pycairo
```

## References

The bibliography (`references.bib`) contains 35 references covering:
- Surgical robotics and robot-assisted surgery trends
- Collaborative robots in biomedical applications
- Radiotherapy patient positioning and motion management
- Robotic needle placement and steerable needles
- Social companion robots in pediatric settings
- Humanoid robot locomotion and interaction
- Robotic ultrasound and imaging assistance
- Rehabilitation exoskeletons
- ISO/IEC standards for medical device symbols and safety signs
- Physical AI oncology trials repository and USL framework

## License

- **Paper and illustrations:** CC BY 4.0
- **Generation scripts:** MIT (same as repository)
