# Physical AI Oncology Clinical Trial Site Documentation

**Version**: 2.9.0
**Last Updated**: March 2026

## Overview

Comprehensive documentation for establishing the first California Physical AI
oncology clinical trial site in a new building with a new parking lot in a
prominent and safe San Francisco location. These 11 LaTeX documents collectively
provide legislation drafts, regulatory updates, and building and premises code
standards required for site authorization and operation.

## Evidence Base

All documents are grounded in the 24-hour on-demand Physical AI oncology
clinical trial simulation (March 23, 2026) demonstrating:

- 168 unique patients across 15 cancer types in 24 hours
- 10 robot types, 29 instances, 99.7% fleet uptime
- 7 adverse events (all Grade 1-2, all resolved same hour)
- Average wait time: 8 minutes; satisfaction: 4.7/5.0
- Staffing: 8-10 FTE vs. 80-120 traditional
- Per-patient cost reduction: 75-85%
- Both minute-level detail and 24-hour operations handled correctly by AI

## Documents

| No. | Document | Type | Description |
|-----|----------|------|-------------|
| 01 | SB 1042 | Legislation | Trial authorization and site establishment act |
| 02 | AB 2847 | Legislation | Patient rights and robotic safety act |
| 03 | SB 892 | Legislation | Clinical data protection and transparency act |
| 04 | SF Municipal | City Regulation | Zoning, permits, health code for SF site |
| 05 | Title 22 Ch.14 | State Regulation | CDPH authorization, inspection, enforcement |
| 06 | FDA Guide | National Regulation | Federal compliance guide with correction map |
| 07 | Building Code | Building Code | Structural, MEP, technology infrastructure |
| 08 | Premises Code | Premises Code | Security, access, robot zones, waste management |
| 09 | Parking | Premises Code | Parking facility and transportation standards |
| 10 | Site Operations | Operations | Activation checklist and SOPs |
| 11 | Emergency Plan | Operations | Emergency response and business continuity |

## Regulatory Foundations

These documents implement and extend three adapted regulatory frameworks:

- **ICH E6(R3)** adaptation (DOI: 10.5281/zenodo.18973368)
- **21 CFR Part 50** adaptation (DOI: 10.5281/zenodo.19040707)
- **21 CFR Part 312** adaptation (DOI: 10.5281/zenodo.19057628)

Scoring frameworks applied:

- **PSL** (Physical AI Standard Level): Three regulatory dimensions, 0-100 site score
- **USL** (Unification Standard Level): Four dimensions, 1.0-10.0 robot score

## Directory Structure

```
site/
├── README.md                        # This file
├── 01-legislation-authorization/    # SB 1042
│   ├── physical_ai_trial_authorization.tex
│   ├── physical_ai_trial_authorization.bib
│   ├── physical_ai_legislation.sty
│   └── README.md
├── 02-legislation-patient-rights/   # AB 2847
│   ├── physical_ai_patient_rights.tex
│   ├── physical_ai_patient_rights.bib
│   ├── physical_ai_legislation.sty
│   └── README.md
├── 03-legislation-data-transparency/ # SB 892
│   ├── physical_ai_data_transparency.tex
│   ├── physical_ai_data_transparency.bib
│   ├── physical_ai_legislation.sty
│   └── README.md
├── 04-city-regulations/             # SF Municipal Code
│   ├── sf_city_regulations.tex
│   ├── sf_city_regulations.bib
│   ├── physical_ai_legislation.sty
│   └── README.md
├── 05-state-regulations/            # CA Title 22
│   ├── ca_state_regulations.tex
│   ├── ca_state_regulations.bib
│   ├── physical_ai_legislation.sty
│   └── README.md
├── 06-national-regulations/         # FDA Compliance Guide
│   ├── fda_national_regulations.tex
│   ├── fda_national_regulations.bib
│   ├── physical_ai_legislation.sty
│   └── README.md
├── 07-building-code/                # Building Code
│   ├── physical_ai_building_code.tex
│   ├── physical_ai_building_code.bib
│   ├── physical_ai_legislation.sty
│   └── README.md
├── 08-premises-code/                # Premises Code
│   ├── physical_ai_premises_code.tex
│   ├── physical_ai_premises_code.bib
│   ├── physical_ai_legislation.sty
│   └── README.md
├── 09-parking-transportation/       # Parking Standards
│   ├── physical_ai_parking.tex
│   ├── physical_ai_parking.bib
│   ├── physical_ai_legislation.sty
│   └── README.md
├── 10-site-operations/              # Site Operations
│   ├── physical_ai_site_operations.tex
│   ├── physical_ai_site_operations.bib
│   ├── physical_ai_legislation.sty
│   └── README.md
├── 11-emergency-preparedness/       # Emergency Plan
│   ├── physical_ai_emergency.tex
│   ├── physical_ai_emergency.bib
│   ├── physical_ai_legislation.sty
│   └── README.md
└── zips/                            # LaTeX source archives
    ├── 01-legislation-authorization.zip
    ├── 02-legislation-patient-rights.zip
    ├── 03-legislation-data-transparency.zip
    ├── 04-city-regulations.zip
    ├── 05-state-regulations.zip
    ├── 06-national-regulations.zip
    ├── 07-building-code.zip
    ├── 08-premises-code.zip
    ├── 09-parking-transportation.zip
    ├── 10-site-operations.zip
    ├── 11-emergency-preparedness.zip
    └── all-documents-combined.zip
```

## Compilation

Each document can be compiled independently:

```bash
cd <document-directory>
pdflatex <document>.tex
biber <document>
pdflatex <document>.tex
pdflatex <document>.tex
```

## Author

Kevin Kawchak, CEO, ChemicalQDevice (kevink@chemicalqdevice.com)

## Disclaimer

These are independent drafts not endorsed, sponsored, or approved by any trial
sponsor, CRO, site, IRB, regulator, or medical society. Adapted using Claude
Code Opus 4.6. CFR-derived content is from public domain documents. ICH content
is copyrighted and may be used under a public license.
