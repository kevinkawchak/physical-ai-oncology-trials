# sections - Stage 4 publication (author-edited, paper URL directory) (v1.1.0)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Sections](https://img.shields.io/badge/NIH%20sections-13-800020.svg)](.)
[![Figures](https://img.shields.io/badge/TikZ%20figures-22-6B6B6B.svg)](.)
[![Tables](https://img.shields.io/badge/Full--width%20tables-11-6B6B6B.svg)](.)

The 13 NIH-FDA Phase 2/3 IND/IDE sections, one `.tex` per section, each `\input`
by the stage `main.tex`. Figure numbers (1 to 22) and table labels are global and
identical across the draft, full, final, and publication stages.

| File | NIH section | Figures | Tables |
|:--|:--|:--|:--|
| `sec-00-compliance.tex` | Statement of Compliance | 1 | |
| `sec-01-summary.tex` | Protocol Summary (Synopsis, Schema, SoA) | 2 | tab:soa |
| `sec-02-introduction.tex` | Introduction (Rationale, Background, Risk/Benefit, Co-Investment) | 3, 4, 5, 6 | tab:concerns, tab:coinvest |
| `sec-03-objectives.tex` | Objectives and Endpoints | 7 | tab:objend |
| `sec-04-design.tex` | Study Design (randomization, multicenter) | 8, 9 | |
| `sec-05-population.tex` | Study Population (CONSORT, equity fund) | 10 | |
| `sec-06-intervention.tex` | Study Intervention (both arms, device) | 11, 12, 13, 14 | tab:arms, tab:sensors |
| `sec-07-discontinuation.tex` | Intervention/Participant Discontinuation | | |
| `sec-08-assessments.tex` | Study Assessments and Procedures | 15, 16, 17 | |
| `sec-09-statistics.tex` | Statistical Considerations (confirmatory) | 18 | tab:power, tab:secendpts |
| `sec-10-oversight.tex` | Regulatory, Ethical, Oversight (capital firewall) | 19, 20, 21 | |
| `sec-11-additional.tex` | Additional, Abbreviations, Amendments | 22 | tab:jurisdictions, tab:amend, tab:abbrev |
| `sec-12-references-backmatter.tex` | References and Back Matter | | |

Source: adapted from the Phase 1 `trial-protocol/.../sections` and the
`trial-phase-2/mermaid` figure catalog. License CC BY 4.0; Kevin Kawchak, ChemicalQDevice.
