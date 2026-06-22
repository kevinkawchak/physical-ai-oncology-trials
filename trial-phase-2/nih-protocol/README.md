# nih-protocol - NIH-FDA Phase 2/3 IND/IDE template grounding (v1.1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Template](https://img.shields.io/badge/Template-NIH--FDA%20Phase%202%2F3%20IND%2FIDE-800020.svg)](../../trial-protocol/nih-protocol)
[![Sections](https://img.shields.io/badge/NIH%20sections-13-800020.svg)](../final-protocol/publication/sections)
[![Grounding](https://img.shields.io/badge/Grounding-canonical%20in%20trial--protocol-6B6B6B.svg)](../../trial-protocol/nih-protocol)

This directory grounds the Phase II protocol in the **NIH-FDA Phase 2 and 3 IND/IDE
Clinical Trial Protocol Template (Version 1.0, 7 April 2017)**. The canonical,
word-for-word 10-chunk copy of that template lives in the Phase I tree at
[`../../trial-protocol/nih-protocol`](../../trial-protocol/nih-protocol) and is not
duplicated here; this README maps the template onto the Phase II build.

## Why the Phase 2/3 template now applies directly

The NIH-FDA template is written for **Phase 2 and 3** trials under an FDA IND
(21 CFR part 312) or IDE (21 CFR part 812), aligned to ICH E6 Good Clinical
Practice. The Phase I protocol adapted it for a single-arm first-in-human study;
this Phase II protocol is the randomized controlled efficacy study the template was
designed for, so the fit is now direct rather than adapted.

## Template section to Phase II file map

| Template section | Phase II file |
|:--|:--|
| Statement of Compliance | [`sec-00-compliance.tex`](../final-protocol/publication/sections/sec-00-compliance.tex) |
| 1 Protocol Summary (Synopsis, Schema, SoA) | [`sec-01-summary.tex`](../final-protocol/publication/sections/sec-01-summary.tex) |
| 2 Introduction (Rationale, Background, Risk/Benefit) | [`sec-02-introduction.tex`](../final-protocol/publication/sections/sec-02-introduction.tex) |
| 3 Objectives and Endpoints | [`sec-03-objectives.tex`](../final-protocol/publication/sections/sec-03-objectives.tex) |
| 4 Study Design | [`sec-04-design.tex`](../final-protocol/publication/sections/sec-04-design.tex) |
| 5 Study Population | [`sec-05-population.tex`](../final-protocol/publication/sections/sec-05-population.tex) |
| 6 Study Intervention | [`sec-06-intervention.tex`](../final-protocol/publication/sections/sec-06-intervention.tex) |
| 7 Intervention/Participant Discontinuation | [`sec-07-discontinuation.tex`](../final-protocol/publication/sections/sec-07-discontinuation.tex) |
| 8 Study Assessments and Procedures | [`sec-08-assessments.tex`](../final-protocol/publication/sections/sec-08-assessments.tex) |
| 9 Statistical Considerations | [`sec-09-statistics.tex`](../final-protocol/publication/sections/sec-09-statistics.tex) |
| 10.1 Regulatory, Ethical, Oversight | [`sec-10-oversight.tex`](../final-protocol/publication/sections/sec-10-oversight.tex) |
| 10.2-10.4 Additional, Abbreviations, Amendments | [`sec-11-additional.tex`](../final-protocol/publication/sections/sec-11-additional.tex) |
| 11 References (+ back matter) | [`sec-12-references-backmatter.tex`](../final-protocol/publication/sections/sec-12-references-backmatter.tex) |

## What Phase II adds within the template structure

Within the same NIH section order, the Phase II build adds the randomized 1:1
multicenter design (the template's randomization and blinding, sample-size, and
analysis-population machinery, which the single-arm Phase I left minimal), the
confirmatory progression-free-survival primary endpoint with a fixed-sequence key
secondary hierarchy and a group-sequential interim, the single IRB of record
(45 CFR 46.114), and the Patient-Aligned Co-Investment Facility with its capital
firewall in the oversight and conflict-of-interest sections.

## Files from other directories used here

| Source | Used for |
|:--|:--|
| [`../../trial-protocol/nih-protocol`](../../trial-protocol/nih-protocol) | the canonical 10-chunk NIH-FDA Phase 2/3 template (section order, required content) |
| [`../final-protocol/publication/sections`](../final-protocol/publication/sections) | the 13 Phase II sections authored to that order |

## License

Reproduced NIH-FDA template text is U.S. Government work under 17 U.S.C. 105.
This README is released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice.
