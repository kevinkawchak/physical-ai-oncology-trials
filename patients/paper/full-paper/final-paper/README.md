# Patient Priority and Proposed U.S. Bills for Physical AI Oncology Clinical Trials - FULL PAPER

**v3.8.0 (Patient Priority Full Paper)** Polished 70+ page LaTeX manuscript at `patients/paper/full-paper/` populating the seven proposed federal bills (HR 9501 through HR 9507) with consolidated bracketed instructions and named repository file paths from the prior v3.7.0 template.

[![Paper DOI](https://img.shields.io/badge/Paper%20DOI-10.5281%2Fzenodo.20045457-blue)](https://doi.org/10.5281/zenodo.20045457)
[![Repo DOI](https://img.shields.io/badge/Repo%20DOI-10.5281%2Fzenodo.18445179-blue)](https://doi.org/10.5281/zenodo.18445179)
[![National Platform DOI](https://img.shields.io/badge/National%20Platform-10.5281%2Fzenodo.19244918-blue)](https://doi.org/10.5281/zenodo.19244918)
[![Sponsor DOI](https://img.shields.io/badge/Sponsor-10.5281%2Fzenodo.19396256-blue)](https://doi.org/10.5281/zenodo.19396256)
[![Site DOI](https://img.shields.io/badge/Site-10.5281%2Fzenodo.19176370-blue)](https://doi.org/10.5281/zenodo.19176370)
[![Patient Journey DOI](https://img.shields.io/badge/Patient%20Journey-10.5281%2Fzenodo.19119939-blue)](https://doi.org/10.5281/zenodo.19119939)
[![Patient Instructions DOI](https://img.shields.io/badge/Patient%20Instructions-10.5281%2Fzenodo.18810541-blue)](https://doi.org/10.5281/zenodo.18810541)
[![Accelerated Patient Prediction DOI](https://img.shields.io/badge/4%20Simulations-10.5281%2Fzenodo.19994945-blue)](https://doi.org/10.5281/zenodo.19994945)

## Overview

This is the polished LaTeX paper for the manuscript:

> Kawchak K. *Patient Priority and Proposed U.S. Bills for Physical AI
> Oncology Clinical Trials*. Zenodo. 2026; 10.5281/zenodo.20045457.

The full paper is a consolidated 7-Bill structure derived from the
v3.7.0 template at `patients/paper/`. All bracketed processing
instructions from the prior template have been MOVED into the seven bill section
files. The result is a tight 7-Bill paper layout (Title page + TOC,
seven bills, References, Acknowledgments, Ethical disclosures, Rights
and permissions, Cite this article) suitable for downstream Claude
Code Opus 4.7 1M Max population into final prose.

## Seven Proposed Bills

The seven proposed bills, each renumbered to HR 9501 through HR 9507
to avoid known active-legislation conflicts (the prior HR 4501-4507
range collides with bills such as the Holy Sovereignty Protection
Act in the 119th Congress) are:

| Bill | Adaption / Revision Of | Patient-Control Aim |
|------|------------------------|---------------------|
| HR 9501 (2026) | 21 CFR Part 50, FDA DCT Final Guidance 2024, 42 USC 300gg-8 | Cancer patient self-selection of trials and 24/7 booking |
| HR 9502 (2026) | California AB 2847 (2026), FDA AI Regulatory Decision-Making Draft 2025 | Cancer patient choice of surgical robot, humanoid, or companion robot |
| HR 9503 (2026) | OHRP Broad Consent 2017, 21st Century Cures Act 2016 | Cancer patient procedural modification authority |
| HR 9504 (2026) | HTI-1 DSI Final Rule 2023, FDA AI Decision-Making Draft Guidance 2025 | Physical AI clinical error reduction |
| HR 9505 (2026) | FDA RTCT Press Announcement April 2026, FDA DCT Guidance 2024 | Real-time patient-sponsor direct communication |
| HR 9506 (2026) | FDORA Section 3209, 21st Century Cures Act 2016 | American physical AI oncology leadership |
| HR 9507 (2026) | HHS HIPAA Right-to-Access 2025, ONC Cures Act Final Rule | Cancer patient health data self-custody and trial-selection |

Every proposed section explicitly states "adaption" or "revision" in
its title to identify the prior U.S. legislative document it builds
from, per the project brief.

## Goals of the Final Paper

1. **Comprehensively introduce the new proposed bills** so that cancer
   patients can use state-of-the-art advanced AI and advanced robotics
   to improve quality and control over their health, replacing prior
   error-prone and physically and intellectually limited human doctors
   and nurses where the patient prefers.
2. **Cite and further address current patient legislation** affecting
   oncology trials. Each bill section opens with the prior layered-
   stack item it adapts or revises (drawn from
   `patients/paper/Deep-Research-2/` and
   `patients/paper/Deep-Research-3/`).
3. **Include new AI-generated proposed bills directly inside this
   paper** with five-subsection legislative-act layouts (Findings,
   Definitions, Operative Rights, Implementation, Reporting and
   Enforcement) for each of HR 9501 through HR 9507.
4. **Reduce human doctor and nurse error rates and lack of physical
   and intellectual capabilities.** HR 9504 in
   `sections/hr_9504_error_reduction.tex` is dedicated to this aim,
   with instructions referring to the AI evidence base in
   `patients/paper/Deep-Research-1/` and
   `patients/paper/Deep-Research-3/part2_ai_robotics_legislation.md`.

The new bills support prior author paper innovations including
patient-booked 24/7 trial time slots, patient choice of surgical
robot or humanoid or companion robot, and patient-controlled
procedure modifications with less human doctor and nurse
intervention.

Mentions of the FDA and other governing bodies remain respectful
and opportunistic, framed as enabling the patient's interests rather
than obstructing them. The United States must remain Number 1 in
the world regarding patient benefit during the current medical AI
and robotics revolution.

## File Structure

```
patients/paper/full-paper/
|-- README.md                                # This file
|-- main.tex                                 # Document entry point + global formatting brief
|-- patient_priority.sty                     # Style file (adapted from prior templates)
|-- patient_priority.bib                     # Bibliography (DOIs + clickable URLs; biber)
|-- LaTeX_Source_Files.zip                   # Overleaf-ready archive (added in final commit)
`-- sections/                                # 8 section .tex files
    |-- hr_9501_patient_self_selection.tex   # Section 1
    |-- hr_9502_robot_humanoid_choice.tex    # Section 2
    |-- hr_9503_procedural_modification.tex  # Section 3
    |-- hr_9504_error_reduction.tex          # Section 4
    |-- hr_9505_realtime_sponsor.tex         # Section 5
    |-- hr_9506_american_leadership.tex      # Section 6
    |-- hr_9507_data_self_custody.tex        # Section 7
    `-- back_matter.tex                      # Acknowledgments + back matter
```

The Deep-Research source materials remain in the parent directory at
`patients/paper/Deep-Research-1/`, `Deep-Research-2/`, and
`Deep-Research-3/`.

## Compilation (Overleaf)

Compile in Overleaf with the standard biber-driven sequence:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

The bibliography uses `biblatex` with `backend=biber`,
`style=numeric`, `sorting=none`. Every reference includes a DOI
string (where one exists) and a single canonical clickable URL. For
repository entries, the `url` field carries the Zenodo doi.org link
and the `note` field carries the GitHub URL, so both render as
separate clickable hyperlinks. The double-URL pattern from the prior
v3.7.0 template (where the same doi.org link appeared twice in one
entry) has been removed.

## Bill Architecture (ASCII)

```
+-------------------------------------------------------------+
|         Patient Priority Paper - 7 Bill Structure           |
+-------------------------------------------------------------+
|                                                             |
|  [Title Page + Table of Contents]                           |
|                                                             |
|  Section 1: HR 9501 - Cancer Patient Self-Selection         |
|             (Trial discovery, 24/7 booking)                 |
|  Section 2: HR 9502 - Cancer Patient Robot/Humanoid Choice  |
|             (10 robot categories, USL transparency)         |
|  Section 3: HR 9503 - Cancer Patient Procedural             |
|             Modification Authority                          |
|             (Real-time consent updates within 5 minutes)    |
|  Section 4: HR 9504 - Physical AI Clinical Error Reduction  |
|             (Per-task error rate publication)               |
|             (Patient-initiated AI second opinion)           |
|  Section 5: HR 9505 - Real-Time Patient-Sponsor Direct      |
|             Communication                                   |
|             (1-hour acknowledgment, FDA RTCT extension)     |
|  Section 6: HR 9506 - American Physical AI Oncology         |
|             Leadership Act                                  |
|             (American Leadership Index, 50-state coverage)  |
|  Section 7: HR 9507 - Cancer Patient Health Data            |
|             Self-Custody and Trial-Selection                |
|             (Same-day pathology, FHIR export at zero cost)  |
|                                                             |
|  References                                                 |
|  Acknowledgments                                            |
|  Ethical disclosures                                        |
|  Rights and permissions                                     |
|  Cite this article                                          |
|                                                             |
+-------------------------------------------------------------+
```

## Consolidation Mapping (v3.7.0 to v3.8.0)

The following table summarizes how prior-template sections were
distributed into the seven bill sections of this full paper:

| Prior Template Section | Consolidated Into |
|------------------------|-------------------|
| Abstract opening thesis | HR 9501 lead paragraph |
| Introduction block 1 (thesis) | HR 9501 + HR 9506 (split) |
| Introduction block 2 (legal stack) | Per-bill current-legislation context |
| Introduction block 3 (transition) | HR 9501 lead paragraph |
| Patient Priority Framework 2.1 (discovery) | HR 9501 |
| Patient Priority Framework 2.2 (consent) | HR 9503 |
| Patient Priority Framework 2.3 (location) | HR 9501 |
| Patient Priority Framework 2.4 (data gov) | HR 9507 |
| Patient Priority Framework 2.5 (AE) | HR 9505 |
| Patient Priority Framework 2.6 (AI exclusion) | HR 9504 |
| Patient Priority Framework 2.7 (NEW robot) | HR 9502 |
| Implementation Metrics 10.1 (timeline) | HR 9506 |
| Implementation Metrics 10.2 Domain 1 (access) | HR 9501 + HR 9507 |
| Implementation Metrics 10.2 Domain 2 (decision) | HR 9503 |
| Implementation Metrics 10.2 Domain 3 (action) | HR 9504 + HR 9505 |
| Implementation Metrics 10.2 Domain 4 (fairness) | HR 9504 |
| Implementation Metrics 10.2 Domain 5 (NEW PAI) | HR 9502 |
| Implementation Metrics 10.3 (guardrails) | HR 9504 + HR 9506 |
| Discussion 11.1 (vs legal stack) | Per-bill discussion |
| Discussion 11.2 (vs AI evidence) | Per-bill discussion |
| Discussion 11.3 (FDA framing) | Per-bill discussion |
| Limitations 12.1.a-g (per-bill) | Each bill's limitation paragraph |
| Limitations 12.1 cross-cutting | HR 9506 |
| Future Work tracks A/B/C | HR 9506 |
| Future Work concrete deliverables | HR 9506 |
| Conclusions 13.1 list | One sentence per bill closing |
| Conclusions 13.2-13.6 | HR 9506 |

## Bibliography Reference Counts

The 47-entry bibliography (down from the prior 56-entry tally after
de-duplication of the double-URL pattern) consists of:

| Category | Count |
|----------|-------|
| Author Zenodo repository entries (each with both Zenodo and GitHub clickable) | 12 |
| U.S. statutory and regulatory baseline entries (single canonical URL) | 15 |
| Layered legal-stack acts (single congress.gov URL) | 9 |
| AI and robotics evidence entries (single doi.org URL) | 8 |
| Patient-control software baselines (single doi.org URL) | 3 |
| AI tooling references (single canonical URL) | 5 |

Every reference includes a DOI string (where one exists) and a
single canonical clickable URL. For repository entries, the `url`
field carries the Zenodo doi.org link and the `note` field carries
the GitHub URL, so both render as separate clickable hyperlinks.

## AVAILABLE DIRECTORIES (cited in section instruction blocks)

The bracketed processing instructions in every `sections/*.tex` file
draw on the four AVAILABLE DIRECTORIES from the project brief
(`national-platform/`, `sponsor/`, `new-trial/national-24-7-trial/paper/`,
`patients/paper/`) plus `patient-journey/` and
`patients/patient_robot_instructions_fixed.tex`. The Deep-Research-1,
Deep-Research-2, and Deep-Research-3 markdown source pools at
`patients/paper/Deep-Research-1/`, `Deep-Research-2/`, and
`Deep-Research-3/` provide the legal-baseline, AI-evidence, and
metrics-guardrails inputs for the bracketed instructions.

## Disclaimer

This is an independent draft and is not endorsed, sponsored, or
approved by any trial sponsor, CRO, site, IRB, regulator, or medical
society. Adapted using Claude Code Opus 4.7 1M Max. CFR-derived
content is from public domain documents. ICH content is copyrighted
and may be used under a public license. Bill numbers and short
titles are illustrative placeholders selected to avoid known active
legislation under the 119th Congress.
