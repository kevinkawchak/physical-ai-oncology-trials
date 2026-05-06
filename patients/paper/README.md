# Patient Priority and Proposed U.S. Bills for Physical AI Oncology Clinical Trials

**v3.7.0 (Patient Priority Paper Template)** *Patient Priority and Proposed U.S. Bills for Physical AI Oncology Clinical Trials* - LaTeX paper template at `patients/paper/` introducing seven proposed federal bills (HR 4501 through HR 4507) that adapt and revise prior U.S. legislation to give cancer patients more control over their disease through Physical AI and advanced robotics. [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20045457-blue)](https://doi.org/10.5281/zenodo.20045457)

## Overview

This is a LaTeX paper template for the manuscript:

> Kawchak K. *Patient Priority and Proposed U.S. Bills for Physical AI
> Oncology Clinical Trials*. Zenodo. 2026; 10.5281/zenodo.20045457.

The template is adapted from two prior templates in this repository (the
all-in-one prior template at `new-trial/site/all-documents/` and the
modular per-section template at `new-trial/national-24-7-trial/paper/`)
into a NEW modular per-section format with one `.tex` file per body
section. Each section file contains bracketed processing instructions
that name the EXACT repository directories and file names (drawn from
the available directories listed below) for the next Claude Code Opus
4.7 1M Max generation pass to consume.

The paper is structured around seven proposed federal bills:

| Bill | Adaption / Revision Of | Patient-Control Aim |
|------|------------------------|---------------------|
| HR 4501 (2026) | 21 CFR Part 50, FDA DCT Final Guidance 2024, 42 USC 300gg-8 | Self-selection of trials and 24/7 booking |
| HR 4502 (2026) | California AB 2847 (2026), FDA AI Regulatory Decision-Making Draft 2025 | Patient choice of surgical robot, humanoid, or companion robot |
| HR 4503 (2026) | OHRP Broad Consent 2017, 21st Century Cures Act 2016 | Patient procedural modification authority |
| HR 4504 (2026) | HTI-1 DSI Final Rule 2023, FDA AI Decision-Making Draft Guidance 2025 | Reduction of human doctor and nurse error rates |
| HR 4505 (2026) | FDA RTCT Press Announcement April 2026, FDA DCT Guidance 2024 | Real-time patient-sponsor direct communication |
| HR 4506 (2026) | FDORA Section 3209, 21st Century Cures Act 2016 | American leadership in medical AI and robotics |
| HR 4507 (2026) | HHS HIPAA Right-to-Access 2025, ONC Cures Act Final Rule | Patient health data self-custody and trial-selection |

The number of bills (seven) was selected to comprehensively cover the
six-dimensional patient-control framework defined in
`patients/paper/Deep-Research-3/part1_legal_baseline.md` plus a seventh
American-leadership statute. Every proposed section explicitly states
"adaption" or "revision" in its title to identify the prior U.S.
legislative document it builds from, per the project brief.

## Goals of the Final Paper (Future Pass)

The next Claude Code Opus 4.7 1M Max generation pass will populate every
bracketed instruction block in the section files into final prose,
yielding a 70+ page manuscript. The paper's main goals are:

1. **Comprehensively introduce the new proposed bills** so that cancer
   patients can use state-of-the-art advanced AI and advanced robotics
   to improve quality and control over their health, replacing prior
   error-prone and physically and intellectually limited human doctors
   and nurses where the patient prefers.
2. **Cite and further address current patient legislation** affecting
   oncology trials. The Introduction section in
   `sections/introduction.tex` names the exact prior statutes and
   regulations to cite, drawn from
   `patients/paper/Deep-Research-2/` and
   `patients/paper/Deep-Research-3/`.
3. **Include new AI generated proposed bills directly inside this
   paper** under development. Sections 3 through 9
   (`sections/hr_4501_*.tex` through `sections/hr_4507_*.tex`)
   are placeholder containers for the bills.
4. **Reduce human doctor and nurse error rates and lack of physical
   and intellectual capabilities.** HR 4504 in
   `sections/hr_4504_error_reduction.tex` is dedicated to this aim,
   with instructions referring to the AI evidence base in
   `patients/paper/Deep-Research-1/` and
   `patients/paper/Deep-Research-3/part2_ai_robotics_legislation.md`.

The new bills support prior author paper innovations including
patient-booked 24/7 trial time slots, patient choice of surgical robot
or humanoid or companion robot, and patient-controlled procedure
modifications with less human doctor and nurse intervention. These
innovations are documented in the AVAILABLE DIRECTORIES below.

Mentions of the FDA and other governing bodies must remain respectful
and opportunistic, framed as enabling the patient's interests rather
than obstructing them. The United States must remain Number 1 in
the world regarding patient benefit during the current medical AI and
robotics revolution.

## File Structure

```
patients/paper/
|-- README.md                              # This file
|-- main.tex                               # Document entry point + global formatting brief
|-- patient_priority.sty                   # Style file (adapted from prior templates)
|-- patient_priority.bib                   # Bibliography (DOIs + clickable URLs; biber)
|-- LaTeX_Source_Files.zip                 # Overleaf-ready archive (added in final commit)
|-- Deep-Research-1/                       # Source: AI and Patient Control evidence
|   |-- README.md
|   |-- chunk_1_foundations.md
|   |-- chunk_2_advanced_systems.md
|   `-- chunk_3_references_bibtex.md
|-- Deep-Research-2/                       # Source: Layered legal-stack baseline
|   |-- README.md
|   |-- chunk_01_intro_ranks1-4.md
|   |-- chunk_02_ranks5-9.md
|   |-- chunk_03_ranks10-13_future.md
|   `-- chunk_04_bibtex.md
|-- Deep-Research-3/                       # Source: Oncology trial laws + patient control
|   |-- README.md
|   |-- part1_legal_baseline.md
|   |-- part2_ai_robotics_legislation.md
|   |-- part3_metrics_guardrails.md
|   `-- part4_bibtex.md
`-- sections/                              # 15 section .tex files (one per main.tex \section)
    |-- abstract.tex
    |-- introduction.tex
    |-- patient_priority.tex
    |-- hr_4501_patient_self_selection.tex
    |-- hr_4502_robot_humanoid_choice.tex
    |-- hr_4503_procedural_modification.tex
    |-- hr_4504_error_reduction.tex
    |-- hr_4505_realtime_sponsor.tex
    |-- hr_4506_american_leadership.tex
    |-- hr_4507_data_self_custody.tex
    |-- implementation_metrics.tex
    |-- discussion.tex
    |-- limitations_future.tex
    |-- conclusions.tex
    `-- back_matter.tex
```

## Compilation (Overleaf)

Compile in Overleaf with the standard biber-driven sequence:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

The bibliography uses `biblatex` with `backend=biber`,
`style=numeric`, `sorting=none`. Every reference includes a DOI string
and a clickable URL through the `note` field (per the project brief).
Repository entries include BOTH a GitHub URL and a Zenodo URL inside
the `note` field, both rendered as clickable hyperlinks.

## AVAILABLE DIRECTORIES (cited in section instruction blocks)

The bracketed processing instructions in every `sections/*.tex` file
draw on the following four AVAILABLE DIRECTORIES from the project
brief. Each directory and its key files are named explicitly in the
sections so that the future Claude Code Opus 4.7 1M Max pass can
locate and process them without ambiguity.

### A. main/national-platform/ and subdirectories

- `national-platform/README.md`
- `national-platform/ich_e6r3_adapt/`
  (`01_preamble_principles_investigator.tex`,
  `02_sponsor_responsibilities.tex`, `03_data_governance.tex`,
  `04_appendices_glossary.tex`)
- `national-platform/21cfr50_adapt/`
  (`01_preamble_scope_definitions_consent.tex`,
  `02_irb_review_pediatric.tex`,
  `03_additional_safeguards_closing.tex`)
- `national-platform/21cfr312_adapt/`
  (`01_preamble_scope_definitions.tex`,
  `02_ind_content_phases.tex`,
  `03_protocol_amendments_reporting.tex`,
  `04_annual_reports_withdrawal.tex`,
  `05_clinical_holds_appendices_closing.tex`)
- `national-platform/patient_robot/`
  (`01_preamble_robots1to5.tex`,
  `02_robots6to10_closing.tex`)
- `national-platform/patient_journey/`
  (`01_preamble_introduction_stages1to4.tex`,
  `02_stages5to8.tex`,
  `03_stages9to10_discussion_closing.tex`)
- `national-platform/usl_standard/`
  (`01_preamble_introduction_framework.tex`,
  `02_results_discussion_closing.tex`)
- `national-platform/national_mcp/`
  (`chunk1_preamble_and_introduction.tex`,
  `chunk2_methods_and_results_part1.tex`,
  `chunk3_results_part2.tex`,
  `chunk4_discussion_and_conclusion.tex`)
- `national-platform/federated_learning/`
  (`main_chunk1_preamble_intro_methods.tex`,
  `main_chunk2_results_architecture_pillars_examples.tex`,
  `main_chunk3_results_peerreview_trust_analytics.tex`,
  `main_chunk4_discussion_limitations_conclusion.tex`)
- `national-platform/new_trial_psl/`
  (`01_preamble_and_sb1042.tex` through `11_emergency_preparedness.tex`,
  11 chunk files mirroring the SB 1042 / AB 2847 / SB 892 stack)
- `national-platform/research_a/`
  (`01_research_a_part1.txt`, `02_research_a_part2.txt`)
- `national-platform/research_b/`
  (`01_research_b_part1.txt`, `02_research_b_part2.txt`)
- `national-platform/new_template/sections/`
  (20 `.tex` files: `cover_page.tex`, `contents.tex`,
  `executive_summary.tex`, `introduction.tex`,
  `regulatory_landscape.tex`, `gov_framework.tex`,
  `patient_journey.tex`, `patient_instructions.tex`,
  `cfr312_adaptation.tex`, `cfr50_adaptation.tex`,
  `ich_e6r3_adaptation.tex`, `psl_usl_standards.tex`,
  `national_mcp.tex`, `federated_learning.tex`,
  `site_establishment.tex`, `implementation_strategy.tex`,
  `financial_analysis.tex`, `discussion.tex`,
  `conclusion.tex`, `appendices.tex`)
- `national-platform/new_paper/main.tex`,
  `national-platform/new_paper/page_styles.tex`,
  `national-platform/new_paper/references.bib`,
  `national-platform/new_paper/sections/` (20 `.tex` files matching
  the new_template),
  `national-platform/new_paper/final_paper/` (final-prose
  population of the new_paper sections)
- `national-platform/paper_template/main.tex`,
  `national-platform/paper_template/biblio.bib`,
  `national-platform/paper_template/page_styles.tex`,
  `national-platform/paper_template/chapters/`
  (`title_pages.tex`, `abstract.tex`, `contents.tex`,
  `introduction.tex`, `theoretical_framework.tex`,
  `methods.tex`, `results.tex`, `discussion.tex`,
  `conclusion.tex`, `acknowledgements.tex`, `appendix.tex`)

### B. main/sponsor/ and subdirectories

- `sponsor/README.md`
- `sponsor/input_files/` (sponsor playbook + organization, 16 chunks)
  including `sponsor_01-08_*.md` and `org_01-07_*.md`
- `sponsor/paper/` (Autonomous Sponsor Paper v3.2.0:
  `main.tex`, `sponsor_paper.sty`, `references.bib`,
  `sections/` 19 `.tex` files including
  `clinical_operations.tex`, `data_management.tex`,
  `discussion.tex`, `financial_analysis.tex`,
  `governance.tex`, `implementation_strategy.tex`,
  `introduction.tex`, `quality_compliance.tex`,
  `regulatory_affairs.tex`, `robotic_execution.tex`,
  `safety_pharmacovigilance.tex`, `site_interface.tex`,
  `supply_chain.tex`, `trial_design.tex`, `trust_layer.tex`,
  `vendor_management.tex`, `writing_disclosure.tex`,
  `appendices.tex`, `conclusion.tex`)
- `sponsor/final_paper/` (final-prose population of the sponsor paper
  with code generations, including `scripts/core_agents/`
  53 agents, `scripts/coordination/`, `scripts/dashboard/`,
  `scripts/diagrams/` 75 ASCII diagrams,
  `scripts/hourly/sponsor_hour_00.py` through
  `sponsor_hour_23.py`, `scripts/safety/`,
  `scripts/sponsor_server/` FastAPI control server, and
  `scripts/output/sponsor_24h_summary.json`)
- `sponsor/final_paper/168_hours/` (7-day extension:
  `day_01/` through `day_07/` each with `hourly/` 24 scripts,
  `diagrams/` 75 per day, `output/` JSON, plus a master
  `run_168h_simulation.py` and the local-verification README at
  `sponsor/final_paper/168_hours/instructions/core_i5_6200u_4gb/README.md`)
- `sponsor/template/` (Autonomous Sponsor Paper Template v3.1.0:
  `main.tex`, `sponsor_paper.sty`, `references.bib`,
  `sections/` 19 `.tex` files matching `sponsor/paper/sections/`)

### C. main/new-trial/national-24-7-trial/paper/ and subdirectories

- `new-trial/national-24-7-trial/paper/README.md`
- `new-trial/national-24-7-trial/paper/main.tex`
- `new-trial/national-24-7-trial/paper/new_paper.sty`
- `new-trial/national-24-7-trial/paper/references.bib`
- `new-trial/national-24-7-trial/paper/orcid_icon.png`
- `new-trial/national-24-7-trial/paper/LaTeX_Source_Files.zip`
- `new-trial/national-24-7-trial/paper/sections/`
  (`abstract.tex`, `introduction.tex`, `methods.tex`, `results.tex`,
  `discussion.tex`, `limitations_future.tex`, `conclusions.tex`,
  `back_matter.tex`)
- `new-trial/national-24-7-trial/paper/full-paper/` (polished v3.6.0
  manuscript: `main.tex`, `new_paper.sty`, `references.bib`,
  `orcid_icon.png`, `README.md`, `LaTeX_Source_Files.zip`,
  `sections/` 8 polished `.tex` files)
- Source materials adjacent to the paper directory:
  - `new-trial/national-24-7-trial/Background-A/`
    (`README.md`, `chunk_01_baseline_and_short_horizon.md`,
    `chunk_02_multimodal_and_limitations.md`, `chunk_03_bibtex.md`)
  - `new-trial/national-24-7-trial/Background-B/`
    (`README.md`, `chunk_01_baseline_and_prediction_domains.md`,
    `chunk_02_response_metrics_conclusions.md`,
    `chunk_03_bibtex_references.md`)
  - `new-trial/national-24-7-trial/FDA-April-2026/FDA_RealTime_Clinical_Trials.md`
  - `new-trial/national-24-7-trial/hour-00/` through `hour-55/`
    (per-hour: `simulation.md`, `robot_logs.md`,
    `patient_records.md`, `psl_scores.md`,
    `diagram_facility.txt`, `diagram_patient_flow.txt`,
    `diagram_robot_status.txt`)
  - `new-trial/national-24-7-trial/extra-hours/hour-56/` through
    `hour-83/` (approximated diagrams)

### D. main/patients/paper/ and subdirectories

- `patients/paper/README.md` (this file)
- `patients/paper/main.tex`
- `patients/paper/patient_priority.sty`
- `patients/paper/patient_priority.bib`
- `patients/paper/Deep-Research-1/`
  (`README.md`, `chunk_1_foundations.md`,
  `chunk_2_advanced_systems.md`,
  `chunk_3_references_bibtex.md`)
- `patients/paper/Deep-Research-2/`
  (`README.md`, `chunk_01_intro_ranks1-4.md`,
  `chunk_02_ranks5-9.md`, `chunk_03_ranks10-13_future.md`,
  `chunk_04_bibtex.md`)
- `patients/paper/Deep-Research-3/`
  (`README.md`, `part1_legal_baseline.md`,
  `part2_ai_robotics_legislation.md`,
  `part3_metrics_guardrails.md`, `part4_bibtex.md`)
- `patients/paper/sections/` (15 `.tex` files; one per
  `\section` in `main.tex`)
- Adjacent prior patient-instructions material:
  - `patients/patient_robot_instructions_fixed.tex`
  - `patients/patient_robot_instructions.sty`
  - `patients/research/v1.9.0/` and `patients/research/v1.9.1/`

## Processing Instructions for the Future Claude Code 4.7 1M Max Pass

When the next generation pass populates this template:

1. **Treat each `sections/*.tex` file as a self-contained scope.**
   The bracketed instruction blocks in each section name the EXACT
   repository directories and file names to read.
2. **Cover both individual patients and individual robots, AND broad
   adoption.** Every bill section must include both: (a) the
   per-patient and per-robot detail (named patient identifiers, named
   robot instances drawn from `patients/patient_robot_instructions_fixed.tex`,
   `patient-journey/`, and `new-trial/national-24-7-trial/hour-NN/patient_records.md`),
   AND (b) the broad-adoption framing for ALL future cancer patients
   in clinical trials nationwide.
3. **Cite all prior author Zenodo references and external references
   correctly.** Every reference must have a DOI string and a clickable
   URL. Repository entries must have both a GitHub URL and a Zenodo
   URL in the `note` field, both clickable.
4. **Frame the FDA and other governing bodies respectfully.** Position
   them as enabling patient interests, not as obstacles. The patient
   is the priority.
5. **Reinforce U.S. leadership.** Each bill must include a
   "United States Number 1" framing paragraph emphasizing patient
   benefit during the medical AI and robotics revolution.
6. **Apply the global formatting brief at the top of `main.tex`.**
   No right-margin overflow, no orphans, no widows, no large empty
   white-space pages, single dashes only, "SS" replaced with `\S`,
   black text only, hyperref bookmarks via `\phantomsection` plus
   `\addcontentsline` where needed.

## Disclaimer

This is an independent draft and is not endorsed, sponsored, or
approved by any trial sponsor, CRO, site, IRB, regulator, or medical
society. Adapted using Claude Code Opus 4.7 1M Max. CFR-derived
content is from public domain documents. ICH content is copyrighted
and may be used under a public license.
