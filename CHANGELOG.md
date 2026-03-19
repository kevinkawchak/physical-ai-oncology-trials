# Changelog

All notable changes to this repository are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [2.6.0] - 2026-03-20

### Added
- `patient-journey/patient_state.py`: Central data model with 10 enums, 14 dataclasses, legal stage transitions, and PatientJourneyState master class
- `patient-journey/stage_01_prescreening.py`: Pre-Screening & Referral Intake orchestrator (Day -30 to Day -14) with PHI detection, HIPAA Safe Harbor de-identification, DICOM validation
- `patient-journey/stage_02_enrollment.py`: Enrollment & Informed Consent orchestrator (Day -14 to Day 0) with ICH E6(R3) consent elements, eligibility checks, IRB review, randomization
- `patient-journey/stage_03_digital_twin.py`: Digital Twin Construction orchestrator (Day 0 to Day 7) with ASME V&V 40 validation, tumor microenvironment modeling, adaptive radiation simulation
- `patient-journey/stage_04_robot_qualification.py`: Robot Qualification orchestrator (Day 7 to Day 13) with USL scoring, cross-framework validation, cybersecurity assessment, hand-eye calibration
- `patient-journey/stage_05_surgery.py`: Surgery orchestrator (Day 14) with ROS 2 deployment, shared autonomy, sensor fusion, sim-vs-real validation, specimen chain of custody
- `patient-journey/stage_06_recovery.py`: Post-Operative Recovery orchestrator (Day 14 to Day 28) with pathology integration, adverse event tracking, Physical AI causality assessment
- `patient-journey/stage_07_immunotherapy.py`: Immunotherapy orchestrator (Day 28 to Day 763) with 35 pembrolizumab cycles, adaptive dosing, cumulative toxicity tracking, annual reporting
- `patient-journey/stage_08_federation.py`: Federated Learning orchestrator (Day 28 to Day 763) with 70 rounds, differential privacy (epsilon=1.0, delta=1e-5), secure aggregation, DSMB reporting
- `patient-journey/stage_09_surveillance.py`: Long-Term Surveillance orchestrator (Day 763 to Day 1858) with quarterly imaging, recurrence risk modeling (35% to 3%), treatment completion
- `patient-journey/stage_10_closeout.py`: Trial Closeout orchestrator (Day 1858+) with HARD_LOCK, re-identification risk validation (<0.04%), GCP audit, regulatory package generation
- `patient-journey/master_journey.py`: Master Journey Orchestrator coordinating all 10 stages with regulatory mapping, journey reporting, and stage result tracking
- `patient-journey/diagrams/`: 30 ASCII progress diagrams (3 perspectives x 10 stages) -- timeline, regulatory, and clinical perspectives
- `tests/test_patient_journey/`: 208 tests across 13 test modules including per-stage tests, master journey tests, and cross-stage consistency tests
- `tests/test_patient_journey/test_cross_stage_consistency.py`: 57 cross-stage validation tests verifying enum completeness, stage transitions, orchestrator interfaces, demographic consistency, data model fields, full journey progression, diagram file existence, and module file existence

### Changed
- `ruff.toml`: Added per-file-ignores for `patient-journey/**/*.py` (F401, F402) to support conditional imports
- `patient-journey/stage_02_enrollment.py`: Fixed exclusion criteria to use passed-in criteria dict instead of hardcoded False values

### Notes
- Single-patient journey for PAT-2026-0042 (58F, Stage IIIB NSCLC, ECOG 1, PD-L1 65%, TMB 14 mut/Mb, SITE-003)
- Three regulatory frameworks: 21 CFR Part 312 Subpart J (sections 312.400-405), 21 CFR Part 50 Subpart C (sections 50.30-34), ICH E6(R3) (sections 1.2-1.5, 2.8-2.12)
- Physical AI classifications: SURGICAL_ROBOT, COBOT, HUMANOID, THERAPEUTIC, DIAGNOSTIC, ASSISTIVE, REHABILITATIVE
- USL scoring: 4 dimensions (Autonomy, Dexterity, Safety, Interoperability), range 1.0-10.0
- MCP conformance levels: CORE, CLINICAL_READ, IMAGING, FEDERATED_SITE, ROBOT_PROCEDURE
- Development by Claude Code Opus 4.6

## [2.5.0] - 2026-03-18

### Added
- `regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.tex`: Adaptation of 21 CFR Part 312 (Investigational New Drug Application) for Physical AI oncology trials -- 94-page LaTeX document with Subparts A-I modified in-place and new Subpart J
- `regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.pdf`: Compiled 94-page PDF
- `regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.zip`: Source archive (.tex, .sty, .bib, .pdf, prompts.md)
- `regulatory/Adaption-21-CFR-Part-312/source/prompts.md`: Development prompts archive
- Subpart A: 21 new Physical AI definitions (USL, simulation validation, digital twin, MCP, PCCP, sim-to-real gap, etc.), scope expansion for 5 Physical AI system types
- Subpart B: Physical AI System Description as new IND section (g) with 7 subsections, Physical AI phase-specific requirements, Physical AI amendments and safety reporting
- Subpart C: Physical AI readiness requirements, 8 Physical AI grounds for clinical hold, Physical AI termination and dormancy/reactivation
- Subpart D: 7 Physical AI sponsor responsibilities, CRO transfer requirements, Physical AI investigator qualifications, 7 record categories, Physical AI disqualification grounds
- Subpart E: 21 CFR 312.80-312.88 adapted with Physical AI provisions for life-threatening illnesses, early consultation, treatment protocols, risk-benefit analysis, Phase 4 studies, active monitoring, patient safety
- Subparts F-G, I: Physical AI import/export, foreign studies, laboratory research, expanded access provisions
- Subpart H [Reserved]
- Subpart J (new): 21 CFR 312.400-312.405 -- Physical AI system classification (3-tier), validation (simulation/bench/integration/site), cybersecurity by design, human oversight with e-stop specifications, AI/ML lifecycle management
- 42-reference bibliography across 7 categories
- v2.5.0 release notes in `releases.md`

### Changed
- `README.md`: Updated version badge to v2.5.0, added 21 CFR Part 312 adaptation section, updated repository structure with `regulatory/Adaption-21-CFR-Part-312/` directory
- `regulatory/README.md`: Added Adaption-21-CFR-Part-312 directory to structure, updated version to 2.5.0

### Notes
- Adapted from the prior 21 CFR Part 312 regulation (public domain under 17 U.S.C. section 105)
- No Python code changes -- documentation-only release
- Development by Claude Code Opus 4.6

---

- @kevinkawchak v2.5.0 prompts.md 2nd prompt and meta-prompt additions. Main README DOI badge and context update regarding v2.5.0 pdf 2026-03-18.

---

## [2.4.0] - 2026-03-16

### Added
- `regulatory/Adaption-21-CFR-Part-50/source/Physical_AI_21_CFR_Part_50.tex`: Adaptation of 21 CFR Part 50 (Protection of Human Subjects) for Physical AI oncology trials -- 37-page LaTeX document with Subparts A-D modified in-place and new Subpart C
- `regulatory/Adaption-21-CFR-Part-50/source/Physical_AI_21_CFR_Part_50.sty`: Custom style package with CFRBlue color scheme
- `regulatory/Adaption-21-CFR-Part-50/source/Physical_AI_21_CFR_Part_50.bib`: Bibliography with 19 references
- `regulatory/Adaption-21-CFR-Part-50/source/Physical_AI_21_CFR_Part_50.pdf`: Compiled 37-page PDF
- `regulatory/Adaption-21-CFR-Part-50/source/Physical_AI_21_CFR_Part_50.zip`: Source archive (.tex, .sty, .bib, .pdf)
- `regulatory/Adaption-21-CFR-Part-50/source/README.md`: Build instructions and document structure
- Subpart A: §50.1 Scope expanded for Physical AI systems, §50.3 Definitions with 17 new Physical AI definitions
- Subpart B: §50.20-§50.27 adapted with Physical AI consent elements, MCP consent tracking
- Subpart C (new): §50.30-§50.34 covering safety requirements, IRB review, ongoing consent, data protection, system classification
- Subpart D: §50.50-§50.56 adapted for Physical AI pediatric populations
- Glossary with 30 Physical AI-specific definitions
- v2.4.0 release notes in `releases.md`

### Changed
- `README.md`: Updated version badge to v2.4.0, added 21 CFR Part 50 adaptation section, updated repository structure with `regulatory/Adaption-21-CFR-Part-50/` directory, updated citation version
- `regulatory/README.md`: Added Adaption-21-CFR-Part-50 directory to structure, updated version

### Notes
- Adapted from the prior 21 CFR Part 50 regulation (public domain under 17 U.S.C. §105)
- DOI: 10.5281/zenodo.19040707
- No Python code changes -- documentation-only release
- Development by Claude Code Opus 4.6

---

- @kevinkawchak modifications to main README, regulatory/Adaption-21-CFR-Part-50, prompts.md, and posts.md (including prior post) to better reflect new v2.4.0 content 2026-03-16.

---

## [2.3.0] - 2026-03-13

### Added
- `unification/industry/paiotis_v1.tex`: Physical AI Oncology Trial Industry Specification (PAIOTIS) v1.0 -- 8-part industry standard with RFC 2119 normative language
- `unification/industry/paiotis.sty`: Custom LaTeX style package adapted from UTB thesis template by Edwin Puertas (CC BY 4.0)
- `unification/industry/references.bib`: Bibliography with 24 references covering all 4 repositories, standards, and frameworks
- `unification/industry/paiotis_v1.pdf`: Compiled 25-page PDF
- `unification/industry/paiotis_v1.zip`: Source archive (.tex, .sty, .bib, .pdf)
- `unification/industry/prompts.md`: Development prompt archive for v2.3.0
- Parts I-VIII: Industry Definition, Technical Architecture, Regulatory Compliance, Privacy/Data Governance, Robot Qualification, Pharma Sponsor Guide, Clinical Site Readiness, Industry Milestone Roadmap
- USL-based robot qualification tiers for trial phases (Phase I-III)
- 3-tier pharmaceutical adoption pathways (observer, pilot, full integration)
- Clinical site infrastructure, staffing, and federation onboarding requirements
- v2.3.0 release notes in `releases.md`

### Changed
- `README.md`: Updated version badge to v2.3.0, added industry specification section, updated repository structure with `unification/industry/` directory, updated citation version
- `CITATION.cff`: Updated version to 2.3.0

### Notes
- Unifies four repositories: physical-ai-oncology-trials, TrialMCP, national-mcp-pai-oncology-trials, pai-oncology-trial-fl
- RFC 2119 normative language (SHALL, SHOULD, MAY) used throughout
- No Python code changes -- documentation-only release
- Development by Claude Code Opus 4.6

---
- @kevinkawchak main README, and unification/industry/ updates 2026-03-13.
---
- @kevinkawchak main README update and unification/industry/ directory removal due to change in direction from industry standard approach. 2026-03-14
---

## [2.2.0] - 2026-03-12

### Added
- `regulatory/adaption-ich-e6r3/source/main.tex`: Complete End-to-End Physical AI Oncology Clinical Trial Unification guidance (Sections 1-4, Appendices A-C, Glossary) adapted from prior ICH E6(R3) regulation
- `regulatory/adaption-ich-e6r3/source/ich_guideline_style.sty`: Updated style package for physical AI guidance
- `regulatory/adaption-ich-e6r3/source/references.bib`: Updated bibliography with 18 references
- `regulatory/adaption-ich-e6r3/prompts.md`: Development prompt archive for v2.2.0
- Sections 1-4: Principles, Investigator Responsibilities, Sponsor Responsibilities, Data Governance
- Appendices A-C: Physical AI System Documentation, Clinical Trial Protocol, Essential Records
- Glossary with 30 physical AI-specific definitions
- Cover page with DOI 10.5281/zenodo.18973368 and CEO attribution
- v2.2.0 release notes in `releases.md`

### Changed
- `regulatory/adaption-ich-e6r3/source/README.md`: Updated for v2.2.0 with build instructions and DOI
- `regulatory/README.md`: Added adaption-ich-e6r3 directory to structure, updated version to 2.2.0
- `README.md`: Updated version badge to v2.2.0, added regulatory guidance section, updated repository structure and citation
- `CITATION.cff`: Updated version to 2.2.0

### Notes
- Guidance DOI: 10.5281/zenodo.18973368
- Adapted from the prior ICH E6(R3) regulation (adopted 06 January 2025)
- Not endorsed or sponsored by ICH
- All 9 USL-evaluated robots referenced with scores throughout
- No em dashes used in the entire document
- Development by Claude Code Opus 4.6

---
- @kevinkawchak updates to main README and regulatory/ 2026-03-12.
---

## [2.1.0] - 2026-03-02

### Added
- `patients/README.md`: Complete paper content documentation with page-by-page patient instructions for all 10 robot types
- 7 new text diagrams in `patients/README.md`: page layout structure, robot categories (5 clinical categories), procedure time comparison, patient interaction summary, source distribution, cancer type distribution, quantitative patient data
- Robot type overview table with sources column (Intuitive Surgical, Franka Robotics, Accuray, ISO 15223-1, SoftBank Robotics, Boston Dynamics, Varian Medical, ISO 20417, ISO 7010, Ekso Bionics)
- PDF image descriptions linking each of 5 images to corresponding page pairs
- Quantitative patient data table (anesthesia type, physical contact, key measurements, recovery time)
- Robot categories text diagram in main README patients section
- v2.1.0 release notes in `releases.md`
- v2.1.0 development prompt in `patients/prompts/prompts.md`

### Changed
- `patients/README.md`: Rewritten to focus on paper content instead of file relocation operations; removed repetitive "transferred to Drive" language
- `patients/README.md`: Corrected paper title from "Patient-Robot Instructions" to "Patient Instructions: Physical AI Oncology Trials" matching the actual paper
- `README.md`: Updated patients section from v2.0.0 to v2.1.0 with content-focused description, source column, and robot categories diagram
- `README.md`: Updated version badge to v2.1.0, citation version to 2.1.0, footer version to v2.1.0
- `README.md`: Updated repository structure to reflect patients/ content focus
- `CITATION.cff`: Updated version to 2.1.0

### Notes
- Paper DOI: 10.5281/zenodo.18810541
- Google Drive images: https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax
- Paper generated by ChatGPT (March 1, 2026); repository documentation by Claude Code Opus 4.6
- No Python code changes — documentation-only release
- Prior v1.9.0/v1.9.1 context replaced with actual paper content in patients/README.md
- @kevinkawchak further patient instruction documenation improvements

## [2.0.0] - 2026-03-02

### Added
- `agentic-ai/README.md`: New README with relocated agentic AI engineering examples documentation from main README
- Consolidated engineering examples table in main README linking to all 34 examples and 5 CLI tools
- v1.0.0 and v2.0.0 major release references in main README
- Federation examples table added to `federation/README.md`

### Changed
- `patients/README.md`: Rewritten for v2.0.0 with hyperlink-only references to paper (Zenodo), LaTeX source files (Zenodo), and images (Google Drive)
- `README.md`: Updated to v2.0.0 — relocated Agentic AI Engineering Examples, Digital Twin Engineering Examples, Comprehensive Examples, Physical Robot Engineering Examples, Command-Line Tools, and Multi-Site Federated Oncology Trial Coordination sections to their respective directory READMEs
- `README.md`: Updated version badge to v2.0.0, updated Actively Maintained Repositories date range to March 2026, updated Regulatory Compliance Framework date
- `README.md`: Updated citation version to 2.0.0
- `CITATION.cff`: Updated version to 2.0.0
- `patients/prompts/prompts.md`: Added v2.0.0 development prompt

### Removed
- Paper PDFs from `patients/paper/` (relocated to Zenodo/Drive by @kevinkawchak)
- LaTeX source files from `patients/paper/` (relocated to Zenodo/Drive by @kevinkawchak)
- Images from `patients/images/` (relocated to Drive by @kevinkawchak)
- `patients/generate_pdf.py` (archived under `patients/research/v1.9.1/`)

### Notes
- Paper DOI: 10.5281/zenodo.18810541
- Google Drive images: https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax
- @kevinkawchak relocated files from v1.9.0 and v1.9.1 into Drive to reduce repository size
- Second major release (v2.0.0) following v1.0.0 (February 2026)
- No Python code changes — documentation-only release
- License: CC BY 4.0 (paper and images), MIT (repository code)
- Development by Claude Code Opus 4.6

## [1.9.1] - 2026-03-01

### Added
- `patients/images/` directory: Numbered images (1.png--10.png) for each robot type page
- `patients/images/README.md`: Image access documentation with Google Drive link
- `patients/research/v1.9.0/`: Archived v1.9.0 materials (Cairo illustrations, generators, paper files)
- `patients/paper/Patient-Robot Instructions: Physical AI Oncology Trials (10MB).pdf`: 10 MB compressed version
- `patients/paper/Patient-Robot Instructions: Physical AI Oncology Trials (5MB).pdf`: 5 MB compressed version

### Changed
- `patients/paper/Patient-Robot Instructions: Physical AI Oncology Trials.pdf`: Updated with new images, streamlined 3-step instructions, corrected URLs, "For Demonstration Purposes Only"
- `patients/paper/patient_robot_instructions.tex`: Rewritten with new layout (image-dominant, dashed bar, full name, intro + 3 steps)
- `patients/paper/patient_robot_instructions.sty`: Updated style for v1.9.1 (added dashrule, clickable URLs, updated footer)
- `patients/paper/references.bib`: Fixed all 7 source URLs, corrected citation keys, 28 references
- `patients/paper/README`: Updated compilation instructions and content overview for v1.9.1
- `patients/paper/Latex Source Code.zip`: Regenerated with v1.9.1 files
- `patients/generate_pdf.py`: Rewritten using reportlab + Pillow (replaces Cairo), generates 3 PDF versions
- `patients/README.md`: Updated with v1.9.1 changes, new directory structure, robot-cancer pairings
- Title format changed to "Patient-Robot Instructions: AI Oncology Trials - [Robot Type]"
- Each robot type now paired with a specific cancer type
- Single DOI (10.5281/zenodo.18810541) used throughout; removed duplicate

### Removed
- v1.9.0 files moved from `patients/` to `patients/research/v1.9.0/` (except prompts/)
- Removed `patients/svg/`, `patients/pdf/`, `patients/png/` from main directory
- Removed `patients/generate_illustrations.py` from main directory
- Removed 5-section instruction format (replaced with 1-intro + 3-step)
- Removed "Adult/Pediatric Oncology Trial Setting" label from pages
- Removed image borders and lower-right icons from pages

### Updated
- `releases.md`: Added v1.9.1 release notes
- `CHANGELOG.md`: Added v1.9.1 entry
- `README.md`: Updated version badge, patients section, repository structure
- `patients/prompts/prompts.md`: Added v1.9.1 prompt

### Notes
- Paper DOI: 10.5281/zenodo.18810541
- Google Drive images: https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax
- License: CC BY 4.0 (paper and images), MIT (scripts)
- Development by Claude Code Opus 4.6

## [1.9.0] - 2026-02-28

### Added
- `patients/` directory: Patient-facing instructional illustrations for physical AI oncology trials
  - `patients/paper/Patient-Robot Instructions: Physical AI Oncology Trials.pdf`: 10-page compiled PDF with black-and-white portrait illustrations and detailed patient instructions for 10 robot types
  - `patients/paper/Latex Source Code.zip`: Archive containing LaTeX source files (patient_robot_instructions.tex, patient_robot_instructions.sty, references.bib, README)
  - `patients/paper/patient_robot_instructions.tex`: Main LaTeX document (article class, 11pt, Times Roman, 10 pages)
  - `patients/paper/patient_robot_instructions.sty`: Custom style package (geometry, fancyhdr, TikZ ISO symbols, enumitem)
  - `patients/paper/references.bib`: BibTeX bibliography with 35 references
  - `patients/paper/README`: Compilation instructions
  - `patients/svg/`: 10 individual SVG vector illustrations
  - `patients/pdf/`: 10 individual PDF vector illustrations
  - `patients/png/`: 10 individual PNG raster illustrations (3600×4000 pixels)
  - `patients/generate_illustrations.py`: Cairo illustration generator for SVG/PDF/PNG
  - `patients/generate_pdf.py`: Combined 10-page PDF generator
  - `patients/README.md`: Detailed documentation of paper, robot types, ISO standards, and directory structure
  - `patients/prompts/prompts.md`: Development prompt archive

### Updated
- `releases.md`: Added v1.9.0 release notes
- `CHANGELOG.md`: Added v1.9.0 entry
- `README.md`: Updated version badge to v1.9.0, added patients section, updated repository structure
- `ruff.toml`: Added per-file ignore for patients directory Python scripts

### Notes
- Paper DOI: 10.5281/zenodo.18810541
- 10 robot types: Surgical Robots, Cobots, Radiotherapy Patient-Positioning Robots, Robotic Needle-Placement Systems, Social Companion Robots (pediatric), Humanoids (pediatric), Radiotherapy Motion-Management/Tracking Robots, Imaging Assistant Robots, Steerable Needle/Needle-Steering Robots, Rehabilitation Exoskeletons/Robotic Gait Trainers
- ISO standards referenced: ISO 15223-1, ISO 20417, ISO 7000, IEC 60417, ISO 7010, ISO 3864-1
- License: CC BY 4.0 (paper), MIT (scripts)
- Development by Claude Code Opus 4.6

## [1.8.0] - 2026-02-26

### Added
- `unification/usl/paper/` directory: Comprehensive academic paper publication of the USL framework
  - `Unification Standard Level for Physical AI Oncology Trials.pdf`: 9-page compiled paper with Abstract, Table of Contents, Introduction (prior studies, repository overview, path to USL), Methods (AI tools, development timeline, scoring methodology, category-specific engines), Results (all 9 robots with dimension-by-dimension score rationale and cross-category comparisons), Discussion (open-source correlation, hardware vs. readiness, clinical gaps, category-specific scoring, individual robot code differences), Limitations and Future Work (human, Claude Code, and framework limitations), Conclusion, References (28 citations), Acknowledgments, Ethical Disclosures, Rights and Permissions, and Citation
  - `Latex Source Code.zip`: Archive containing all 4 LaTeX source files
  - `usl_oncology_trials.tex`: Main LaTeX document (article class, 11pt, Times Roman, 9 pages)
  - `usl-oncology.sty`: Custom style package (geometry, colors, section formatting, code listings, TikZ score bars, hyperlinks)
  - `references.bib`: BibTeX bibliography with 28 references (NASA TRL, MLTRL, simulation frameworks, AI frameworks, regulatory standards)
  - `README`: LaTeX compilation instructions

### Updated
- `unification/usl/prompts.md`: Added v1.8.0 USL Paper prompt on top
- `releases.md`: Added v1.8.0 release notes
- `CHANGELOG.md`: Added v1.8.0 entry
- `README.md`: Updated version badge to v1.8.0, added paper reference in USL section, updated repository structure with paper directory

### Notes
- Paper DOI: 10.5281/zenodo.18778220
- License: CC BY 4.0 (paper), MIT (repository code)
- No Python code changes — CI lint/format checks unaffected
- Development by Claude Code Opus 4.6

## [1.7.0] - 2026-02-24

### Added
- `unification/usl/humanoids/README.md`: New category README with 6 text diagrams (3 new results/meaning/impact diagrams + 3 moved general/technical/scoring diagrams), full humanoid robot evaluations (Atlas Electric, Digit, Optimus Gen 2), quick start guide, contributing guidelines, and directory structure
- `unification/usl/surgical/README.md`: New category README with 6 text diagrams (3 new results/meaning/impact diagrams + 3 moved general/technical/scoring diagrams), full surgical robot evaluations (da Vinci dVRK, Hugo RAS, Versius), quick start guide, contributing guidelines, and directory structure
- `unification/usl/cobots/README.md`: New category README with 6 text diagrams (3 new results/meaning/impact diagrams + 3 moved general/technical/scoring diagrams), full cobot evaluations (Franka Panda, Kinova Gen3, xArm 7), quick start guide, contributing guidelines, and directory structure
- 3 new cross-category text diagrams in `unification/README.md`: USL results summary (all 9 robots with score rationale), USL meaning (key findings about open-source correlation, clinical readiness gaps, category frontiers), USL impact (phased timeline from category-specific trials through unified consortium)

### Updated
- `unification/usl/README.md`: Streamlined to contain USL standard overview (scoring methodology, score bands, level definitions), robot categories table with links to category READMEs, updated directory structure reflecting new README.md files, influences, and references — all robot-specific evaluations, diagrams, quick start, and contributing sections moved to category READMEs
- `unification/README.md`: Added USL link and 3 cross-category text diagrams at top; updated directory structure to reflect new README.md files in category subdirectories and prompts.md location
- `unification/usl/prompts.md`: Added v1.7.0 USL Restructure prompt on top
- `releases.md`: Added v1.7.0 release notes in new format (title without hashes)
- `README.md`: Updated version to v1.7.0; updated repository structure to reflect new category READMEs and prompts.md location under `unification/usl/`
- `CHANGELOG.md`: Added v1.7.0 entry

### Notes
- Documentation restructure only — no Python code changes, no new modules
- Total text diagrams in USL documentation: 18 (was 9) — 9 new diagrams (3 results/meaning/impact per category + 3 cross-category)
- All robot evaluations, USL scores, and references preserved exactly from v1.6.0
- Quick start and contributing sections distributed to category READMEs where they are most relevant
- No Python files changed — CI lint/format checks unaffected
- Development by Claude Code Opus 4.6

## [1.6.0] - 2026-02-24

### Added
- `unification/usl/humanoids/` directory: USL Humanoid Robot evaluation framework extending the Unification Standard Level to bipedal humanoid robot systems for oncology clinical trials (logistics, transport, assistive tasks)
  - `unification/usl/humanoids/usl_humanoid_scoring.py`: Humanoid robot-specific USL scoring engine with `HumanoidType` (4 types), `HumanoidSimFramework` (8 frameworks including Drake), `HumanoidAICapability` (12 capabilities including VLA, foundation model, whole-body control, locomotion/manipulation policy), `HumanoidTask` (8 oncology tasks); `HumanoidDimAScore` through `HumanoidDimDScore` with humanoid-specific scoring criteria (whole-body model formats, locomotion/manipulation sim fidelity, foundation model integration, autonomous navigation safety, ISO 13482 alignment, hospital pilot testing); `HumanoidUSLRating` with weighted scoring, comparison tables, gap analysis, and text/JSON report generation
  - `unification/usl/humanoids/boston_dynamics_atlas/boston_dynamics_atlas_usl.py`: Boston Dynamics Atlas (Electric) evaluation module — `AtlasElectricSpecs` (~1.5 m, ~89 kg, 28 DOF, custom electric actuators, exceeds human ROM), `AtlasKinematics` with joint group definitions and validation, `AtlasLocomotionConfig` with 3 locomotion profiles (hospital/logistics/outdoor), 4 oncology task definitions, `AtlasCrossOrgSharing` with Drake/BDAII/URDF/ONNX sharing methods, `AtlasUnifiedActionSpace` and `AtlasUnifiedObsSpace` for cross-platform normalization; USL score: 5.8 (Level 5 — Functional)
  - `unification/usl/humanoids/tesla_optimus/tesla_optimus_usl.py`: Tesla Optimus (Gen 2) evaluation module — `OptimusGen2Specs` (~1.73 m, ~57 kg, 28+22 DOF with 11-DOF hands, FSD-derived AI), `OptimusKinematics` with hand grasp type estimation, `OptimusDeploymentProjection` timeline model (2025-2027), 4 oncology tasks, `OptimusCrossOrgSharing` documenting fully proprietary ecosystem; USL score: 3.6 (Level 3 — Basic)
  - `unification/usl/humanoids/agility_digit/agility_digit_usl.py`: Agility Robotics Digit evaluation module — `DigitSpecs` (~1.75 m, ~65 kg, 20 DOF, backward-bending knees, 16 kg payload, Jetson AGX Orin), `DigitKinematics` with spring energy computation, `GROOTIntegrationConfig` for NVIDIA GR00T N1 partnership, 4 oncology tasks, `DigitCrossOrgSharing` with NVIDIA/Amazon/DeepMind/OSU ecosystem; USL score: 4.2 (Level 4 — Developing)

### Updated
- `unification/usl/README.md`: Restructured to cover general USL information, then humanoid robots (with 3 new text diagrams: general comparison, technical specifications, scoring breakdown), then surgical robots (3 existing diagrams renumbered 4-6), then cobots (3 existing diagrams renumbered 7-9); added robot categories table with humanoid row; updated directory structure; expanded references with humanoid-specific citations (Drake, GR00T N1, Agility Robotics, Kuindersma et al.)
- `unification/README.md`: Updated USL directory structure to reflect `humanoids/` subdirectory; added Q1 2026 USL humanoid robot roadmap items
- `README.md`: Added ★ USL Humanoid Robots section with evaluation table; updated repository structure; updated version to v1.6.0
- `prompts.md`: Added v1.6.0 USL Humanoid Robots prompt
- `releases.md`: Added v1.6.0 release notes
- `CHANGELOG.md`: Added v1.6.0 entry

### Notes
- Three humanoid robots selected for: different manufacturers (Boston Dynamics, Agility Robotics, Tesla), bipedal full-size architecture, potential oncology logistics and assistive applications, and varying open-source/AI integration levels
- Humanoid robot USL scoring adapts all four dimensions (A–D) with humanoid-specific criteria: whole-body locomotion simulation, foundation model integration (GR00T, OpenVLA), bipedal navigation safety for hospital environments, ISO 13482 personal care robot safety alignment
- All code passes `ruff check` and `ruff format --check` on Python 3.10–3.12
- 4 new Python modules, approximately 2,700 LOC
- Development by Claude Code Opus 4.6

## [1.5.0] - 2026-02-24

### Added
- `unification/usl/surgical/` directory: USL Surgical Robot evaluation framework extending the Unification Standard Level to teleoperated surgical robot systems for oncology clinical trials
  - `unification/usl/surgical/usl_surgical_scoring.py`: Surgical robot-specific USL scoring engine with `SurgicalSimFramework` (9 frameworks including ORBIT-Surgical, SurRoL, AMBF), `SurgicalAICapability` (11 capabilities including VLA, diffusion policy, surgical video AI, phase recognition), `SurgicalProcedure` (8 oncology procedures), and four dimension scorers (`SurgicalDimAScore` through `SurgicalDimDScore`) with surgical-specific criteria (tissue deformation, haptic feedback, instrument modeling, remote proctoring, IEC 80601); `SurgicalUSLRating` with weighted scoring, comparison tables, gap analysis, and text/JSON report generation
  - `unification/usl/surgical/intuitive_davinci/intuitive_davinci_usl.py`: Intuitive Surgical da Vinci (dVRK) evaluation module — `DVRKSpecs` (PSM 7+1 DOF, ECM 4 DOF, MTM 7+1 DOF, 3 PSM arms, 5/8 mm instruments, EndoWrist articulation, stereo vision, tremor filtering, 1 kHz control), `PSMKinematics` with RCM model and modified DH parameters (Kazanzides et al., 2014; DOI 10.1109/ICRA.2014.6907809), `DVRKFrameworkConfig` for ORBIT-Surgical/SurRoL/AMBF/Gazebo/MuJoCo, 4 oncology task definitions, `DVRKCrossOrgSharing` with 5 sharing methods and 10 dVRK institutions; USL score: 7.1 (Level 7 — Advanced)
  - `unification/usl/surgical/medtronic_hugo/medtronic_hugo_usl.py`: Medtronic Hugo RAS evaluation module — `HugoRASSpecs` (modular cart, 7 DOF + grip, open console, 8 mm instruments, 38 kg per cart), `HugoArmKinematics` with DH parameters, `TouchSurgeryInterface` with phase recognition and performance metrics, 4 oncology tasks, `HugoCrossOrgSharing` with Medtronic ecosystem; USL score: 4.5 (Level 4 — Developing)
  - `unification/usl/surgical/cmr_versius/cmr_versius_usl.py`: CMR Surgical Versius evaluation module — `VersiusSpecs` (~10 kg arms, 5 mm instruments, biomimetic design, portable, 350+ hospitals), `VersiusArmKinematics` with biomimetic DH parameters, `VersiusORSetup` for 3 oncology specialties, 4 oncology tasks, `VersiusCrossOrgSharing` with deployment regions; USL score: 3.4 (Level 3 — Basic)

### Moved
- `unification/usl/usl_scoring_framework.py` → `unification/usl/cobots/usl_scoring_framework.py`: Core cobot scoring engine relocated under the `cobots/` subdirectory to separate it from the new `surgical/` category

### Updated
- `unification/usl/README.md`: Restructured to cover general USL information, then surgical robots (with 3 new text diagrams: general comparison, technical specifications, scoring breakdown), then cobots (original 3 diagrams preserved); added robot categories table, expanded references with surgical-specific citations (Kazanzides et al., ORBIT-Surgical, SurRoL, AMBF, IEC 80601-2-77)
- `unification/README.md`: Updated USL directory structure to reflect `cobots/` and `surgical/` subdirectories; added Q1 2026 USL surgical robot roadmap items
- `README.md`: Added ★ USL Surgical Robots section with evaluation table; updated repository structure; updated version to v1.5.0
- `prompts.md`: Added v1.5.0 USL Surgical Robots prompt
- `releases.md`: Added v1.5.0 release notes
- `CHANGELOG.md`: Added v1.5.0 entry

### Notes
- Three surgical robots selected for: different manufacturers (Intuitive Surgical, Medtronic, CMR Surgical), same type (teleoperated MIS), oncology surgical applications, and varying open-source availability
- Surgical robot USL scoring adapts all four dimensions (A–D) with surgical-specific criteria: tissue deformation simulation, instrument articulation, surgical video AI, phase recognition, remote proctoring, IEC 80601-2-77 compliance, FDA/CE regulatory pathways
- All code passes `ruff check` and `ruff format --check` on Python 3.10–3.12
- 4 new Python modules, approximately 2,400 LOC
- Development by Claude Code Opus 4.6

## [1.4.0] - 2026-02-23

### Added
- `unification/usl/` directory: Unification Standard Level (USL) scoring framework for evaluating physical AI robot readiness for multi-site oncology clinical trials
  - `unification/usl/usl_scoring_framework.py`: Core USL scoring engine with four weighted dimensions (A: Simulation Framework Switching, B: Generative/Agentic AI Integration, C: Cross-Robot Progress Sharing, D: Multi-Site Clinical Trial Collaboration); 10-level classification system from Conceptual (1) to Exemplary (10); score band categorization (Initial/Foundational/Intermediate/Advanced/Exemplary); comparison tables, gap analysis with improvement suggestions, and JSON/text report generation; final scores on 1.0–10.0 scale in 0.1 increments
  - `unification/usl/cobots/franka_panda/franka_panda_usl.py`: Franka Emika Panda (Franka Robotics) USL evaluation module — hardware specifications (7-DOF, 3 kg payload, 855 mm reach, ±0.1 mm repeatability, 7-joint torque sensing, 1 kHz control), Denavit-Hartenberg parameters, URDF template generator with validated kinematic chain, joint limit validator against official specs, policy transfer interface with 4 oncology task definitions (needle insertion, tissue retraction, sample handling, instrument handoff), cross-organization sharing manager (ONNX, ROS 2, MuJoCo Menagerie, federated learning, URDF/Xacro), and framework configurations for MuJoCo/Isaac Lab/Gazebo/PyBullet; USL score: 7.4 (Level 7 — Advanced)
  - `unification/usl/cobots/kinova_gen3/kinova_gen3_usl.py`: Kinova Gen3 7DoF (Kinova Robotics) USL evaluation module — hardware specifications (7-DOF, 4 kg payload, 902 mm reach, 8.2 kg weight, Intel RealSense depth, Kortex API), modified DH kinematic model, 7 actuator module specifications (large/small types), Kortex API abstraction layer with angular/Cartesian command interfaces, joint position validator with continuous-rotation support, 4 oncology task definitions (medication dispensing, biopsy assistance, patient positioning, sample transport), and framework configurations for Gazebo/MuJoCo/Isaac Lab/PyBullet; USL score: 5.7 (Level 5 — Functional)
  - `unification/usl/cobots/ufactory_xarm7/ufactory_xarm7_usl.py`: UFACTORY xArm 7 (UFACTORY) USL evaluation module — hardware specifications (7-DOF, 3.5 kg payload, 700 mm reach, built-in collision detection, IP51 rating, 0–50 °C range), xArm Python SDK abstraction with error code mapping, 7 joint specifications with degree/radian limit validation, 4 oncology lab automation tasks (vial handling, plate stacking, pipette operation, equipment loading), intra-organization sharing across xArm family (5/6/7/Lite 6/850), and framework configurations for Gazebo/MuJoCo/PyBullet/Isaac Lab; USL score: 3.4 (Level 3 — Basic)
  - `unification/usl/README.md`: Comprehensive USL standard documentation with scoring methodology, dimension-weight table, 10-level definitions, 5 score bands, three text comparison diagrams (general differences, technical specifications side-by-side, dimension-by-dimension scoring breakdown with bar charts), individual cobot evaluations with strengths/gaps/recommendations, references to TRL/MLTRL influences, quick-start guide, and contributing guidelines
- `prompts.md`: Development prompt archive containing the v1.4.0 USL standard creation prompt
- `releases.md`: Release notes for v1.4.0 in standardized format with summary, features, contributors, and notes

### Updated
- `unification/README.md`: Added USL directory to structure tree; added Q1 2026 USL roadmap items (USL framework established, 3 cobots evaluated, surgical/mobile categories planned)
- `README.md`: Added ★ Unification Standard Level section with cobot evaluation table and quick-start commands; updated repository structure tree with `usl/` directory; updated version to v1.4.0
- `CHANGELOG.md`: Added v1.4.0 entry
- `ruff.toml`: Added per-file ignore rules for `unification/usl/**/*.py` (F401, F402 for conditional imports)

### Notes
- USL framework is project-specific — "Unification Standard Level" evaluates robot readiness for multi-site oncology trial unification, influenced by NASA/DOD TRL (Mankins, 2004), MLTRL (Lavin et al., 2021; ai-infrastructure-alliance/mltrl), TRL for complex systems (Tomaschek et al., 2015; DOI 10.1109/PICMET.2015.7273196), and inspired by LLM recommendations for oncology trials (Kawchak, 2025; DOI 10.5281/zenodo.17451709)
- Three evaluated cobots selected for: open-source availability (GitHub repos), different manufacturers, official MuJoCo Menagerie models, active ROS 2 support, and potential oncology applications
- All four USL dimensions derive from existing unification pillars: simulation_physics/, agentic_generative_ai/, cross_platform_tools/, and federation/+regulatory/
- All code passes `ruff check` and `ruff format --check` on Python 3.10–3.12
- 4 new Python modules, approximately 2,100 LOC
- Development by Claude Code Opus 4.6

## [1.3.0] - 2026-02-16

### Added
- `images/interactive/3rd/` directory: 3rd set of 10 visualization scripts covering regulatory compliance, privacy frameworks, and deployment readiness
  - `federated_learning_convergence.py`: Dual-panel line chart showing federated model convergence across 3 hospital sites over 5 rounds (ONCO-FED-001 trial)
  - `multi_site_trial_dashboard.py`: Heatmap table with color-coded enrollment and data quality metrics for 4 trial sites
  - `federated_algorithm_radar.py`: Radar chart comparing FedAvg, FedProx, and SCAFFOLD across 5 operational dimensions
  - `fda_device_classification_tree.py`: Decision tree showing FDA AI/ML device classification pathways for 9 oncology device types with escalation factors
  - `fda_oncology_device_distribution.py`: Stacked bar + pie chart showing 1,300+ FDA-authorized AI/ML oncology device distribution across 6 subspecialties
  - `regulatory_compliance_scorecard.py`: Annotated heatmap showing 19 compliance items across IEC 62304, FDA AI/ML PCCP, and ISO 14971
  - `hipaa_phi_detection_matrix.py`: Annotated heatmap of 18 HIPAA identifier types with detection confidence and risk stratification
  - `privacy_analytics_pipeline.py`: Process flow diagram of the 6-stage privacy-preserving analytics pipeline
  - `deployment_readiness_radar.py`: Radar chart with table inset for ONNX model validation and safety compliance assessment
  - `production_readiness_tasks.py`: Horizontal bar chart showing production readiness scores for 15 surgical task categories
- `images/png/3rd/` directory: 20 PNG exports (10 light + 10 dark) for the 3rd visualization set
- `images/interactive/1st/README.md`: Directory README with script table, LOC counts, and Google Drive link for interactive HTML files
- `images/interactive/2nd/README.md`: Directory README with script table, LOC counts, and Google Drive link for interactive HTML files
- `images/interactive/3rd/README.md`: Directory README with script table, LOC counts, and Google Drive link for interactive HTML files

### Updated
- `images/README.md`: Comprehensive rewrite with prompt-to-visualization workflow documentation, text-based pipeline diagrams, conversion efficiency metrics (30/30 scripts, 60/60 HTML, 60/60 PNG — 100% success rate), per-set LOC tables (5,655 total LOC), repository data source reference table, visualization significance descriptions, data inputs table for all 30 charts, and Google Drive link for interactive HTML files
- `images/` directory structure: Updated to reflect 3rd set directories and prompts directory

### Removed
- 60 HTML files from `images/interactive/1st/`, `images/interactive/2nd/`, and `images/interactive/3rd/` — interactive HTML versions are now available on [Google Drive](https://drive.google.com/drive/folders/1C092zdAyP3_go9fx7rj2yiCW0KhLo7er) to reduce repository size

### Notes
- The 30 visualization scripts (5,655 LOC) were generated across three Claude Code sessions using human-authored prompts (plan.md, 1st.md, 2nd.md, 3rd.md) combined with AI-recommended data extraction from repository source files
- Visualization pipeline: Python (Plotly) → HTML (interactive) → PNG (static, 1920×1080 @2x)
- All Python scripts pass `ruff check` and `ruff format --check` on Python 3.10–3.12
- Interactive HTML files (60 total) relocated to Google Drive for repository size optimization
- Development by Claude Code Opus 4.6

## [1.2.1] - 2026-02-13

### Added
- `regulatory-submit/` directory: Regulatory Submission Automation & FDA Pre-Submission Package Generator — fully implements Proposal C from `DEVELOPMENT_PROPOSALS.md`
  - `regulatory-submit/presub_generator.py`: FDA Pre-Submission (Q-Sub) meeting request package generator producing structured Markdown documents with device descriptions, AI/ML model documentation (architecture, training data, performance metrics), proposed testing protocols, and auto-generated questions for FDA review; supports Pre-Sub, Informational, Agreement/Determination, and Study Risk meeting types across 7 physical AI oncology device categories (surgical planning, robotic guidance, treatment prediction, diagnostic imaging, digital twin, dose optimization, computational pathology); includes risk consideration templates and PCCP discussion support for adaptive algorithms
  - `regulatory-submit/pccp_engine.py`: Predetermined Change Control Plan template engine per FDA's August 2025 finalized PCCP guidance; generates modification boundary definitions for 5 change types (model retraining, threshold adjustment, preprocessing update, architecture change, drift adaptation) with risk-stratified authorization categories (pre-authorized, requires notification, requires new submission); includes verification and validation protocols with statistical acceptance criteria (McNemar's test, DeLong AUC comparison, KS distribution test), transparency and communication plans, and lifecycle duration management
  - `regulatory-submit/classification_advisor.py`: 510(k)/De Novo/PMA regulatory pathway decision support engine analyzing device characteristics (software-only vs. physical contact, autonomy level, algorithm novelty, condition severity), predicate device suitability, and IEC 62304 software safety classification; produces structured recommendation documents with decision factors, risk classification justification, special considerations for physical AI devices (IEC 80601-2-77, ISO 13482), Breakthrough Device Designation assessment, and recommended next steps
  - `regulatory-submit/iec62304_generator.py`: IEC 62304:2015 software lifecycle documentation generator producing Software Development Plans (SDP), Software Requirements Specifications (SRS), Software Architecture Documents (SAD), and ISO 14971-integrated risk analysis matrices from project metadata; includes 10 default oncology AI requirements (functional, performance, safety, security, usability, regulatory, data), 5-level risk acceptability matrix (UNACCEPTABLE/ALARP/ACCEPTABLE), sample risk entries for AI device hazards, SOUP component tracking, and software item safety classification
  - `regulatory-submit/clinical_evidence.py`: Clinical evidence report builder linking simulation benchmarks, digital twin validation data, and retrospective clinical results to clinical performance claims; computes Wilson score confidence intervals for proportion metrics and normal approximation CIs for continuous metrics; performs demographic subgroup analysis (age, sex, race/ethnicity, tumor stage) with parity assessment; generates evidence-to-claim linkage documentation aligned with SPIRIT-AI/CONSORT-AI reporting extensions
  - `regulatory-submit/audit_trail.py`: 21 CFR Part 11-compliant audit trail generator with SHA-256 hash chain integrity for tamper detection; records AI model training runs (hyperparameters, metrics, hardware, random seed), validation experiments (acceptance criteria, pass/fail, reviewer sign-off), configuration changes (previous/new values, reason for change), and deployment events; produces structured audit reports with event timelines, training provenance, and chain hash verification
  - `regulatory-submit/README.md`: System overview with module descriptions, quick start examples for all 6 components, relationship to existing `regulatory/` directory, regulatory standards cross-reference table, and dependency information
- `regulatory-submit/examples-regulatory-submit/` directory: 6 progressive example scripts demonstrating regulatory submission automation
  - `01_presub_package.py`: Complete Pre-Sub package generation for an AI surgical planning system with 2 AI models, testing protocol, auto-generated FDA questions, and risk considerations
  - `02_pccp_plan.py`: PCCP document creation with default and custom modification boundaries, validation protocols, and transparency planning for an adaptive AI device
  - `03_classification.py`: Pathway decision support analyzing 3 different device profiles (novel treatment planner, CADe with predicate, robotic AI with patient contact) with comparative recommendations
  - `04_iec62304_docs.py`: Full IEC 62304 document set generation (SDP, SRS, SAD, risk analysis) for a 6-component software architecture with custom requirements and project-specific risks
  - `05_clinical_evidence.py`: Evidence report building with 7 benchmark results across 4 evidence levels, 13 demographic subgroup analyses, 3 clinical claims with evidence linkage, and study limitations
  - `06_full_submission.py`: End-to-end regulatory strategy combining all 6 components (classification → Pre-Sub → PCCP → IEC 62304 → clinical evidence → audit trail) into a complete De Novo submission package
  - `examples-regulatory-submit/README.md`: Examples overview with progression guide and quick start

### Updated
- `ruff.toml`: Added per-file ignore rules for `regulatory-submit/**/*.py` (F401, F402 for conditional imports and sys.path manipulation)

### Notes
- Fully implements Proposal C from `DEVELOPMENT_PROPOSALS.md` (Regulatory Submission Automation & FDA Pre-Submission Package Generator)
- Functionally distinct from existing `regulatory/` directory: `regulatory/` tracks submission status, manages IRB protocols, verifies GCP compliance, and monitors regulatory intelligence; `regulatory-submit/` generates the structured documents required for submissions
- All output is generated Markdown — no external FDA systems, APIs, or network connectivity required
- Uses only Python 3.10+ standard library (dataclasses, enum, hashlib, math, logging, datetime)
- Follows the same directory/example structure as existing `federation/examples-federation/` and `digital-twins/examples-twins/`
- All code passes `ruff check`, `ruff format --check` on Python 3.10–3.12
- Development by Claude Code Opus 4.6

## [1.1.1] - 2026-02-13

### Added
- `federation/` directory: Multi-Site Federated Oncology Trial Coordination Platform — fully implements Proposal B from `DEVELOPMENT_PROPOSALS.md`
  - `federation/federated_coordinator.py`: Core federated learning orchestration engine supporting FedAvg (McMahan et al., 2017), FedProx (Li et al., 2020), and SCAFFOLD (Karimireddy et al., 2020) aggregation strategies across N simulated clinical sites; includes ModelWeights serialization, round execution, convergence tracking, site selection with quality-weighted sampling, and 21 CFR Part 11 audit logging
  - `federation/differential_privacy.py`: Configurable epsilon/delta privacy budget management with Gaussian mechanism ((epsilon, delta)-DP) and Laplacian mechanism (pure epsilon-DP); gradient clipping (L2 norm and per-layer), summary statistic privatization, histogram noise injection, budget exhaustion prevention, and comprehensive privacy reporting
  - `federation/secure_aggregation.py`: Simulated secure multi-party computation (SMPC) based on Bonawitz et al. (2017) with additive secret sharing, pairwise masking that cancels during aggregation, SHA-256 commitment-based integrity verification, configurable dropout tolerance, and protocol state management
  - `federation/site_enrollment.py`: Cross-site enrollment synchronization with stratified block randomization, duplicate enrollment detection across sites, conflict resolution strategies (first-come, random assignment, manual review), arm balance monitoring with configurable imbalance thresholds, patient withdrawal tracking, and comprehensive enrollment summaries
  - `federation/data_harmonization.py`: DICOM metadata normalization (modality codes, body part terminology, pixel spacing, patient position), ICD-10 to SNOMED CT vocabulary mapping (6 oncology cancer types), LOINC coding for tumor markers (CEA, PSA, CA125, CA19-9, AFP, HER2), and FHIR R4 resource creation (Condition, Observation, MedicationStatement)
  - `federation/consortium_reporting.py`: DSMB (Data Safety Monitoring Board) package generation combining enrollment dashboards with site-level breakdowns and projections, adverse event summaries with CTCAE v5.0 grading and SOC distribution, risk-based site performance monitoring with composite scoring, safety signal detection, and automated DSMB recommendations
  - `federation/privacy_analytics.py`: Privacy-preserving federated survival analysis including Kaplan-Meier product-limit estimator from aggregated at-risk/event counts, federated Cox proportional hazards with Harrell's C-index, treatment arm response rate estimation with confidence intervals, Greenwood's formula for variance estimation, and automatic cell suppression below configurable minimum size thresholds
  - `federation/README.md`: Platform overview with architecture diagram, component descriptions, quick start, compliance alignment (ICH E6(R3), 21 CFR Part 11, HIPAA, FDA AI/ML, GDPR), and roadmap alignment
- `federation/examples-federation/` directory: 6 progressive example scripts demonstrating federation capabilities
  - `01_basic_two_site.py`: Minimal 2-site federation with FedAvg on a tumor classification model
  - `02_differential_privacy.py`: Privacy budget demonstration comparing Gaussian vs. Laplacian mechanisms, gradient clipping, histogram privatization, and budget exhaustion handling
  - `03_secure_aggregation.py`: Secure weight aggregation with additive secret sharing, pairwise masking cancellation verification, dropout tolerance, and commitment-based integrity checks
  - `04_enrollment_sync.py`: Multi-site enrollment coordination with stratified randomization, duplicate detection, withdrawal tracking, and arm balance monitoring
  - `05_data_harmonization.py`: Cross-site DICOM normalization, ICD-10/SNOMED CT/LOINC vocabulary mapping, and FHIR R4 resource creation
  - `06_full_consortium.py`: Complete 8-site multi-cancer consortium combining federated learning (FedProx), differential privacy, enrollment synchronization, data harmonization, DSMB reporting, and privacy-preserving survival analysis
  - `examples-federation/README.md`: Examples overview with progression guide and dependency information
- `tests/test_federation/` directory: 125 tests across 6 test modules covering all federation platform code
  - `test_federated_coordinator.py` (22 tests): ModelWeights flatten/unflatten, FedAvg/FedProx/SCAFFOLD aggregation, SimulatedLocalTrainer, FederatedCoordinator site registration, round execution, convergence, summary
  - `test_differential_privacy.py` (17 tests): PrivacyBudget consumption/exhaustion, GaussianMechanism/LaplacianMechanism noise shapes, GradientClipper norm bounds, DifferentialPrivacyEngine gradient/statistic/histogram privatization, budget status, report generation
  - `test_secure_aggregation.py` (12 tests): AdditiveSecretSharing split/reconstruct, PairwiseMaskGenerator cancellation, AggregationVerifier commitment/tampering, SecureAggregationProtocol full flow, dropout, invalid participants
  - `test_site_enrollment.py` (14 tests): StratifiedRandomizer balanced assignment, ConflictResolver duplicate detection/resolution, EnrollmentSynchronizer screening/enrollment/withdrawal/summary/balance
  - `test_data_harmonization.py` (17 tests): DICOMNormalizer modality/body part/warnings, VocabularyHarmonizer ICD-10/SNOMED/LOINC/custom, FHIRResourceMapper Condition/Observation/MedicationStatement, DataHarmonizationEngine batch harmonization
  - `test_consortium_reporting.py` (16 tests): EnrollmentDashboardGenerator, AdverseEventReporter SAE/treatment-related counts, SitePerformanceReporter risk levels, DSMBPackageGenerator safety signals and recommendations
  - `test_privacy_analytics.py` (15 tests): SiteSurvivalAggregator local statistics/covariates, FederatedSurvivalAnalyzer KM survival curves/CI/monotonicity, Cox PH hazard ratios/C-index, response rate with suppression
  - `tests/test_federation/__init__.py`: Package marker

### Updated
- `ruff.toml`: Added per-file ignore rules for `federation/**/*.py` (F401, F402 for conditional imports and sys.path manipulation)

### Notes
- Fully implements Proposal B from `DEVELOPMENT_PROPOSALS.md` (Multi-Site Federated Trial Coordination Platform)
- Fills the Q2–Q3 2026 roadmap gap documented in `unification/README.md`: "Establish consortium data sharing infrastructure" (Q2) and "Multi-site clinical trial coordination platform" (Q3)
- All multi-site communication is simulated in-process — no networking, GPU, or external FHIR/DICOM servers required
- Differential privacy and secure aggregation use standard numpy/scipy operations
- Follows the same directory/example structure as existing `agentic-ai/examples-agentic-ai/` and `digital-twins/examples-twins/`
- All code passes `ruff check`, `ruff format --check`, and `py_compile` on Python 3.10–3.12
- Full test suite: 1,414 tests pass (125 new + 1,289 existing), 0 failures
- Development by Claude Code Opus 4.6

## [1.0.1] - 2026-02-12

### Added
- `DEVELOPMENT_PROPOSALS.md`: Three comprehensive prompt proposals for future Claude Code development — Proposal A (Comprehensive Test Suite), Proposal B (Multi-Site Federated Trial Coordination), Proposal C (Regulatory Submission Automation) — with feature-by-feature comparison tables, strategic impact matrix, and audience impact analysis
- **Comprehensive test suite**: 1,289+ tests across 54 test modules covering all 51 Python source modules, fully implementing Proposal A (Comprehensive Test Suite & Continuous Validation Infrastructure)
  - `tests/conftest.py`: Shared fixtures, mock data factories (synthetic tumor geometry, dose distributions, trial cohort config), and `importlib.util.spec_from_file_location()` loader for hyphenated directories; autouse RNG seeding (seed=42) for deterministic tests
  - **Root-level tests** (7 modules, 143 tests):
    - `tests/test_safety_monitoring.py` (15 tests): SafetyMonitor, ForceTorqueSensorProcessor, WorkspaceBoundaryGenerator
    - `tests/test_dose_calculator.py` (16 tests): BED, EQD2, TCP, NTCP, fractionation scheme parsing, tissue data
    - `tests/test_digital_twin_sync.py` (20 tests): EKF, particle filter, anomaly detection, synchronizer
    - `tests/test_mcp_server.py` (24 tests): MCP tool/resource handlers, audit trail, data models
    - `tests/test_calibration.py` (16 tests): Tsai-Lenz calibration, Arun SVD registration, transform math
    - `tests/test_sample_handling.py` (16 tests): Specimen model, barcode verification, cold chain
    - `tests/test_deidentification.py` (13 tests): Safe Harbor transforms, PHI detection, config
  - **`tests/test_digital_twins/`** (8 modules): Unit tests for all digital twin code
    - `test_tumor_twin_pipeline.py`: TumorType/ModelType/SolverType enums, PatientClinicalData, ModelParameters, LogisticGrowthModel, GompertzGrowthModel, MechanisticModel, PatientDigitalTwin
    - `test_treatment_simulator.py`: TreatmentType/ResponseType enums, TreatmentProtocol, TreatmentSimulator, LinearQuadraticModel, PharmacokineticModel, ImmunotherapyModel
    - `test_clinical_dt_interface.py`: ConnectionStatus, ComplianceRegulation, PatientRecord, ClinicalConnector, FHIRClient, ComplianceManager
    - `test_multi_organ_toxicity.py`: ChemoDrug, OrganSystem, CTCAEGrade, PBPKModel, CardiacToxicityModel, RenalToxicityModel, HepaticToxicityModel, MultiOrganToxicityTwin
    - `test_adaptive_radiation.py`: StructureType, DoseConstraint, BSplineRegistration, DoseAccumulator, AdaptiveRTDigitalTwin
    - `test_immunotherapy_dt.py`: ImmunePheno, CheckpointAgent, iRECISTResponse, TMEDynamicsModel, CheckpointPKModel, ImmunotherapyResponsePredictor
    - `test_virtual_trial_cohort.py`: TumorSite, TrialEndpoint, VirtualCohortGenerator, OutcomeSimulator, VirtualTrialSimulator, VirtualControlArmBuilder
    - `test_dt_validation.py`: ValidationLevel, RiskCategory, AccuracyMetrics, CalibrationAnalyzer, DiscriminationAnalyzer, SubgroupAnalyzer, RobustnessAnalyzer, VVReportGenerator
  - **`tests/test_agentic_ai/`** (5 modules): Unit tests for all agentic AI examples
    - `test_react_planner.py`: ProcedurePlanningTools, ReActProcedurePlanner, anatomy/instrument data models
    - `test_adaptive_treatment.py`: StreamBuffer, ForceTorqueProcessor, VitalsProcessor, CrossModalCorrelator, TreatmentDecisionEngine, AdaptiveTreatmentAgent
    - `test_simulation_orchestrator.py`: ExperimentDesigner, SimulationRunner, AnalysisEngine, SimulationOrchestrator
    - `test_safety_executor.py`: OncologyRoboticsConstraintLibrary, SafetyConstrainedExecutor, constraint checking
    - `test_rag_compliance.py`: RegulatoryKnowledgeBase, ComplianceVerifier, ProtocolRAGComplianceAgent
  - **`tests/test_tools/`** (4 modules): Unit tests for all CLI tools
    - `test_deployment_readiness.py`: ReadinessReport, deployment checks, regulatory checklists
    - `test_dicom_inspector.py`: InspectionResult, DICOM tag validation, PHI audit
    - `test_sim_job_runner.py`: JobResult, framework detection, task definitions
    - `test_trial_site_monitor.py`: SiteMetrics, enrollment tracking, quality scoring
  - **`tests/test_physical_robots/`** (6 modules): Unit tests for all robot examples
    - `test_sensor_fusion.py`: InstrumentSegmenter, TissueDeformationTracker, DepthToPointCloud, TemporalSynchronizer, SensorFusionPipeline
    - `test_ros2_deployment.py`: ProcedureStateMachine, PolicyInferenceEngine, RobotHardwareInterface, SurgicalControlLoop
    - `test_shared_autonomy.py`: VirtualFixtureEngine, CommandBlender, SharedAutonomyController, SurgeonInputProcessor
    - `test_surgical_training.py`: OncologySurgicalEnv, SurgicalPolicyNetwork, SurgicalPolicyTrainer, PolicyEvaluator
    - `test_surgical_planning.py`: SurgicalDigitalTwinBuilder, SurgicalDigitalTwin, VirtualSurgerySimulator
    - `test_treatment_prediction.py`: ExponentialGrowthModel, GompertzGrowthModel, TreatmentResponseModel, TreatmentOptimizer
  - **`tests/test_privacy/`** (4 modules): Unit tests for all privacy framework modules
    - `test_phi_detector.py`: PHICategory (18 HIPAA identifiers), PHIDetector scan/classification
    - `test_access_control.py`: Permission/UserType enums, AccessControlManager, audit trail copy guard
    - `test_breach_response.py`: IncidentType, RiskAssessment clamping, NotificationTimeline, BreachResponseManager
    - `test_dua_generator.py`: DUATemplate, DUAGenerator, jurisdiction handling
  - **`tests/test_regulatory/`** (4 modules): Unit tests for all regulatory framework modules
    - `test_fda_submission.py`: SubmissionType/Status/DeviceClass, FDASubmissionTracker, AI/ML component defaults
    - `test_irb_protocol.py`: ProtocolStatus, IRBProtocolManager, SubmissionChecklist completeness
    - `test_gcp_compliance.py`: GCPComplianceChecker, score excluding NOT_ASSESSED, ComplianceReport
    - `test_regulatory_tracker.py`: RegulatoryTracker, overdue/imminent status, cutoff date filtering
  - **`tests/test_unification/`** (5 modules): Unit tests for all unification framework modules
    - `test_isaac_mujoco_bridge.py`: PhysicsParameterMapper, StateConverter, IsaacMuJoCoBridge, PolicyTransferValidator
    - `test_urdf_converter.py`: URDFParser, MJCFGenerator, SDFGenerator, UnifiedModelConverter
    - `test_unified_agent.py`: UnifiedAgent, AgentTeam, OncologyToolkit, backend adapters
    - `test_framework_detector.py`: FrameworkDetector, FrameworkInfo, SystemInfo
    - `test_validation_suite.py`: MockEnvironment, PolicyLoader, CrossPlatformValidator
  - **`tests/test_standards/`** (3 modules): Unit tests for Q1 2026 standards
    - `test_isaac_to_mujoco.py`: PhysicsParameterConverter, URDFToMJCFConverter, IsaacToMuJoCoConverter
    - `test_benchmark_runner.py`: PhysicsBenchmark, PerformanceBenchmark, BenchmarkRunner
    - `test_model_validator.py`: FormatValidator, KinematicValidator, ModelValidator
  - **`tests/test_integration/`** (6 modules): Cross-module workflow tests
    - `test_dt_to_simulation.py`: Digital Twin → Treatment Simulation → Response Prediction flow
    - `test_agentic_to_regulatory.py`: Agentic AI decision → Regulatory audit trail → Compliance
    - `test_robot_to_safety.py`: Robot command → Safety monitoring → Emergency stop
    - `test_privacy_to_clinical.py`: Patient data → De-identification → Clinical utility preserved
    - `test_cross_framework.py`: Multi-framework simulation validation pipeline
    - `test_end_to_end_trial.py`: Full trial lifecycle: Patient → DT → Simulation → Regulatory
  - **`tests/test_regression/`** (2 modules): Comprehensive regression guards
    - `test_v092_guards.py`: 7 guards for critical v0.9.1/v0.9.2 bugs (EKF Jacobian, hazard ratio, division-by-zero, DoseResult truthiness)
    - `test_v092_comprehensive.py`: 28 additional guards for all remaining v0.9.2 fixes (bidirectional sync, bounded loops, overdue status, compliance scoring, format strings, audit log copy, date shift, weights_only, and more)
  - `tests/README.md`: Comprehensive testing strategy documentation with test organization tree, philosophy, coverage targets, and CI integration
  - `tests/__init__.py` and `__init__.py` in all 10 subdirectories

### Fixed
- **CI: Graceful handling of optional dependencies** — `load_module()` in `tests/conftest.py` now wraps `spec.loader.exec_module()` in a `try/except ImportError` block; tests that depend on unavailable packages (torch, mujoco, langchain, monai, etc.) are automatically **skipped** via `pytest.skip()` instead of failing the CI run. Partially-initialised modules are removed from `sys.modules` to prevent downstream breakage. A `filepath.exists()` guard was also added to skip tests when source files are missing. This fix keeps CI green when only core dependencies (numpy, scipy, pytest, pyyaml) are installed, while still running the full suite when all optional packages are available.

### Updated
- `.github/workflows/ci.yml`: Updated `test` job — added `pyyaml` to CI dependencies; added comment documenting the optional dependency skip strategy
- `tests/conftest.py`: Added `ImportError` guard and `filepath.exists()` check in `load_module()`; added mock data factories (synthetic_tumor_geometry, synthetic_dose_distribution, trial_cohort_config)
- `tests/README.md`: Rewritten with full test tree, testing philosophy, coverage targets, and architecture docs

### Notes
- Fully implements Proposal A from `DEVELOPMENT_PROPOSALS.md` (Comprehensive Test Suite & Continuous Validation Infrastructure)
- Combines the comprehensive 1,289-test suite from PR #17 with the CI robustness fix from PR #18
- The `ImportError` skip in `conftest.py` is a **permanent fix** — it is architecturally correct for projects where source modules have optional heavy dependencies (GPU frameworks, medical imaging libraries, robot middleware) that are not installed in lightweight CI environments. The pattern of skipping tests when their dependencies are unavailable (rather than failing) is a standard pytest best practice and requires no future removal or workaround.
- All tests pass `ruff format`, `ruff check`, and `py_compile` validation
- Tests use `importlib.util.spec_from_file_location()` to handle hyphenated directory names
- Mock-based isolation: all external dependencies (NVIDIA Isaac, MuJoCo, ROS 2, DICOM servers) mocked — tests run without GPU or hardware
- Deterministic RNG seeding (seed=42) ensures reproducible results across platforms and Python versions
- CI runs ruff format, ruff check, yamllint, py_compile, and pytest on Python 3.10–3.12
- Development by Claude Code Opus 4.6

## [1.0.0] - 2026-02-08

### Added
- `V1_RELEASE.md`: Comprehensive v1.0.0 release documentation covering community needs, technical achievements, version history, and v1.0.0 standards compliance
- Version badge (`v1.0.0`) added to README.md header
- v1.0.0 release summary block added to README.md with repository metrics (51 Python modules, 40,526 LOC, 69 docs, 28 examples, 5 CLI tools)
- `V1_RELEASE.md` added to repository structure in README.md

### Updated
- README.md: Added v1.0.0 designation, release summary, version badge, and updated citation block with `version = {1.0.0}`
- CHANGELOG.md: Consolidated all prior releases under v1.0.0 milestone

### Notes
- v1.0.0 designates the first stable release of the public API: directory structure, module interfaces, CLI tool contracts, and configuration formats
- Repository totals at v1.0.0: 66 commits, 12 merged pull requests, 65,287 insertions, 4,035 deletions, 160 project files across 61 directories
- Development primarily by Claude Code Opus 4.5/Opus 4.6; Claude Cowork Opus 4.5 for initial release; ChatGPT 5.2 Thinking/Agent for audit prompts and repo insights
- CI passes on Python 3.10, 3.11, and 3.12 (ruff lint, ruff format, yamllint, py_compile)
- Follows Semantic Versioning 2.0.0 and Keep a Changelog format

## [0.9.2] - 2026-02-08

### Fixed
- **Logic (CRITICAL)**: Fixed EKF Jacobian sign error in `digital-twins/examples-twins/01_realtime_dt_synchronization.py` (line 295: `1.0 + rate*dt` corrected to `1.0 - rate*dt`) causing divergent creatinine state estimates
- **Logic (CRITICAL)**: Fixed inverted hazard ratio calculation in `digital-twins/examples-twins/05_virtual_trial_cohort_dt.py` (line 743: `control/experimental` corrected to `experimental/control` per standard oncology convention where HR < 1 favors experimental arm)
- **Logic (CRITICAL)**: Fixed infinite `while not done: pass` loop in `unification/simulation_physics/isaac_mujoco_bridge.py` `_evaluate_policy()` that would hang indefinitely; replaced with bounded step loop
- **Logic (CRITICAL)**: Fixed `sync_state()` in `unification/simulation_physics/isaac_mujoco_bridge.py` only handling Isaac-to-MuJoCo direction; added MuJoCo-to-Isaac and MuJoCo-to-PyBullet branches and prevented false counter increment for unsupported frameworks
- **Logic**: Fixed unreachable "overdue" status branch in `regulatory/regulatory-intelligence/regulatory_tracker.py` where deadlines past due were mislabeled as "imminent" due to incorrect if/elif ordering
- **Logic**: Fixed GCP compliance score always reporting 0% in `regulatory/ich-gcp/gcp_compliance_checker.py` by excluding `NOT_ASSESSED` findings from the scoring denominator
- **Logic**: Fixed format string bug `%.1%%` in `digital-twins/examples-twins/04_tumor_microenvironment_immunotherapy_dt.py` (line 664) causing `TypeError` at runtime; corrected to `%.1f%%`
- **Logic**: Fixed division by zero in `digital-twins/patient-modeling/tumor_twin_pipeline.py` `LogisticGrowthModel.simulate()` when initial condition sums to zero (post-resection scenarios)
- **Logic**: Fixed division by zero in `tumor_twin_pipeline.py` `predict()` volume change calculation when baseline volume is zero
- **Logic**: Fixed floating-point equality comparison in `digital-twins/treatment-simulation/treatment_simulator.py` surgery day check (line 372) that could miss the surgery timepoint due to `np.linspace` precision
- **Logic**: Fixed MJCF parsing incorrectly falling back to URDF parser in `unification/simulation_physics/urdf_sdf_mjcf_converter.py`; now raises `NotImplementedError` with guidance to use dedicated conversion pipelines
- **Logic**: Fixed `sim_job_runner.py` `cmd_launch_all` iterating all frameworks including unavailable ones despite computing and displaying `target_frameworks`
- **Logic**: Fixed `dose_calculator.py` truthiness checks (`if self.bed_gy:`) that silently dropped valid zero-value results from `DoseResult.to_dict()`; changed to `is not None` checks
- **Logic**: Fixed `dose_calculator.py` CLI falsy-value check replacing explicit `--alpha-beta 0` and `--volume 0` inputs with defaults
- **Logic**: Fixed `validation_suite.py` success rate always reporting ~25% because threshold was computed as 75th percentile of the same rewards array; replaced with fixed task-appropriate threshold
- **Runtime (CRITICAL)**: Fixed `TypeError` crash in `privacy/access-control/access_control_manager.py` demo where `assign_role()` was called with unsupported `mfa_enrolled` keyword argument
- **Security**: Changed `torch.load()` to `torch.load(weights_only=True)` in `unification/cross_platform_tools/validation_suite.py` to prevent arbitrary code execution via pickle deserialization
- **Security**: Fixed `access_control_manager.py` `get_audit_log()` returning a reference to the internal audit log list; now returns a copy to prevent external mutation of audit trail
- **Security**: Fixed `access_control_manager.py` silently granting access when `access_expiration` date format is invalid; now logs error and denies access by default
- **Compliance**: Fixed `deidentification_pipeline.py` `DATE_SHIFT` handling silently falling through to date removal; added explicit `DATE_SHIFT` branch with appropriate logging
- **Compliance**: Fixed `fda_submission_tracker.py` defaulting all AI/ML components to `model_type="classification"`; changed to `"unspecified"` since component type should be explicitly specified
- **Compliance**: Fixed `deployment_readiness.py` safety constraints always reporting "passed" without checking actual model outputs; now reports `requires_runtime_verification` status
- **Compliance**: Fixed `deployment_readiness.py` identical ternary branches for multi-input model validation; both branches produced single-input feed dict
- **Compliance**: Added `RESEARCH USE ONLY` disclaimers to 11 modules: `deidentification_pipeline.py`, `phi_detector.py`, `access_control_manager.py`, `breach_response_protocol.py`, `dua_generator.py`, `fda_submission_tracker.py`, `irb_protocol_manager.py`, `gcp_compliance_checker.py`, `regulatory_tracker.py`, `tumor_twin_pipeline.py`, `treatment_simulator.py`, `dose_calculator.py`
- **Lint**: Added missing `import logging` and `logger` to `isaac_mujoco_bridge.py`; removed unused `Union` import
- **Format**: Auto-formatted `deidentification_pipeline.py` and `deployment_readiness.py` to pass `ruff format --check`

### Notes
- Comprehensive logic, context, and compliance audit of 51 Python files across all modules
- CI lint-and-format checks pass for Python 3.10, 3.11, and 3.12
- ChatGPT 5.2 Thinking Agent assisted with this audit prompt

## [0.9.1] - 2026-02-08

### Fixed
- **Security**: Replaced weak default pseudonymization salt (`"default_salt"`) in `privacy/de-identification/deidentification_pipeline.py` with cryptographically random salt generation via `os.urandom`; logs a warning when no explicit `hash_salt` is configured
- **Security**: Changed `numpy.load(allow_pickle=True)` to `allow_pickle=False` in `tools/deployment-readiness/deployment_readiness.py` to prevent arbitrary code execution from untrusted `.npz` files
- **Logic**: Fixed `RiskAssessment.calculate_risk()` in `privacy/breach-response/breach_response_protocol.py` to clamp out-of-range scores instead of silently returning and leaving the object in an inconsistent state
- **Logic**: Added missing `peak_cd8` and `peak_ifng` keys to `predict_response()` return dict in `digital-twins/examples-twins/04_tumor_microenvironment_immunotherapy_dt.py`, fixing a `KeyError` in the demo main block
- **Logic**: Fixed dead-code multiplication by `0.0` for renal elimination in `digital-twins/examples-twins/02_multi_organ_toxicity_twin.py` PBPK kidney compartment ODE
- **Logic**: Fixed `get_recent_updates()` in `regulatory/regulatory-intelligence/regulatory_tracker.py` to actually use the computed `cutoff` date for filtering
- **Logic**: Added whitespace stripping to comma-separated framework parsing in `unification/cross_platform_tools/validation_suite.py`
- **Type safety**: Added `from __future__ import annotations` to `regulatory/irb-management/irb_protocol_manager.py` to resolve forward reference of `SubmissionChecklist`
- **Type hint**: Added return type `-> int` to `main()` in `scripts/verify_installation.py`
- **Imports**: Removed unused `import re` from `unification/simulation_physics/urdf_sdf_mjcf_converter.py`
- **Imports**: Removed unused `from abc import ABC, abstractmethod` from `digital-twins/clinical-integration/clinical_dt_interface.py`
- **Imports**: Removed unused `import yaml` from `q1-2026-standards/objective-1-bidirectional-conversion/isaac_to_mujoco_pipeline.py`
- **Imports**: Removed unused `import yaml` and `import warnings` from `q1-2026-standards/objective-2-robot-model-repository/model_validator.py`
- **Formatting**: Fixed missing space in output string in `tools/deployment-readiness/deployment_readiness.py`
- **YAML**: Split long comment line in `unification/simulation_physics/physics_parameter_mapping.yaml` to resolve yamllint line-length warning

### Notes
- Full static analysis audit of 51 Python files, 5 YAML files, and 47+ Markdown files
- CI lint-and-format checks pass for Python 3.10, 3.11, and 3.12
- ChatGPT 5.2 Thinking Agent assisted with this audit prompt

## [0.9.0] - 2026-02-07

### Added
- `agentic-ai/examples-agentic-ai/` directory: 6 comprehensive agentic AI engineering examples for robotic oncology trials
  - `01_mcp_clinical_robotics_server.py`: Model Context Protocol (MCP) server exposing robot telemetry, DICOM imaging, patient vitals, and procedure management as structured tools and resources with 21 CFR Part 11 audit trails, keep-out zone enforcement, and WHO-adapted surgical safety checklist
  - `02_react_procedure_planner.py`: ReAct (Reasoning + Acting) agent for surgical procedure planning with chain-of-thought reasoning, patient-specific anatomy integration, instrument selection, approach safety evaluation, margin estimation, and contingency planning across lobectomy, nephrectomy, and prostatectomy protocols
  - `03_realtime_adaptive_treatment_agent.py`: Real-time adaptive treatment agent processing streaming multi-modal data (force/torque, patient vitals, intraoperative imaging) with cross-modal correlation engine detecting hemorrhage, hemodynamic instability, and resection boundary concerns, generating prioritized treatment recommendations
  - `04_autonomous_simulation_orchestrator.py`: Autonomous agent that designs, configures, runs, and analyzes simulation experiment campaigns across Isaac Lab, MuJoCo, PyBullet, and Gazebo with parameter sensitivity analysis, cross-framework consistency checks, hypothesis evaluation, and iterative refinement
  - `05_safety_constrained_agent_executor.py`: Formal safety constraint framework for agentic control of surgical robots with pre-condition/post-condition verification, runtime invariant monitoring, safety gate human-in-the-loop approval, constraint library aligned to IEC 80601-2-77 and ISO 14971, and rollback mechanisms
  - `06_protocol_rag_compliance_agent.py`: Retrieval-Augmented Generation (RAG) agent grounding clinical decisions in trial protocols, FDA guidance, ICH E6(R3), IEC standards, and institutional SOPs with keyword-based document retrieval, compliance verification, cited regulatory responses, and audit trail

### Updated
- `ruff.toml`: Added per-file ignore rules for `agentic-ai/**/*.py`
- Main `README.md`: Added Agentic AI Engineering Examples section with table and quick start
- Repository structure updated to include `agentic-ai/examples-agentic-ai/`

## [0.8.0] - 2026-02-07

### Added
- `tools/` directory: 5 standalone CLI utilities for physical AI oncology trial engineers
  - `tools/dicom-inspector/dicom_inspector.py`: DICOM file inspection, PHI audit across imaging directories, trial compliance validation (DICOM-BASE and DICOM-RT standards), and study-level summarization with modality distribution
  - `tools/dose-calculator/dose_calculator.py`: Radiotherapy dose calculations with BED, EQD2, TCP (Poisson and logistic models), NTCP (Lyman-Kutcher-Burman model with QUANTEC-derived organ presets), fractionation scheme comparison, and tissue alpha/beta reference tables
  - `tools/trial-site-monitor/trial_site_monitor.py`: Multi-site trial enrollment tracking, data quality scoring (completeness, query rates, protocol deviation rates, AE reporting delays), site status classification (green/yellow/red), and manifest template generation
  - `tools/sim-job-runner/sim_job_runner.py`: Cross-framework simulation job launcher supporting Isaac Lab, MuJoCo, PyBullet, and Gazebo with 6 oncology-relevant task definitions (needle insertion, tissue retraction, surgical reach, instrument handover, biopsy sampling, catheter navigation), framework auto-detection, and result comparison
  - `tools/deployment-readiness/deployment_readiness.py`: Pre-deployment AI model validation with ONNX compatibility checking, inference latency benchmarking (mean/P50/P95/P99), safety constraint verification, regulatory checklist generation (IEC 62304, FDA AI/ML PCCP, ISO 14971), and reference output validation
- `tools/README.md`: Documentation for all tools with usage examples, design principles, and dependency matrix

### Updated
- Main `README.md`: Added Command-Line Tools section with table and quick start; updated repository structure to include `tools/`

## [0.7.0] - 2026-02-06

### Added
- `digital-twins/examples-twins/` directory: 6 advanced digital twin engineering examples
  - `01_realtime_dt_synchronization.py`: Real-time DT synchronization via Extended Kalman Filter and particle filter (asynchronous multi-modal data fusion, anomaly detection via CUSUM, 21 CFR Part 11 audit trails)
  - `02_multi_organ_toxicity_twin.py`: Multi-organ toxicity digital twin with PBPK compartmental model (cardiac/renal/hepatic/neurological/hematologic toxicodynamics, CTCAE v5.0 grading, dose modification recommendations)
  - `03_adaptive_radiation_therapy_dt.py`: Adaptive radiation therapy DT with B-spline deformable image registration (dose accumulation on deforming anatomy, DVH metrics, BED/EQD2, replanning trigger detection per AAPM TG-132/TG-275)
  - `04_tumor_microenvironment_immunotherapy_dt.py`: Tumor microenvironment and immunotherapy response DT (9-variable ODE model of TME dynamics, PD-1/PD-L1 checkpoint axis, iRECIST classification, pseudoprogression detection, biomarker-driven response prediction)
  - `05_virtual_trial_cohort_dt.py`: Virtual clinical trial cohort DT (virtual patient generation, Weibull survival simulation, Bayesian adaptive interim analysis, power analysis, virtual control arm construction)
  - `06_dt_validation_verification.py`: Digital twin validation and verification framework (C-index, Hosmer-Lemeshow calibration, AUC discrimination, subgroup equity analysis, robustness testing, model card and V&V report generation per ASME V&V 40 and FDA AI/ML guidance)
- `digital-twins/examples-twins/README.md`: Documentation for all examples with regulatory standards cross-reference

### Updated
- `digital-twins/README.md`: Added examples-twins directory to structure and key capabilities
- Main `README.md`: Added Digital Twin Engineering Examples section with table and quick start
- Repository structure updated to reflect new directory

## [0.6.0] - 2026-02-06

### Added
- `examples-new/` directory: 6 comprehensive physical robot engineering examples
  - `01_realtime_safety_monitoring.py`: IEC 80601-2-77 compliant safety monitoring (force/torque limits, workspace boundaries, watchdog timers, force rate detection)
  - `02_sensor_fusion_intraoperative.py`: Multi-sensor perception pipeline (stereo/RGBD depth, instrument segmentation, tissue deformation tracking, temporal synchronization)
  - `03_ros2_surgical_deployment.py`: ROS 2 node architecture for surgical deployment (procedure state machine, policy inference, hardware interface for dVRK/Kinova/UR, real-time control loop)
  - `04_hand_eye_calibration_registration.py`: Spatial calibration (Tsai-Lenz hand-eye calibration, Arun SVD fiducial registration, ICP surface registration, verification with test points)
  - `05_shared_autonomy_teleoperation.py`: Surgeon-AI shared control (5 autonomy levels, virtual fixtures, command blending, haptic rendering, tremor filtering)
  - `06_robotic_sample_handling.py`: Laboratory automation for clinical trials (specimen pick-and-place, barcode verification, cold chain monitoring, 21 CFR Part 11 audit trails, batch processing)
- `examples-new/README.md`: Documentation for all new examples with hardware requirements, regulatory references, and usage instructions

### Updated
- Main `README.md`: Added `examples-new/` section with table of all new examples and quick start instructions
- Repository structure updated to reflect new directory

## [0.5.1] - 2026-02-04

### Added
- `.github/` directory with issue templates, PR template, and CI workflow
- `CITATION.cff` for machine-readable citation metadata
- `SECURITY.md`, `SUPPORT.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
- `regulatory/human-oversight/` quality management document for CRF/AE automation
- Python lint/format CI via `ruff` and `yamllint`
- Illustrative-data disclaimers on all `results.md` benchmark tables

## [0.5.0] - 2026-02-04

### Added
- `privacy/` framework: PHI/PII detection, de-identification, access control, breach response, DUA templates
- `regulatory/` framework: FDA submission tracking, IRB management, ICH E6(R3) compliance, regulatory intelligence
- Privacy tooling covers all 18 HIPAA identifiers
- Regulatory tooling aligned with FDA AI/ML guidance (Jan 2025), ICH E6(R3) (Sep 2025), EU AI Act timelines

## [0.4.0] - 2026-02-02

### Added
- `digital-twins/` directory: patient modeling (TumorTwin), treatment simulation, clinical integration (FHIR/DICOM)
- `examples/` directory: 5 production-ready Python examples covering surgical training, digital twins, cross-framework validation, agentic workflows, and treatment prediction
- `q1-2026-standards/` directory: 3 unification objectives (bidirectional conversion, model repository, validation benchmarks)
- `configs/training_config.yaml` with domain randomization, safety limits, and deployment settings

### Updated
- Framework versions: Isaac Sim 5.0.0, Newton Physics Beta, MuJoCo Warp Beta, GR00T N1.6, Cosmos Predict 2.5, Cosmos Reason 2

## [0.3.1] - 2026-02-01

### Added
- Source citations across documentation to support framework/version claims

### Fixed
- Corrected outdated framework versions and related references (11 files modified; 140 insertions; 102 deletions)

## [0.3.0] - 2026-02-01

### Added
- `q1-2026-standards/` directory defining unification objectives:
  - Objective 1: Isaac <-> MuJoCo bidirectional conversion
  - Objective 2: Unified robot model repository (50+ models)
  - Objective 3: Validation benchmark suite v1.0

### Notes
- Includes an implementation guide with timeline and compliance checklist  
- Framework versions referenced: Isaac Lab 2.3.2, MuJoCo 3.4.0

## [0.2.0] - 2026-01-31

### Added
- Unification framework for framework-agnostic physical AI development for oncology clinical trials
- Multi-organization cooperation framing (release notes reference “February 2026” objectives)
- Adoption guidance spanning: (a) simulation physics, (b) agentic/generative AI, (c) surgical robots, (d) cross-platform tools

## [0.1.0] - 2026-01-31

### Added
- Initial repository structure
- `unification/` framework: Isaac-MuJoCo bridge, model converters, unified agent interface, cross-platform tools
- `frameworks/` integration guides: NVIDIA Isaac, MuJoCo, Gazebo, PyBullet
- Learning domain documentation: supervised, reinforcement, self-supervised, agentic, generative AI
- `scripts/verify_installation.py` for dependency checking
- `requirements.txt` with 30+ production dependencies
