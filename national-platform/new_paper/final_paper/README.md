# National Platform for Physical AI Oncology Trials — LaTeX Source Guide

**Version:** 3.0.0 (Draft 1.0, March 28, 2026)
**Author:** CEO Kevin Kawchak, ChemicalQDevice
**DOI:** [10.5281/zenodo.19244918](https://doi.org/10.5281/zenodo.19244918)
**License:** MIT
**Template:** Adapted from University of Groningen MSc AI/CCS Master's Thesis Template (Overleaf, CC BY 4.0), original by Manvi Agarwal (2020).

---

## Purpose of This Document

This README is designed as a comprehensive reference for **Claude Code Opus 4.6** (1M context) to understand the structure, content, interdependencies, and conventions of all LaTeX source files in this paper. The paper is a 186-page end-to-end resource for the pharmaceutical and regulatory industries, providing adapted clinical trial regulation and guidelines to accelerate oncology drug approval using Physical AI (robotics + AI).

---

## Compilation

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires: `lmodern`, `microtype`, `subcaption`, `amsmath`, `graphicx`, `fancyhdr`, `float`, `longtable`, `booktabs`, `array`, `tabularx`, `enumitem`, `textcomp`, `geometry`, `tocbibind`, `url`, `hyperref`, `xcolor`. All standard in TeX Live / MiKTeX.

---

## File Inventory

| File | Lines | Role |
|------|------:|------|
| `main.tex` | ~130 | Entry point. Document class, packages, page style import, section ordering via `\input{}` |
| `page_styles.tex` | ~30 | Defines 5 `fancyhdr` page styles: `coverpage`, `body`, `contents`, `appendix`, `acknowledgements` |
| `references.bib` | ~230 | 34 BibTeX entries (IEEE style). Mix of federal regulations, Zenodo papers, AI model refs, NCI/FDA resources |
| `sections/cover_page.tex` | 50 | Title, author, DOI link, legal notices, disclaimers, date |
| `sections/contents.tex` | 5 | Table of contents with "Page" header |
| `sections/source_documents.tex` | 146 | Catalog of all 12 source documents with descriptions and section cross-references |
| `sections/executive_summary.tex` | 154 | 7-theme summary (Regulatory, Standards, Evidence, Patient, Infrastructure, Economic, Governance) + 1 figure |
| `sections/introduction.tex` | 551 | §1: Background, definition of Physical AI, case for national platform, current AI landscape |
| `sections/gov_framework.tex` | 712 | §2: Constitutional foundation, legislative/executive/judicial branch operations for PAI governance |
| `sections/regulatory_landscape.tex` | 586 | §3: Dual-jurisdiction (California + federal) regulatory analysis |
| `sections/ich_e6r3_adaptation.tex` | 379 | §4: Adapted ICH E6(R3) GCP — principles, investigator/sponsor responsibilities, data governance |
| `sections/cfr50_adaptation.tex` | 670 | §5: Adapted 21 CFR Part 50 — informed consent, safety matrix, pediatric protections, classification |
| `sections/cfr312_adaptation.tex` | 501 | §6: Adapted 21 CFR Part 312 — IND requirements, Phase 0, Subpart J, safety reporting |
| `sections/psl_usl_standards.tex` | 572 | §7: PSL (3-dimension site compliance) + USL (4-dimension robot readiness) scoring frameworks |
| `sections/site_establishment.tex` | 514 | §8: 11-document site package (3 bills, regulations, codes, SOPs, emergency plans) + 1 figure |
| `sections/patient_journey.tex` | 481 | §9: 10-stage single-patient simulation (Stage IIIB NSCLC, 1,120 days) + 1 figure |
| `sections/patient_instructions.tex` | 401 | §10: Patient-facing documentation for 10 robot types |
| `sections/national_mcp.tex` | 607 | §11: 5-server MCP architecture, 23 tools, safety modules, deployment topology |
| `sections/federated_learning.tex` | 429 | §12: Privacy-preserving multi-site FL framework, 5 pillars, triple AI peer review |
| `sections/financial_analysis.tex` | 536 | §13: Cost analysis, timeline compression, ROI, national economic projections |
| `sections/implementation_strategy.tex` | 457 | §14: 3-phase rollout (California → multi-site → nationwide), regulatory pathway, workforce |
| `sections/discussion.tex` | 501 | §15: Synthesis across all sections, paradigm shift argument, limitations, future directions |
| `sections/conclusion.tex` | 454 | §16: 7 key findings, implications for pharma/regulators/patients, call to action + 1 figure |
| `sections/appendices.tex` | 314 | Appendices A–E: source file directory, glossary, cross-reference matrix, simulation data, scoring |

**Total:** ~9,020 lines across 22 `.tex` files + 1 `.bib` file.

---

## Document Architecture

### main.tex — The Orchestrator

`main.tex` is the single entry point. It does three things:

1. **Preamble:** Loads all packages, sets geometry (`a4paper, 12pt, twoside`, margins 2cm L/R, 1.5cm top, 1.8cm bottom), configures hyperlinks (all black, no borders), enables `\sloppy` mode with `\emergencystretch=3em` for long URLs/paths.
2. **Page styles:** Imports `page_styles.tex` via `\input{page_styles}`.
3. **Section ordering:** Includes all 20 section files in sequence. Unnumbered sections (Source Documents, Executive Summary, Appendices) use `\section*{}` with manual `\addcontentsline`. Numbered sections (§1–§16) use `\section{}` with `\label{sec:...}`.

**Section labels defined in main.tex** (these are the targets for all `\ref{sec:...}` cross-references):

| Label | Section |
|-------|---------|
| `sec:introduction` | §1 Introduction |
| `sec:gov-framework` | §2 Government Framework |
| `sec:regulatory-landscape` | §3 Regulatory Landscape |
| `sec:ich-e6r3` | §4 ICH E6(R3) Adaptation |
| `sec:cfr50` | §5 21 CFR Part 50 Adaptation |
| `sec:cfr312` | §6 21 CFR Part 312 Adaptation |
| `sec:psl-usl` | §7 PSL/USL Standards |
| `sec:site-establishment` | §8 Site Establishment |
| `sec:patient-journey` | §9 Patient Journey |
| `sec:patient-instructions` | §10 Patient Instructions |
| `sec:national-mcp` | §11 National MCP |
| `sec:federated-learning` | §12 Federated Learning |
| `sec:financial` | §13 Financial Analysis |
| `sec:implementation` | §14 Implementation Strategy |
| `sec:discussion` | §15 Discussion |
| `sec:conclusion` | §16 Conclusion |

**Critical note:** All `\label{sec:...}` tags are in `main.tex`, NOT in the section files. The section files only contain `\label{subsec:...}` and `\label{tab:...}` and `\label{fig:...}` tags.

### page_styles.tex — Header/Footer Definitions

Defines 5 `fancyhdr` page styles. The `body` style is used for most of the paper (page number outer corner, section name inner). The `appendix` style replaces the section name with "APPENDICES". The `contents` style is used for ToC and references. The `coverpage` style suppresses all headers/footers.

### references.bib — Bibliography

34 entries in BibTeX format, compiled with `ieeetr` style. Entries fall into these categories:

| Category | Keys | Count |
|----------|------|------:|
| Federal regulations | `cfr-part-312`, `cfr-part-50`, `ich-e6r3`, `hipaa` | 4 |
| FDA guidance/resources | `fda-clinical-trials-guidance`, `fda-oncology-guidance`, `fda-adaptive-designs`, `fda-ai-glossary` | 4 |
| NCI resources | `nci-clinical-trials`, `nci-trials-safety`, `nci-trials-paying`, `nci-cirb`, `nci-modernizing` | 5 |
| Healthcare standards | `hl7-fhir`, `dicom` | 2 |
| AI models | `claude-opus`, `gpt-5-4`, `gemini` | 3 |
| External research | `federated-learning-healthcare`, `tufts-delay` | 2 |
| Author's Zenodo papers | `national-mcp-paper`, `fl-paper`, `site-docs`, `ich-adapt`, `cfr50-adapt`, `cfr312-adapt`, `usl-standard`, `patient-journey-paper`, `patient-instructions-paper` | 9 |
| Author's GitHub repos | `main-repo`, `mcp-repo`, `fl-repo` | 3 |
| Other | `mcp-protocol`, `clinicaltrials-gov`, `groningen-template` | 3 |

---

## Detailed Section Descriptions

### sections/cover_page.tex (50 lines)

Title page with centered layout. Contains: paper title ("National Platform for Physical AI Oncology Trials"), "Draft 1.0", Zenodo DOI hyperlink, author with ORCID link, email, two legal disclaimers (not affiliated with CFR/ICH/FDA; independent work adapted using Claude Code Opus 4.6), location "San Diego", and date "March 28, 2026". Uses `coverpage` page style. Sets `\headheight` to 32pt.

### sections/contents.tex (5 lines)

Minimal file. Sets page style to `acknowledgements`/`contents`, generates `\tableofcontents`, and adds a "Page" header to the ToC column.

### sections/source_documents.tex (146 lines)

A catalog describing all 12 source documents referenced throughout the paper, organized into 5 subsections: Regulatory Adaptation Documents (ICH adaptation, CFR Part 50 adaptation, CFR Part 312 adaptation), Standards and Framework Documents (USL standard, site documentation package), Patient-Centered Documents (patient journey paper, patient instructions paper), National Infrastructure Documents (MCP servers paper, federated learning paper), and Supporting Research (Research A on government operations, Research B on California/federal regulatory analysis). Each entry names the source, cites it, describes its content, and lists which numbered sections reference it. A commented-out "Code Repositories" subsection exists listing the 3 GitHub repos.

**Key role:** This file is the paper's "map" — it tells the reader where each source document is used. Any new section added to the paper should be registered here.

### sections/executive_summary.tex (154 lines)

Seven thematic paragraphs summarizing the entire platform, each with a bold heading: Regulatory Foundation, Quantitative Standards, Simulation Evidence, Patient-Centered Design, National Infrastructure, Economic Impact, Governance Framework. Ends with a figure (`Images/Abdomen.jpeg`, label `Abdomen`) showing an imaging assistant robot. Contains a commented-out concluding paragraph about the 5 categories of platform capability. Cites 16 unique bibliography entries — the most citation-dense section per line.

### sections/introduction.tex (551 lines, §1)

The longest argumentative section. Organized into subsections:

- **§1.1 Background and Motivation** (`subsec:background`): Frames the problem — oncology trials are slow, expensive, and structurally siloed. Introduces the constitutional framework (Articles I–III), key federal statutes (APA, FOIA, Privacy Act, HIPAA), California dual-jurisdiction challenges, and the 24-hour simulation evidence (168 patients, 29 robots, 99.7% uptime).
- **§1.2 Defining Physical AI in Oncology** (`subsec:defining-pai`): Formal definition of Physical AI. References 22 definitions from adapted CFR Part 312, 17 from adapted CFR Part 50, FDA AI glossary gap. Lists 3 AI models (Claude Opus 4.6, GPT-5.4, Gemini) and their roles. Enumerates 10 robot categories with specific models.
- **§1.3 The Case for a National Platform** (`subsec:case-national`): Compares this platform to FDA's piecemeal guidance approach. Lists 5 unique capabilities (simultaneous 3-standard adaptation, 11-doc site package, dual-scale simulation evidence, PSL/USL quantitative standards, MCP/FL national infrastructure).
- **§1.4 Current Oncology Clinical Trial Landscape** (`subsec:current-trials`): Enrollment challenges, cost barriers, equity gaps.
- **§1.5 Scope and Contributions** (`subsec:scope`, `subsec:contributions`): Paper's scope and 10 enumerated contributions.
- **§1.6 AI and Oncology** (`subsec:ai-oncology`): Current state of AI in oncology, what's missing.
- **§1.7 Robotics History** (`subsec:robotics-history`, `subsec:robotics-regulatory`): History of robotics in surgery, regulatory gaps.

**Cites 26 unique bibliography entries** — the widest citation spread of any section.

### sections/gov_framework.tex (712 lines, §2)

The longest file. Provides a detailed legal analysis of how the three branches of U.S. government apply to Physical AI trial governance.

- **§2.1 Constitutional Foundation** (`subsec:constitutional`): Articles I, II, III; Appointments Clause; Take Care duty; Congressional Review Act; APA.
- **§2.2 Legislative Branch** (`subsec:legislative`): Budget and Impoundment Control Act, Congressional Review Act, Congressional Accountability Act, GAO authorities. Table: `tab:legislative-statutes`.
- **§2.3 Executive Branch** (`subsec:executive`): APA, FOIA, Privacy Act, Federal Register Act, Paperwork Reduction Act, Inspectors General, Federal Vacancies Reform Act. Table: `tab:executive-statutes`.
- **§2.4 Judicial Branch** (`subsec:judicial`): Administrative law review, Chevron deference (now modified), judicial review standards.
- **§2.5 Cross-Branch Coordination** (`subsec:cross-branch`): How all three branches interact for PAI regulation.
- **§2.6 Current FDA AI Framework** (`subsec:current-ai-framework`): Existing FDA guidance documents on AI, their limitations for Physical AI.
- **§2.7 Federal Agency Coordination** (`subsec:federal-agency-coordination`): Multi-agency landscape (FDA, HHS, NIST, etc.).

**Contains multiple large tables** with statutes and their Physical AI relevance. Forward-references `sec:implementation` heavily (21 times across the full paper).

### sections/regulatory_landscape.tex (586 lines, §3)

Dual-jurisdiction analysis of California state law and federal law for Physical AI trials.

- **§3.1 Regulatory Classification** (`subsec:regulatory-classification`): Two-axis classification (medical product vs. research tool; investigational vs. approved). 21 U.S.C. §321(h) device definition, 21st Century Cures Act software exclusions.
- **§3.2 California Regulatory Authority** (`subsec:california-authority`): Five sub-areas: human subjects protections (Cal. Health & Safety Code §§24170–24179.5), health data privacy (CMIA §§56–56.37), consumer privacy (CCPA/CPRA), AI-specific healthcare communications (AB 3030, AB 489), medical practice/licensing (Bus. & Prof. Code §2052).
- **§3.3 Federal Regulatory Authority** (`subsec:federal-authority`): IND frameworks (21 CFR Part 312), IDE (Part 812), human subjects (Parts 50, 56, Common Rule), electronic records (Part 11), healthcare data standards (FHIR, DICOM).
- **§3.4 Comparative Obligations** (`subsec:comparative-obligations`): Side-by-side California vs. federal requirements.
- **§3.5 Gaps and Opportunities** (`subsec:gaps-opportunities`): Where existing law falls short for Physical AI.
- **§3.6 FDA Clearance Pathways** (`subsec:fda-clearance-pathways`): 510(k), De Novo, PMA for Physical AI devices.
- **§3.7 Compliance Synthesis** (`subsec:compliance-synthesis`): How the adapted standards fill regulatory gaps.

### sections/ich_e6r3_adaptation.tex (379 lines, §4)

Adaptation of ICH E6(R3) Good Clinical Practice to Physical AI.

- **§4.1 Principles** (`subsec:ich-principles`): 8 foundational GCP principles extended for Physical AI. Table `tab:ich-principles` maps traditional principle → Physical AI extension.
- **§4.2 Investigator Responsibilities** (`subsec:ich-investigator`, `subsec:ich-investigator-detail`): 12 categories adapted. Includes qualifications/training, resources, medical care, protocol compliance, informed consent, records, safety reporting, premature termination. Table `tab:ich-investigator-map`.
- **§4.3 Sponsor Responsibilities** (`subsec:ich-sponsor`): Quality management (`tab:ich-quality`), monitoring, auditing, Physical AI-specific obligations.
- **§4.4 Data Governance** (`subsec:ich-data-governance`): Data management for robot telemetry, AI decision logs, electronic records. References FHIR, DICOM, MCP protocol.
- **§4.5 Documentation** (`subsec:ich-documentation`): Protocol requirements, essential records.
- **§4.6 Glossary** (`subsec:ich-glossary`): 30 Physical AI terms. Table `tab:ich-definitions`.

**Key dependency:** Cross-references USL scores from §7 (`sec:psl-usl`) for protocol requirements (Principle 5).

### sections/cfr50_adaptation.tex (670 lines, §5)

Adaptation of 21 CFR Part 50 (Protection of Human Subjects) for Physical AI. Second-longest section file.

- **§5.1 Scope and Definitions** (`subsec:cfr50-scope`): 17 Physical AI-specific definitions. Cross-references 22 definitions in adapted Part 312 and 30 in adapted ICH E6(R3) for terminological consistency.
- **§5.2 Informed Consent** (`subsec:cfr50-consent`, `subsec:cfr50-consent-comprehensive`, `subsec:cfr50-consent-templates`): Additional consent elements for robotic procedures. Ongoing consent beyond traditional one-time event (`subsec:cfr50-ongoing`).
- **§5.3 Physical AI Safety Requirements** (`subsec:cfr50-safety`): **New Subpart C** — no precedent in original Part 50. Pre-procedure safety matrix, runtime monitoring, post-procedure verification.
- **§5.4 Three-Tier Classification** (`subsec:cfr50-three-tier`): Physical AI systems classified by autonomy level. Table `tab:three-tier-detail`.
- **§5.5 IRB Review** (`subsec:cfr50-irb`): Physical AI-specific IRB requirements.
- **§5.6 Pediatric Protections** (`subsec:cfr50-pediatric`): Additional protections for minors in Physical AI trials.
- **§5.7 Classification Framework** (`subsec:cfr50-classification`): 10-category robot classification.
- **§5.8 Regulatory Pathway** (`subsec:cfr50-regulatory-pathway`): Regulatory pathway integration.
- **§5.9 Data Protection** (`subsec:cfr50-data`): Data protections specific to Physical AI.
- **§5.10 Interrelationship** (`subsec:cfr50-interrelationship`): How adapted Part 50 relates to adapted Part 312 and ICH E6(R3).

**Key theme:** "Control has shifted towards the patient's side" — expanded patient rights for robotic interactions.

### sections/cfr312_adaptation.tex (501 lines, §6)

Adaptation of 21 CFR Part 312 (IND) for Physical AI. The most extensive regulatory adaptation.

- **§6.1 Scope and Definitions** (`subsec:cfr312-scope`): 22 Physical AI definitions — the most comprehensive set across all 3 standards.
- **§6.2 IND Requirements** (`subsec:cfr312-ind`): When IND is required for Physical AI trials. **Phase 0 simulation validation** — mandatory pre-enrollment phase (major innovation). Table `tab:trial-phases`. Physical AI System Description requirement for IND submissions.
- **§6.3 Safety Reporting** (`subsec:cfr312-safety`): Physical AI adverse event reporting, cybersecurity incident reporting. Tables `tab:ae-categories`, `tab:ae-categories-expanded`.
- **§6.4 Responsibilities** (`subsec:cfr312-responsibilities`): Sponsor and investigator responsibilities.
- **§6.5 Administrative Actions** (`subsec:cfr312-admin`): Clinical holds, IND withdrawal.
- **§6.6 Expanded Access** (`subsec:cfr312-expanded-access`, `subsec:cfr312-life-threatening`): Emergency use provisions for Physical AI.
- **§6.7 Subpart J** (`subsec:cfr312-subpart-j`, `subsec:cfr312-subpart-j-expanded`): **New subpart dedicated entirely to Physical AI systems** — no precedent in original Part 312. Table `tab:subpart-j`. Covers cybersecurity, decommissioning, inter-system communication.
- **§6.8 Cross-Reference** (`subsec:cfr312-cross-ref`): How Part 312 connects to Part 50 and ICH E6(R3).

**The three regulatory adaptations (§4, §5, §6) form a tightly coupled triad.** Definitions are cross-referenced across all three. When editing any one, check the other two for consistency.

### sections/psl_usl_standards.tex (572 lines, §7)

Defines two complementary quantitative scoring frameworks.

- **§7.1 Overview** (`subsec:standards-overview`): PSL evaluates sites, USL evaluates robots. Both must pass for trial activation.
- **§7.2 PSL Framework** (`subsec:psl-framework`): 3 dimensions — Legislative Authorization (SB 1042, AB 2847, SB 892), Regulatory Compliance (Title 22, FDA, building codes), Operational Readiness (SOPs, training, equipment). Table `tab:psl-dimensions`. Pass/fail scoring per dimension. Table `tab:psl-scoring`.
- **§7.3 USL Framework** (`subsec:usl-framework`): 4 dimensions — Simulation Framework Switching (A), AI Integration (B), Cross-Robot Sharing (C), Clinical Trial Collaboration (D). Table `tab:usl-dimensions`. Scores 1.0–10.0.
- **§7.4 USL Levels** (`subsec:usl-levels`, `subsec:usl-level-progression`): 5 bands — Initial (1.0–2.9), Foundational (3.0–4.9), Intermediate (5.0–6.9), Advanced (7.0–8.9), Exemplary (9.0–10.0). Tables `tab:usl-levels`, `tab:usl-level-implications`.
- **§7.5 USL Results** (`subsec:usl-results`): Scores for 9 robots across 3 categories. **Table `tab:usl-all-scores`** — the most-referenced table in the paper. Key results: Franka Panda USL 7.4 (highest overall), da Vinci dVRK USL 7.1 (highest surgical), Tesla Optimus USL 5.1 (highest humanoid).
- **§7.6 Dimension Analysis** (`subsec:dimension-analysis`, `subsec:dimension-patterns`): Cross-cutting patterns. Dim D (clinical trial collaboration) is weakest field-wide. Open-source ecosystem maturity is strongest predictor.
- **§7.7 Scoring Engine** (`subsec:scoring-engine`): Computational implementation details.
- **§7.8 PSL-USL Complement** (`subsec:psl-usl-complement`): How the two standards work together.
- **§7.9 Standards Impact** (`subsec:standards-impact`): Implications for implementation.

**Key dependency:** USL scores are referenced in §4 (protocol requirements), §5 (classification), §9 (patient journey robot selection), §14 (implementation robot procurement), §15 (discussion patterns).

### sections/site_establishment.tex (514 lines, §8)

Complete documentation package for establishing the first Physical AI trial site (California).

- **§8.1 Overview** (`subsec:site-overview`): Table `tab:site-docs-overview` — 11 documents across 4 categories (legislation, regulations, federal compliance, infrastructure, operations).
- **§8.2 State Legislation** (`subsec:site-legislation`): SB 1042 (authorization), AB 2847 (patient rights — the bill that legally enforces patient right to refuse robotic procedures), SB 892 (data protection).
- **§8.3 Regulations** (`subsec:site-regulations`): SF Municipal Code updates, Title 22 state regulations.
- **§8.4 Federal Compliance** (`subsec:site-federal`, `subsec:site-federal-expanded`): FDA compliance guide bridging state and federal.
- **§8.5 Infrastructure** (`subsec:site-infrastructure`): Building code, premises code, parking/transportation.
- **§8.6 Operations** (`subsec:site-operations`): Activation SOPs, emergency preparedness.
- **§8.7 Documentation Comprehensive** (`subsec:site-docs-comprehensive`): Extended detail table `tab:site-docs-comprehensive`.
- **§8.8 24-Hour Simulation** (`subsec:site-simulation`): The pivotal 168-patient, 29-robot, 99.7% uptime simulation. Tables `tab:simulation-metrics`, `tab:simulation-robots`, `tab:multi-patient`.
- Figure: `Images/Motion.jpeg` (label `Motion`) — motion-tracking robot during radiotherapy.

**Key dependency:** PSL dimensions in §7 directly correspond to the documentation categories here. SB 1042 → PSL Dim 1, Title 22 → PSL Dim 2, Activation SOPs → PSL Dim 3.

### sections/patient_journey.tex (481 lines, §9)

Documents a complete single-patient simulation: Stage IIIB NSCLC patient over 1,120 days through 10 stages.

- **§9.1 Pipeline Architecture** (`subsec:journey-pipeline`): Table `tab:journey-stages` — 10 stages from Prescreening through Closeout.
- **§9.2 Pre-Trial Stages** (`subsec:journey-pretrial`): Stages 1–4 (prescreening, screening, informed consent, enrollment). Maps each stage to all 3 adapted standards.
- **§9.3 Treatment Stages** (`subsec:journey-treatment`): Stages 5–8 (treatment planning with digital twins, surgical intervention with da Vinci, post-surgical robotic monitoring, follow-up therapy).
- **§9.4 Closeout** (`subsec:journey-closeout`): Stages 9–10 (long-term monitoring, regulatory closeout documentation).
- **§9.5 Safety Outcomes** (`subsec:journey-safety`): Zero patient harm events, all adverse events documented per ICH E6(R3).
- **§9.6 Regulatory Compliance** (`subsec:journey-compliance`): Table `tab:journey-compliance` mapping stages to standards.
- **§9.7 Timeline** (`subsec:journey-timeline`, `subsec:journey-detailed-timeline`): Tables `tab:journey-timeline`, with day counts per stage.
- **§9.8 Cost Analysis** (`subsec:journey-cost`, `subsec:journey-cost-detail`, `subsec:journey-cost-expanded`): Per-stage cost savings. Table `tab:journey-cost-stage`.
- **§9.9 Discussion** (`subsec:journey-discussion`): Comparison to traditional trials (`subsec:journey-traditional-comparison`), ecosystem implications (`subsec:journey-ecosystem`), limitations (`subsec:journey-limitations`).

**Key dependency:** Cost data flows directly into §13 (Financial Analysis). Safety outcomes feed into §15 (Discussion evidence). The 10-stage pipeline is the experiential counterpart to the regulatory framework in §4–§6.

### sections/patient_instructions.tex (401 lines, §10)

Patient-facing documentation covering all 10 robot types.

- **§10.1 Purpose** (`subsec:instructions-purpose`): Plain language, consistent structure, emphasis on patient rights.
- **§10.2–§10.11** — One subsection per robot type:
  - `subsec:instructions-surgical` — Surgical robots (da Vinci, Hugo RAS, Versius)
  - `subsec:instructions-cobots` — Collaborative robots (Franka Panda, Kinova Gen3, xArm 7)
  - `subsec:instructions-rt-positioning` — Radiotherapy positioning robots
  - `subsec:instructions-needle` — Needle-placement robots
  - `subsec:instructions-companion` — Social companion robots
  - `subsec:instructions-humanoid` — Humanoid robots (Tesla Optimus, Digit, Atlas)
  - `subsec:instructions-rt-tracking` — Radiotherapy motion-tracking robots
  - `subsec:instructions-imaging` — Imaging robots
  - `subsec:instructions-steerable` — Steerable needle robots
  - `subsec:instructions-rehab` — Rehabilitation exoskeletons
- **§10.12 Benefits** (`subsec:instructions-benefits`, `subsec:instructions-category-benefits`): Benefits by robot category.
- **§10.13 Rights Summary** (`subsec:instructions-rights-summary`): Consolidated patient rights.
- **§10.14 Summary Table** (`subsec:instructions-summary-table`): Table `tab:robot-summary`.
- **§10.15 Accessibility** (`subsec:instructions-accessibility`): Accessibility design considerations.

**Key dependency:** Robot type numbering (1–10) is consistent with ICH adaptation (§4), USL evaluation (§7), patient journey (§9), and MCP tool inventory (§11). All 10 types use the same classification across the entire paper.

### sections/national_mcp.tex (607 lines, §11)

Five-server Model Context Protocol infrastructure for national-scale Physical AI trial coordination.

- **§11.1 Fragmentation Problem** (`subsec:mcp-fragmentation`): Current siloed systems (EDC, LIMS, CTMS) can't support Physical AI real-time needs.
- **§11.2 MCP for Physical AI** (`subsec:mcp-protocol`): Adaptation of Anthropic's MCP for robot control, sensor streams, safety interlocks, audit trails.
- **§11.3 Five-Server Architecture** (`subsec:mcp-architecture`): Table `tab:mcp-servers` — Governance, Data Flow, Safety, Interoperability, Analytics servers.
- **§11.4 Tool Inventory** (`subsec:mcp-tools`, `subsec:mcp-tool-inventory`, `subsec:mcp-tool-specs`): 23 tools across 5 servers. Tables `tab:mcp-all-tools`, `tab:mcp-tool-specs`.
- **§11.5 Safety Modules** (`subsec:mcp-safety`, `subsec:mcp-patient-safety`): Emergency stop propagation, anomaly detection, incident reporting.
- **§11.6 Integration** (`subsec:mcp-integration`): FHIR, DICOM, HL7 integration adapters.
- **§11.7 Governance** (`subsec:mcp-governance`): Role-based access, audit trails, 21 CFR Part 11 compliance.
- **§11.8 Deployment** (`subsec:mcp-deployment`, `subsec:mcp-deployment-phases`, `subsec:mcp-network-arch`): Hub-and-spoke national topology. Tables `tab:mcp-deployment-phases`, `tab:mcp-network-config`.
- **§11.9 CI/CD** (`subsec:mcp-cicd`): Continuous integration for MCP server updates.

**Key dependency:** The Safety Server connects to the adapted Part 50 safety matrix (§5). The Governance Server enforces the 3 adapted standards (§4–§6). The Data Flow Server feeds the federated learning framework (§12). The Analytics Server generates data for financial analysis (§13).

### sections/federated_learning.tex (429 lines, §12)

Privacy-preserving multi-site machine learning framework. 235 modules, 86,800 lines of code, 82 test files.

- **§12.1 Privacy Challenge** (`subsec:fl-privacy`): Why centralized data collection is unacceptable under HIPAA + California privacy laws.
- **§12.2 Methods** (`subsec:fl-methods`): Prompt-driven development using Claude Code Opus 4.6.
- **§12.3 Architecture** (`subsec:fl-architecture`): Three layers — client nodes, aggregation servers, coordination layer. Integrates with MCP servers at each site.
- **§12.4 Five Pillars** (`subsec:fl-pillars`, `subsec:fl-pillar-details`): Tables `tab:fl-pillars`, `tab:fl-pillar-details`.
- **§12.5 Triple AI Peer Review** (`subsec:fl-peer-review`, `subsec:fl-peer-review-extended`): Model updates reviewed by Claude Opus 4.6, GPT-5.4, and Gemini before deployment.
- **§12.6 Code Trust** (`subsec:fl-trust`): 82 test files, safety validation.
- **§12.7 Analytics** (`subsec:fl-analytics`): Clinical analytics, digital twin integration.
- **§12.8 MCP Integration** (`subsec:fl-mcp-integration`): How FL client nodes connect through MCP servers.
- **§12.9 Technical Stack** (`subsec:fl-tech-stack`): Implementation details.
- **§12.10 Workflows** (`subsec:fl-workflows`): Federated learning round lifecycle.
- **§12.11 Limitations** (`subsec:fl-limitations`): Communication overhead, heterogeneous data challenges.

**Key dependency:** Tightly coupled with §11 (MCP servers provide the communication layer). Privacy requirements come from §3 (regulatory landscape, HIPAA/CMIA/CCPA) and §8 (SB 892 data protection). Triple AI peer review uses the same 3 AI models defined in §1.

### sections/financial_analysis.tex (536 lines, §13)

Economic analysis drawing from patient journey cost data and Tufts CSDD delay valuations.

- **§13.1 Cost of Delays** (`subsec:financial-delays`): Tufts CSDD data on daily cost of drug development delay.
- **§13.2 Per-Patient Analysis** (`subsec:financial-per-patient`): Table `tab:cost-comparison` — traditional vs. Physical AI across 7 cost categories.
- **§13.3 Speed Advantages** (`subsec:financial-speed`): Timeline compression at enrollment, treatment, documentation stages.
- **§13.4 Single-Site Economics** (`subsec:financial-comparison`): ROI analysis. Table `tab:roi-analysis`.
- **§13.5 Scale Economics** (`subsec:financial-scale`): Cost per patient decreases with site throughput.
- **§13.6 National Projections** (`subsec:financial-national`, `subsec:financial-detailed-projections`): Tables `tab:financial-projections`, `tab:detailed-financial-projections`.
- **§13.7 Patient Benefits** (`subsec:financial-patient`, `subsec:financial-patient-quantified`): Table `tab:patient-financial-benefits` — lower wait times, reduced costs, higher quality.
- **§13.8 Pharma Perspective** (`subsec:financial-pharma`): ROI for pharmaceutical companies. Table `tab:detailed-roi-analysis`.
- **§13.9 Infrastructure Costs** (`subsec:financial-infrastructure`): MCP server and FL deployment costs.
- **§13.10 Detailed ROI** (`subsec:financial-detailed-roi`): Extended analysis.

**Key dependency:** Per-patient cost data comes from §9 (patient journey). Scale data comes from §8 (24-hour simulation). Infrastructure costs reference §11 (MCP) and §12 (FL). Forward-references §14 (implementation phasing).

### sections/implementation_strategy.tex (457 lines, §14)

Three-phase rollout plan for nationwide Physical AI trial deployment.

- **§14.1 Implementation Phases** (`subsec:impl-phases`): Phase 1 (California first site), Phase 2 (5–10 multi-state sites), Phase 3 (nationwide). Each phase details activities, evidence base, and success criteria.
- **§14.2 Regulatory Pathway** (`subsec:impl-regulatory`): How the 3 adapted standards sequence through implementation.
- **§14.3 Standards Integration** (`subsec:impl-standards`): PSL and USL gate progression through phases.
- **§14.4 Infrastructure** (`subsec:impl-infrastructure`): MCP deployment at each phase. Regional hubs in Phase 2, national mesh in Phase 3.
- **§14.5 Workforce** (`subsec:impl-workforce`, `subsec:impl-workforce-detail`, `subsec:impl-expanded-workforce`): New roles (Physical AI Operators, robot technicians, AI safety specialists). Tables `tab:workforce-roles`, `tab:workforce-detailed-roles`.
- **§14.6 Timeline** (`subsec:impl-timeline`, `subsec:impl-detailed-timeline`): Tables `tab:impl-timeline`, `tab:impl-detailed-timeline`.
- **§14.7 Metrics** (`subsec:impl-metrics`): Success metrics per phase.
- **§14.8 Risk Management** (`subsec:impl-risk`): Risk registry and mitigation.
- **§14.9 Quality** (`subsec:impl-quality`): Quality management system.
- **§14.10 Patient Access** (`subsec:impl-patient`): Equity and access strategy.

**§14 is the most forward-referenced section** (21 cross-references from other files). It is the convergence point where all preceding sections feed into actionable plans.

### sections/discussion.tex (501 lines, §15)

Synthesis across all sections. The "so what" of the paper.

- **§15.1 Paradigm Shift** (`subsec:discuss-paradigm`): Integrated platform vs. FDA's piecemeal approach.
- **§15.2 Credibility** (`subsec:discuss-credibility`): Why building on existing standards (ICH E6(R3), CFR Parts 50/312) adds legitimacy.
- **§15.3 PSL/USL Impact** (`subsec:discuss-standards`, `subsec:discuss-psl-usl-impact`): Patterns from USL scoring (open-source maturity predicts readiness; Dim D is weakest field-wide; hardware ≠ readiness).
- **§15.4 Evidence** (`subsec:discuss-evidence`): Both simulations as proof of concept.
- **§15.5 Patient Empowerment** (`subsec:discuss-patient`): "Control has shifted towards the patient's side."
- **§15.6 Infrastructure** (`subsec:discuss-infrastructure`): MCP + FL as enabling technologies.
- **§15.7 Governance** (`subsec:discuss-governance`): Three-branch framework legitimacy.
- **§15.8 FDA Comparison** (`subsec:discuss-expanded-comparison`): Table `tab:platform-vs-fda`, extended table `tab:expanded-platform-vs-fda`.
- **§15.9 Stakeholder Concerns** (`subsec:discuss-concerns`, `subsec:discuss-stakeholder-concerns`, `subsec:discuss-structured-concerns`): Tables `tab:stakeholder-concerns`, `tab:structured-concerns`.
- **§15.10 International** (`subsec:discuss-international`): Global applicability.
- **§15.11 Future Directions** (`subsec:discuss-future`).
- **§15.12 Limitations** (`subsec:discuss-limitations`): Both simulations are AI-generated; no real patient data yet.

**References 10 of 16 section labels** — the widest cross-referencing scope of any section file.

### sections/conclusion.tex (454 lines, §16)

- **§16.1 Summary** (`subsec:conclusion-summary`): Recaps the full platform.
- **§16.2 Key Findings** (`subsec:conclusion-findings`): 7 enumerated findings (regulatory compatibility, simulation evidence, quantitative standards, patient empowerment, national infrastructure, economic viability, governance legitimacy).
- **§16.3 Pharma Implications** (`subsec:conclusion-pharma`): Industry action items.
- **§16.4 Regulatory Implications** (`subsec:conclusion-regulatory`): FDA and state regulator action items.
- **§16.5 Recommendations** (`subsec:conclusion-recommendations`): Table `tab:recommended-actions`.
- **§16.6 Future** (`subsec:conclusion-future`, `subsec:conclusion-expanded-future`).
- **§16.7 Call to Action** (`subsec:conclusion-action`, `subsec:conclusion-expanded-action`).
- Figure: `Images/Walking.jpeg` (label `Walking`) — humanoid robot in clinical setting.

### sections/appendices.tex (314 lines)

Five appendices (labeled A–E via `\Alph{subsection}`):

- **Appendix A** (`app:source-files`): Complete source file directory listing all files from `national-platform/` with line counts and descriptions. 11 subdirectories, ~50 source files total.
- **Appendix B** (`app:glossary`): Unified glossary of 23 Physical AI terms in a `longtable`. Drawn from glossaries in all 3 adapted standards.
- **Appendix C** (`app:cross-reference`): Regulatory cross-reference matrix mapping each paper section to source docs, source files, applicable standards, and key references.
- **Appendix D** (`app:simulation`): Simulation summary data.
- **Appendix E** (`app:scoring`): USL scoring methodology details.

---

## Cross-Reference Dependency Map

This map shows which section files reference which other sections (via `\ref{sec:...}`), revealing the dependency structure:

```
source_documents.tex ──→ ALL sections (it's the master index)

introduction.tex ──→ sec:financial, sec:implementation
gov_framework.tex ──→ sec:implementation
regulatory_landscape.tex ──→ sec:implementation
ich_e6r3_adaptation.tex ──→ sec:psl-usl
cfr50_adaptation.tex ──→ sec:psl-usl
cfr312_adaptation.tex ──→ sec:cfr50, sec:psl-usl
psl_usl_standards.tex ──→ sec:site-establishment
site_establishment.tex ──→ sec:financial
patient_journey.tex ──→ sec:financial
federated_learning.tex ──→ sec:implementation
financial_analysis.tex ──→ sec:national-mcp
national_mcp.tex ──→ sec:federated-learning, sec:implementation
implementation_strategy.tex ──→ sec:national-mcp
discussion.tex ──→ sec:gov-framework, sec:regulatory-landscape, sec:cfr50,
                    sec:psl-usl, sec:site-establishment, sec:patient-instructions,
                    sec:national-mcp, sec:federated-learning, sec:financial,
                    sec:implementation
conclusion.tex ──→ sec:patient-journey, sec:patient-instructions,
                    sec:financial, sec:implementation
```

**Most forward-referenced targets:**
1. `sec:implementation` — 21 references (convergence point for all plans)
2. `sec:financial` — 13 references (economic case is pervasive)
3. `sec:psl-usl` — 10 references (standards are used everywhere)
4. `sec:national-mcp` — 8 references (infrastructure is foundational)

---

## Shared Concepts and Consistency Requirements

### The Three Adapted Standards (§4, §5, §6)

These three sections form a **regulatory triad** that must remain internally consistent:

| Standard | Definitions | Key Innovation |
|----------|----------:|----------------|
| ICH E6(R3) adaptation (§4) | 30 terms | 8 GCP principles extended for Physical AI |
| 21 CFR Part 50 adaptation (§5) | 17 terms | New Subpart C (Physical AI safety requirements) |
| 21 CFR Part 312 adaptation (§6) | 22 terms | Phase 0 simulation validation + new Subpart J |

**Total: 69 definitions across 3 standards.** While each standard has its own glossary scoped to its domain, key terms (Physical AI System, Physical AI Adverse Event, Physical AI Operator, Autonomous Mode, Supervised Mode, etc.) must have compatible definitions across all three.

### The Dual Quantitative Standards (§7)

PSL (3 dimensions, site-level) and USL (4 dimensions, robot-level) are referenced in 10+ other sections. Key data points that must remain consistent:

- Franka Panda: USL 7.4, Advanced band
- da Vinci dVRK: USL 7.1, Advanced band, highest clinical trial readiness (Dim D: 7.0)
- Tesla Optimus: USL 5.1, Intermediate band
- PSL pass/fail gates correspond to site documentation categories in §8

### The Two Simulations

| Simulation | Source | Scale | Key Metrics |
|-----------|--------|-------|-------------|
| Single-patient journey | `patient-journey-paper` / §9 | 1 patient, 1,120 days, 10 stages | Stage IIIB NSCLC, prescreening → closeout |
| 24-hour multi-patient | `site-docs` / §8 | 168 patients, 29 robots, 15 cancer types | 99.7% uptime, 0 patient harm events, 7 adverse events |

These metrics are quoted across §1, §5, §6, §8, §9, §13, §14, §15, §16. Any change to these numbers must be propagated everywhere.

### The 10 Robot Categories

The same 10-type classification is used consistently in §4 (GCP requirements per type), §5 (classification framework), §7 (USL evaluation), §9 (patient journey robot selection), §10 (patient instructions per type), §11 (MCP tool inventory), and §14 (procurement):

1. Surgical robots (da Vinci, Hugo RAS, Versius)
2. Collaborative robots / cobots (Franka Panda, Kinova Gen3, xArm 7)
3. Radiotherapy positioning robots
4. Needle-placement robots
5. Social companion robots
6. Humanoid robots (Tesla Optimus, Digit, Boston Dynamics Atlas)
7. Radiotherapy motion-tracking robots
8. Imaging robots
9. Steerable needle robots
10. Rehabilitation exoskeletons

### The Three AI Models

Claude Opus 4.6, GPT-5.4, and Gemini are referenced together in:
- §1 (defining Physical AI, enabling technologies)
- §2 (executive branch AI context)
- §7 (development tool)
- §12 (triple AI peer review pipeline)
- Executive summary (infrastructure)

### The 5 MCP Servers

Governance, Data Flow, Safety, Interoperability, Analytics — referenced as a unit in §8 (site infrastructure), §11 (full architecture), §12 (FL integration), §14 (deployment phases), §15 (discussion).

---

## Images

Three images are referenced (stored in `Images/` directory, not included in this repo):

| File | Label | Location | Description |
|------|-------|----------|-------------|
| `Images/Abdomen.jpeg` | `Abdomen` | executive_summary.tex | Imaging assistant robot scanning liver tumor |
| `Images/Motion.jpeg` | `Motion` | site_establishment.tex | Motion-tracking robot during radiotherapy |
| `Images/Walking.jpeg` | `Walking` | conclusion.tex | Humanoid robot in clinical setting |

[Full image folder on Google Drive](https://drive.google.com/drive/folders/1E6lWyrVqCiGe97oKIUI2sIxG3xbh51tT)

---

## LaTeX Conventions Used Throughout

- **Section numbering:** `\section{}` in main.tex (§1–§16), `\subsection{}` and `\subsubsection{}` within section files.
- **Labels:** `sec:` prefix for sections (in main.tex only), `subsec:` for subsections, `tab:` for tables, figure labels are plain words (`Abdomen`, `Motion`, `Walking`).
- **Tables:** Mostly `tabularx` with `X` columns inside `table[H]` floats. Some `longtable` in appendices. All use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`).
- **Citations:** IEEE numeric style via `\bibliographystyle{ieeetr}`. Grouped citations use `\cite{key1, key2, key3}`.
- **Spacing:** `\vspace{0.3cm}` between major paragraphs. `\noindent\textbf{...}` for inline headings within unnumbered sections.
- **Lists:** `enumitem` package with `[nosep]` for compact lists.
- **Cross-references:** `\ref{}` for sections/tables, `\autoref{}` for figures (produces "Figure N").
- **URLs:** Handled via `\url{}` with extensive `\UrlBreaks` configuration. `\sloppy` and `\emergencystretch=3em` prevent overfull hbox warnings.
- **Page breaks:** `\newpage` between all sections (enforced in main.tex) and `\clearpage` before some tables/figures within sections.
- **Color:** All text forced to black via `\color{black}` and `\hypersetup` with all link colors black.

---

## Source Document to Section File Mapping

| Source Document (Zenodo) | Cite Key | Primary Section | Also Referenced In |
|--------------------------|----------|----------------|--------------------|
| ICH E6(R3) Adaptation | `ich-adapt` | §4 `ich_e6r3_adaptation.tex` | §5, §6, §7, §8, §9, §10, §11, §12, §13, §14, §15, §16 |
| 21 CFR Part 50 Adaptation | `cfr50-adapt` | §5 `cfr50_adaptation.tex` | §4, §6, §8, §9, §10, §12, §14, §15, §16 |
| 21 CFR Part 312 Adaptation | `cfr312-adapt` | §6 `cfr312_adaptation.tex` | §4, §5, §8, §9, §11, §12, §14, §15, §16 |
| USL Standard | `usl-standard` | §7 `psl_usl_standards.tex` | §1, §4, §5, §9, §10, §14, §15, §16 |
| Site Documentation | `site-docs` | §8 `site_establishment.tex` | §1, §3, §5, §6, §7, §9, §10, §12, §13, §14, §15, §16 |
| Patient Journey | `patient-journey-paper` | §9 `patient_journey.tex` | §1, §6, §8, §13, §14, §15, §16 |
| Patient Instructions | `patient-instructions-paper` | §10 `patient_instructions.tex` | §1, §3, §4, §5, §9, §13, §14, §15, §16 |
| National MCP Servers | `national-mcp-paper` | §11 `national_mcp.tex` | §1, §3, §7, §8, §12, §13, §14, §15, §16 |
| Federated Learning | `fl-paper` | §12 `federated_learning.tex` | §1, §2, §3, §5, §11, §13, §14, §15, §16 |
| Research A (Gov Ops) | — | §2 `gov_framework.tex` | §1 (indirectly) |
| Research B (CA/Fed Law) | — | §3 `regulatory_landscape.tex` | §1 (indirectly) |

---

## Key Recurring Phrases

These exact phrases appear across multiple sections and should be preserved for consistency:

- "**Physical AI**" — always capitalized, always two words
- "**state-of-the-art AI has the ability to understand, work with, and scale Physical AI applications beyond human and prior technology capabilities**" — used in §8, §9
- "**more patients can be treated with more robots and fewer workers**" — used in §1, §8, §13
- "**control has shifted towards the patient's side**" — used in §5, §10, §15, executive summary
- "**adds credibility**" / "**adds immediate credibility and practical applicability**" — used in §1, §4, §5, §6, §15, §16, executive summary
- "**99.7 percent uptime**" — always written out, never "99.7%"
- "**zero patient harm events**" — always lowercase "zero"
- "**168 patients**", "**29 robots**", "**15 cancer types**" — the 24-hour simulation triad

---

## Notes for Processing

1. **The paper is self-contained.** All regulatory content is adapted (not copied) from public domain sources. The ICH document has a public license. CFR documents are public domain.

2. **Definitions cascade.** ICH adaptation (30 defs) → CFR Part 50 (17 defs) → CFR Part 312 (22 defs). The glossary in Appendix B unifies these. When modifying definitions, check all three standards and the appendix glossary.

3. **The implementation section (§14) is the convergence point.** It pulls from every preceding section. When modifying any earlier section, check whether §14's references still hold.

4. **The discussion section (§15) cross-references 10 of 16 sections.** It synthesizes arguments from across the paper. Changes to evidence or claims in earlier sections should be checked against §15.

5. **Tables are dense and numerous.** The paper contains approximately 50+ tables. Major tables like `tab:usl-all-scores`, `tab:mcp-all-tools`, `tab:site-docs-comprehensive`, and `tab:platform-vs-fda` are referenced from multiple sections.

6. **All section `\label{sec:...}` tags are defined in main.tex, not in section files.** Section files only define `\label{subsec:...}`, `\label{tab:...}`, and figure labels.

7. **The three images are external.** They're referenced by path `Images/*.jpeg` but not included in the repo. The Google Drive link is provided above.
