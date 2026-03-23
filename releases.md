# Releases

Release notes for the physical-ai-oncology-trials repository.

---

24-Hour On-Demand Physical AI Oncology Clinical Trial Simulation
v2.8.0 - On-Demand Trial Simulation

## Summary

Full 24-hour simulation of an autonomous, patient-centric Physical AI oncology
clinical trial at a single site with 1-minute resolution. Introduces the
Physical AI Standard Level (PSL) framework, a new scoring system evaluating
each of 10 robot types on three regulatory dimensions: Omniscient (ICH E6(R3)),
Omnipresent (21 CFR Part 50), and Omnipotent (21 CFR Part 312). The simulation
demonstrates 168 patients across 15 cancer types served by 29 robot instances
in a 24/7 on-demand format, achieving a cumulative site PSL of 63.4 to 64.4
(Advanced Site band).

## Features

- 178 output files across 25 sequential commits
- PSL framework with three regulatory dimensions (0.0 to 10.0 per robot)
- 168 unique patients with minute-level vital sign simulation
- 15 cancer types treated simultaneously
- 10 robot types (29 instances) with detailed telemetry
- 72 ASCII text diagrams (3 per hour from 3 perspectives)
- 7 adverse events (all Grade 1-2, managed successfully)
- Site specification with building, staffing, and infrastructure requirements
- Format comparison document (on-demand vs. traditional trials)
- Error review and cumulative 24-hour summaries

## Contributors
@kevinkawchak
@claude
@openai
@google-gemini

## Notes

- PSL scores complement USL scores (DOI: 10.5281/zenodo.18778220)
- Extends single-patient journey work (DOI: 10.5281/zenodo.19119939)
- Governed by 3 adapted regulatory frameworks
- Development by Claude Code Opus 4.6

---

Repository-Wide Documentation Structure Update
v2.7.1 - Documentation Refresh

Released on 21 March 2026
CEO Kevin Kawchak, ChemicalQDevice

## Summary

Repository-wide documentation refresh updating 38 README files, version badges, project structure, framework version numbers, and metadata across all 25 top-level directories. Aligns all module documentation with current v2.7.1 release state, adds missing `regulatory-submit/` directory to the project structure, removes deleted `unification/industry/` reference, updates CITATION.cff to v2.7.1, and ensures consistent "Last Updated: March 2026" dates across all modules. Updates citation version from 2.4.0 to 2.7.1. No Python code changes.

## Features

- **38 README files updated**: Version badges updated to v2.7.1 and dates updated to March 2026 across all module and sub-module READMEs (agentic-ai, digital-twins, examples, examples-new, federation, images, patients, patient-journey, privacy, q1-2026-standards, regulatory, regulatory-submit, tests, tools, unification)
- **Main README structure corrected**: Added `regulatory-submit/` directory (6 Python modules + 6 examples for FDA submission automation), removed deleted `unification/industry/` directory reference, added regulatory-submit to engineering examples table
- **Version metadata standardized**: All module READMEs now include consistent `**Version**: 2.7.1` and `**Last Updated**: March 2026` metadata blocks
- **Citation updated**: CITATION.cff version updated from 2.4.0 to 2.7.1, date-released updated to 2026-03-21, BibTeX citation block in main README updated to v2.7.1
- **Framework version references updated**: Core Technologies date range updated to "October 2025 - March 2026", requirements.txt header date updated to March 2026
- **Engineering examples table expanded**: Added `regulatory-submit/` row documenting 6 examples for Pre-Sub packages, PCCP plans, pathway classification, and IEC 62304 documentation
- **Release notes**: v2.7.1 entry added to releases.md and CHANGELOG.md

## Contributors
@kevinkawchak
@claude

## Notes

- No Python code changes -- documentation-only release
- All 242 Python files pass ruff lint and format checks
- All YAML files pass yamllint validation
- CI checks validated on Python 3.10, 3.11, and 3.12
- Development by Claude Code Opus 4.6
- License: MIT (repository code)
- @kevinkawchak PDF and LaTeX source code cleanup for recent works. Added corresponding DOI links in README files for access on Zenodo.
  
---

A Cancer Patient's Journey Through a Regulated and Autonomous Physical AI Oncology Trial Illustration
v2.7.0 - Patient Journey Paper

Released on 20 March 2026
CEO Kevin Kawchak, ChemicalQDevice

## Summary

Publishes **A Cancer Patient's Journey Through a Regulated and Autonomous Physical AI Oncology Trial Illustration**, a comprehensive LaTeX paper documenting the first fully autonomous single-patient journey through a regulated Physical AI oncology clinical trial illustration. The paper covers the complete 10-stage journey of PAT-2026-0042 (58F, Stage IIIB NSCLC) orchestrated by Claude Code Opus 4.6 in 13 commits over 72 minutes. Includes treatment outcomes (CR, R0 resection, HR 0.62, 36-month EFS), regulatory coverage (84+ sections across 21 CFR Part 312, 21 CFR Part 50, ICH E6(R3)), FDA cost-savings projections ($390M-$650M), Physical AI ecosystem architecture (da Vinci Xi USL 87.5, Franka Emika USL 88.75), and 4 guidance documents.

## Features

- **Complete LaTeX paper** (`patient-journey/paper/patient_journey_paper.tex`): Abstract, Introduction with regulatory disclaimer, Table of Contents, Methods, Results, Discussion, Limitations and Future Work, Conclusions, References (18 citations), Acknowledgments, Ethical Disclosures, Rights and Permissions (CC BY 4.0), and Citation
- **Treatment outcomes**: Complete Response (CR), R0 resection via da Vinci Xi (168-min lobectomy), 35 pembrolizumab cycles, 36-month event-free survival, recurrence risk 35% to 3%, HR 0.62 (95% CI: 0.45-0.85)
- **Regulatory coverage**: 84+ sections across three adapted frameworks with regulatory-to-stage mapping diagrams
- **FDA cost-savings analysis**: 30-50% total cost reduction ($390M-$650M), 18-30 months timeline acceleration, 15-30% quality improvements
- **6 text-based diagrams**: Journey overview, regulatory mapping, data flow, Physical AI ecosystem, safety architecture, trial timeline
- **6 regulatory tables**: Patient demographics, regulatory framework, stage summary, adverse events, robot qualifications, treatment outcomes
- **Paper README** (`patient-journey/paper/README.md`): Compilation instructions and key results summary
- **Source archive** (`patient-journey/paper/Latex_Source_Code.zip`): Complete LaTeX source package

## Contributors
@kevinkawchak
@claude
@openai

## Notes

- Paper based on 3 Physical AI regulatory adaptations conducted by the author
- Not to be considered a new or approved regulatory paper
- Development by Claude Code Opus 4.6
- License: MIT (repository code), CC BY 4.0 (paper)

---

End-to-End Physical AI Oncology Clinical Trial Unification: Single-Patient Journey Orchestration
v2.6.0 - Draft release

Released on 20 March 2026
CEO Kevin Kawchak, ChemicalQDevice

## Summary

Publishes the **End-to-End Physical AI Oncology Clinical Trial Unification: Single-Patient Journey Orchestration**, a complete 10-stage patient journey system tracing Patient PAT-2026-0042 (58F, Stage IIIB NSCLC, ECOG 1, PD-L1 65%, TMB 14 mut/Mb, SITE-003) through a Physical AI oncology clinical trial. The system comprises 12 Python orchestrator modules, 30 ASCII progress diagrams, 10 Plotly chart generators, 6 text tables, an FDA cost-savings analysis, 4 guidance documents, and 262 tests. Three regulatory frameworks are implemented throughout: 21 CFR Part 312 Subpart J (§312.400-405), 21 CFR Part 50 Subpart C (§50.30-34), and ICH E6(R3) (§1.2-1.5, §2.8-2.12).

## Features

- **Central data model** (`patient-journey/patient_state.py`): 10 enums (PatientStage, PatientStatus, TreatmentArm, ResponseCategory, AESeverity, ConsentStatus, DataLockStatus, PhysicalAIClassification, USLReadinessLevel, MCPConformanceLevel), 14 dataclasses, legal stage transitions, and PatientJourneyState master class
- **Stage 1: Pre-Screening & Referral Intake** (Day -30 to Day -14): PHI detection, HIPAA Safe Harbor de-identification, ICD-10 to SNOMED harmonization, DICOM validation
- **Stage 2: Enrollment & Informed Consent** (Day -14 to Day 0): ICH E6(R3) consent elements, eligibility criteria evaluation, duplicate enrollment check, IRB review, stratified randomization
- **Stage 3: Digital Twin Construction** (Day 0 to Day 7): ASME V&V 40 validation, tumor microenvironment modeling, adaptive radiation simulation, virtual cohort analysis
- **Stage 4: Robot Qualification** (Day 7 to Day 13): USL scoring (4 dimensions: Autonomy, Dexterity, Safety, Interoperability), cross-framework validation, cybersecurity assessment, hand-eye calibration
- **Stage 5: Robot-Assisted Surgery** (Day 14): ROS 2 deployment, shared autonomy with Level 2 teleoperation, sensor fusion, sim-vs-real validation, specimen chain of custody per 21 CFR Part 11
- **Stage 6: Post-Operative Recovery** (Day 14 to Day 28): Pathology integration (pT2aN2M0), adverse event tracking (Day 16 atrial fibrillation Grade 2), Physical AI causality assessment
- **Stage 7: Immunotherapy Treatment** (Day 28 to Day 763): 35 cycles pembrolizumab 200mg q3w, adaptive dosing, cumulative toxicity tracking, hypothyroidism cycle 6, rash cycle 12, annual reporting
- **Stage 8: Federated Learning** (Day 28 to Day 763): 70 rounds federated averaging, differential privacy (epsilon=1.0, delta=1e-5), secure aggregation, DSMB safety reporting, site performance monitoring
- **Stage 9: Long-Term Surveillance** (Day 763 to Day 1858): Complete response (CR), quarterly imaging, recurrence risk modeling (35% to 3%), long-term safety monitoring
- **Stage 10: Trial Closeout** (Day 1858+): HARD_LOCK data integrity, re-identification risk validation (<0.04%), GCP audit, regulatory package generation, hazard ratio 0.62
- **Master orchestrator** (`patient-journey/master_journey.py`): Coordinates all 10 stages with regulatory mapping, stage result tracking, and journey reporting
- **30 ASCII progress diagrams**: 3 perspectives (timeline, regulatory, clinical) x 10 stages
- **Deliverables package**: 10 Plotly chart generators, 6 text tables, 6 high-level diagrams, FDA cost-savings analysis (15-25% cost reduction), 4 guidance documents (pharmaceutical industry, field observer, site activation, patient information)
- **262 tests** across 14 test modules: per-stage tests, master journey tests, cross-stage consistency tests, and deliverables tests

## Contributors
@kevinkawchak
@claude

## Notes
- Patient journey for PAT-2026-0042 (58F, Stage IIIB NSCLC, ECOG 1, PD-L1 TPS 65%, TMB 14 mut/Mb, SITE-003)
- Physical AI classifications: SURGICAL_ROBOT, COBOT, HUMANOID, THERAPEUTIC, DIAGNOSTIC, ASSISTIVE, REHABILITATIVE
- USL scoring: 4 dimensions (Autonomy, Dexterity, Safety, Interoperability), range 1.0-10.0; da Vinci Xi composite 7.9, Franka Emika composite 7.2
- MCP conformance levels: CORE, CLINICAL_READ, IMAGING, FEDERATED_SITE, ROBOT_PROCEDURE
- 21 CFR Part 11 compliant electronic signatures and audit trails
- Digital twin with ASME V&V 40 validation framework
- Federated learning with differential privacy (epsilon=1.0, delta=1e-5)
- Development by Claude Code Opus 4.6
- License: MIT (repository code)

---

End-to-End Physical AI Oncology Clinical Trial Unification: Adaption of 21 CFR Part 312 -- Investigational New Drug Application
v2.5.0 - March 18, 2026

## Summary

Publishes the **End-to-End Physical AI Oncology Clinical Trial Unification: Adaption of 21 CFR Part 312 -- Investigational New Drug Application**, an 94-page LaTeX document that modifies the prior 21 CFR Part 312 regulation in-place to incorporate Physical AI requirements throughout. The adaptation covers Subpart A (General Provisions with Physical AI scope expansion and 21 new definitions including USL, simulation validation, digital twin, and MCP), Subpart B (IND Application with Physical AI System Description as new IND section, Physical AI pre-clinical requirements, Physical AI amendments, Physical AI adverse event reporting, and Physical AI annual report supplements), Subpart C (Administrative Actions with Physical AI readiness requirements, 8 Physical AI grounds for clinical hold, Physical AI termination grounds, Physical AI dormancy/reactivation, and Physical AI meeting provisions), Subpart D (Responsibilities of Sponsors and Investigators with 7 Physical AI sponsor responsibilities, CRO transfer requirements, Physical AI investigator qualifications, 7 Physical AI record categories, Physical AI investigator responsibilities including informed consent, and Physical AI disqualification grounds), Subpart E (Drugs Intended to Treat Life-threatening and Severely-debilitating Illnesses with Physical AI accelerated development, early consultation, treatment protocols, risk-benefit analysis, Phase 4 studies, active monitoring, and patient safety safeguards), Subpart F (Miscellaneous with Physical AI import/export and supply chain security, foreign study acceptance, information disclosure, and 8 guidance document topics), Subpart G (Drugs for Investigational Use in Laboratory Research with Physical AI pre-clinical testing provisions), Subpart H [Reserved], Subpart I (Expanded Access with Physical AI submission and safeguard requirements), a new Subpart J (Physical AI Systems in Clinical Investigations with 3-tier risk classification, comprehensive validation requirements, cybersecurity by design, human oversight with emergency stop specifications, and AI/ML lifecycle management), and a 42-reference bibliography across 7 categories. The document is compiled to 94 pages from 2,275 lines of LaTeX source.

## Features

- **Complete LaTeX adaptation document** (`regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.tex`): 94 pages compiled, Subparts A-J with Physical AI modifications and new Subpart J
- **Subpart A: General Provisions**: 21 CFR 312.1 Scope expanded for Physical AI systems with 5 system types; 21 CFR 312.2 Applicability with Physical AI exemption criteria; 21 CFR 312.3 Definitions with 21 new Physical AI definitions (USL, simulation validation, digital twin, MCP, PCCP, sim-to-real gap, etc.); 21 CFR 312.6 Labeling with Physical AI system labeling; 21 CFR 312.7 Promotion with Physical AI system promotion restrictions; 21 CFR 312.8 Charging with Physical AI cost recovery; 21 CFR 312.10 Waivers with Physical AI-specific waivers
- **Subpart B: IND Application**: 21 CFR 312.20 IND requirements with Physical AI system documentation; 21 CFR 312.21 Phases with Physical AI requirements per phase (Phase 1 single-site/single-system, Phase 2 multi-site, Phase 3 full deployment); 21 CFR 312.22 General principles with Physical AI data integrity; 21 CFR 312.23 IND Content with new section (g) Physical AI System Description (7 subsections: system architecture, simulation validation, cybersecurity, human oversight, USL assessment, PCCP, MCP); 21 CFR 312.30-312.33 Amendments and reports with Physical AI provisions; 21 CFR 312.38 Withdrawal with Physical AI decommissioning
- **Subpart C: Administrative Actions**: 21 CFR 312.40 with Physical AI readiness requirements (USL verification, pre-procedure safety matrix, MCP infrastructure); 21 CFR 312.42 with 8 Physical AI grounds for clinical hold (robotic safety failure, AI model degradation, simulation-reality divergence, cybersecurity compromise, USL score decline, inadequate system description, digital twin failure, human oversight failure); 21 CFR 312.44 with Physical AI termination grounds; 21 CFR 312.45 with Physical AI dormancy and reactivation; 21 CFR 312.47-312.48 with Physical AI meetings and dispute resolution
- **Subpart D: Responsibilities**: 21 CFR 312.50 with 7 Physical AI sponsor responsibilities; 21 CFR 312.52 CRO transfer with Physical AI obligations; 21 CFR 312.53 with Physical AI investigator qualifications; 21 CFR 312.57 with 7 Physical AI record categories (deployment, maintenance, simulation, telemetry, USL, cybersecurity, training); 21 CFR 312.60 with 7 Physical AI investigator responsibilities including informed consent; 21 CFR 312.69 with Physical AI controlled substance safeguards; 21 CFR 312.70 with Physical AI disqualification grounds
- **Subpart E: Drugs Intended to Treat Life-threatening Illnesses**: 21 CFR 312.80-312.88 adapted with Physical AI provisions for accelerated development pathways, early consultation on simulation validation and PCCP, treatment protocols with USL thresholds, risk-benefit analysis including Physical AI safety records, Phase 4 post-market Physical AI monitoring, active monitoring of Physical AI clinical performance, and comprehensive patient safety safeguards
- **Subparts F-G, I**: Subpart F: Import/export with Physical AI supply chain security, foreign studies with USL assessment comparability, public disclosure with Physical AI confidential information, correspondence, and 8 Physical AI guidance document topics; Subpart G: laboratory research drugs with Physical AI pre-clinical testing provisions; Subpart H [Reserved]; expanded access with Physical AI provisions for individual, intermediate, and treatment use
- **Subpart J: Physical AI Systems (NEW)**: 21 CFR 312.400-312.405 establishing comprehensive Physical AI regulatory framework: 3-tier risk classification (Class I Assistive, Class II Collaborative, Class III Supervised Autonomous); validation (simulation, bench, integration, sim-to-real gap, site IQ/OQ/PQ, ongoing); cybersecurity by design (MFA, encryption, network segmentation, SBOM, incident response); human oversight (class-based levels, 1:1 operator ratio, fatigue management, hardware-independent e-stop <500ms); lifecycle management (configuration, AI/ML model management with drift monitoring, decommissioning)
- **References and Bibliography**: 42 references across 7 categories (primary regulatory sources, FDA guidance, robotics standards, simulation literature, oncology robotics, AI/ML clinical trials, cybersecurity, digital twins)
- **Source archive** (`regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.zip`): .tex, .sty, .bib, .pdf, and prompts.md files
- **Development prompts archive** (`regulatory/Adaption-21-CFR-Part-312/source/prompts.md`)

## Contributors
@kevinkawchak
@claude

## Notes
- Adapted from the prior 21 CFR Part 312 regulation (public domain under 17 U.S.C. section 105)
- Source repositories: physical-ai-oncology-trials v2.4.0, national-mcp-pai-oncology-trials v1.2.0
- No Python code changes -- documentation-only release
- Development by Claude Code Opus 4.6
- License: MIT (repository code)
- The original 21 CFR Part 312 regulation spans approximately 14,000 words across 60 sections and 9 subparts; manually adapting each section with technically consistent Physical AI provisions, cross-references, and a new subpart would require an estimated 200-400 hours of specialized regulatory writing and review by a team with combined FDA regulatory, robotics engineering, and AI/ML expertise
- The 2,275-line LaTeX document with 94 compiled pages, 42 bibliography references, and internally consistent cross-references across 10 subparts was produced in approximately 2 hours of Claude Code processing time, representing a roughly 100-200x acceleration over traditional regulatory drafting workflows
- At typical regulatory consulting rates ($300-500/hour for FDA regulatory affairs specialists with robotics domain expertise), the manual equivalent would cost an estimated $60,000-200,000 for initial drafting alone, excluding iterative review cycles, legal review, and formatting
- The adaptation required simultaneous expertise in FDA IND regulations (21 CFR Part 312), robotic surgery systems, AI/ML lifecycle management, cybersecurity frameworks (NIST), simulation physics engines, and clinical trial design -- a combination of specializations that would typically require a multi-disciplinary team of 4-6 subject matter experts

---

End-to-End Physical AI Oncology Clinical Trial Unification: Adaption of 21 CFR Part 50 -- Protection of Human Subjects
v2.4.0 - March 16, 2026

## Summary

Publishes the **End-to-End Physical AI Oncology Clinical Trial Unification: Adaption of 21 CFR Part 50 -- Protection of Human Subjects**, a 37-page LaTeX document that modifies the prior 21 CFR Part 50 regulation in-place to incorporate Physical AI requirements throughout. The adaptation covers Subpart A (General Provisions with Physical AI scope expansion and 17 new definitions), Subpart B (Informed Consent with 8 Physical AI consent elements and MCP consent tracking), a new Subpart C (Additional Protections for Subjects in Physical AI Clinical Investigations with 5 new sections covering safety requirements, IRB review, ongoing consent, data protection, and system classification), and Subpart D (Additional Safeguards for Children with Physical AI adaptations for pediatric populations). The document includes a 30-definition glossary and 19-reference bibliography. Formatting follows the same style as the ICH E6(R3) adaptation (v2.2.0). The repository README, regulatory README, CHANGELOG, and other documentation are updated for v2.4.0.

## Features

- **Complete LaTeX adaptation document** (`regulatory/Adaption-21-CFR-Part-50/source/Physical_AI_21_CFR_Part_50.tex`): 37 pages compiled, Subparts A-D with Physical AI modifications and new Subpart C
- **Subpart A: General Provisions**: §50.1 Scope expanded to Physical AI systems (autonomous surgical robots, therapeutic positioning systems, diagnostic needle-placement platforms, rehabilitative exoskeletons, companion monitoring systems); §50.3 Definitions with 18 original CFR definitions modified and 17 new Physical AI definitions added
- **Subpart B: Informed Consent**: §50.20 General Requirements adapted for Physical AI interactions; §50.22 Exception for Minimal Risk with Physical AI risk mapping; §50.23 Exception from General Requirements with Physical AI emergency and military provisions; §50.24 Exception for Emergency Research with Physical AI community consultation; §50.25 Elements of Informed Consent with 8 basic, 6 additional, and 8 Physical AI-specific consent elements; §50.27 Documentation of Informed Consent with MCP consent tracking (5 servers, 23 tools)
- **Subpart C: Additional Protections for Physical AI Investigations** (new): §50.30 Physical AI System Safety Requirements (pre-procedure safety matrix, runtime monitoring, post-procedure reporting); §50.31 IRB Review of Physical AI Investigations; §50.32 Ongoing Consent and Subject Notification; §50.33 Data Protection (HIPAA Safe Harbor, RBAC, hash-chained audit trails, federated learning); §50.34 Physical AI System Classification and Regulatory Pathways (510(k), De Novo, PMA, Breakthrough)
- **Subpart D: Additional Safeguards for Children**: §50.50-§50.56 adapted with Physical AI requirements for pediatric populations, including USL minimum thresholds, pediatric-specific safety protocols, and companion robot provisions
- **Glossary**: 30 Physical AI-specific definitions (Agentic AI, Cobot, Digital Twin, Federated Learning, MCP, USL, etc.)
- **Custom style package** (`Physical_AI_21_CFR_Part_50.sty`): CFRBlue color scheme, fancy headers, section formatting adapted from ICH E6(R3) style
- **Bibliography** (`Physical_AI_21_CFR_Part_50.bib`): 19 BibTeX entries covering CFR Part 50, both repositories, ICH E6(R3), FDA guidance, simulation frameworks, safety standards, MCP
- **Compiled PDF** (`Physical_AI_21_CFR_Part_50.pdf`): 37-page compiled document
- **Source archive** (`Physical_AI_21_CFR_Part_50.zip`): .tex, .sty, .bib, and .pdf files
- **Source README** (`README.md`): Build instructions, document structure, version info
- **Cover page**: Title, date "16 March 2026", DOI hyperlink to 10.5281/zenodo.19040707, CEO Kevin Kawchak, ChemicalQDevice, San Diego California, Claude Code attribution
- **No em dashes**: Entire document uses hyphens and "to" ranges per style requirements

## Contributors
@kevinkawchak
@claude

## Notes
- DOI: [10.5281/zenodo.19040707](https://doi.org/10.5281/zenodo.19040707)
- Adapted from the prior 21 CFR Part 50 regulation (public domain under 17 U.S.C. §105)
- Source repositories: physical-ai-oncology-trials v2.3.0 (DOI: 10.5281/zenodo.18445179), national-mcp-pai-oncology-trials v1.2.0 (DOI: 10.5281/zenodo.18869776)
- No Python code changes -- documentation-only release
- Development by Claude Code Opus 4.6
- License: MIT (repository code)

---

Physical AI Oncology Trial Industry Specification (PAIOTIS) v1.0
v2.3.0 - March 13, 2026

## Summary

Publishes the **Physical AI Oncology Trial Industry Specification (PAIOTIS) v1.0**, a formal 25-page LaTeX document that unifies four kevinkawchak repositories into a single industry standard. The specification uses RFC 2119 normative language (SHALL, SHOULD, MAY) throughout and covers 8 parts: Industry Definition and Scope, Technical Architecture, Regulatory Compliance Framework, Privacy and Data Governance, Robot Qualification and Certification, Pharmaceutical Sponsor Implementation Guide, Clinical Site Readiness Criteria, and Industry Milestone Roadmap. The document integrates content from physical-ai-oncology-trials v2.2.0, mcp-pai-oncology-trials/TrialMCP, national-mcp-pai-oncology-trials v1.2.0, and pai-oncology-trial-fl v1.1.1. Adapted from the Overleaf UTB thesis template by Edwin Puertas (CC BY 4.0) for industry specification use.

## Features

- **Complete LaTeX industry specification** (`unification/industry/paiotis_v1.tex`): 8 parts with RFC 2119 normative language, cover page, table of contents, normative language notice, and back matter
- **Part I: Industry Definition and Scope**: Physical AI oncology trial industry definition, stakeholder matrix (6 stakeholder types), normative references (12 standards/specifications)
- **Part II: Technical Architecture**: Three-layer architecture (Physical AI Layer, MCP Protocol Layer, Clinical Trial Layer), MCP server architecture (5 server types), simulation bridge architecture (Isaac Lab/MuJoCo bidirectional), digital twin pipeline
- **Part III: Regulatory Compliance Framework**: ICH E6(R3) adaptation, FDA submission pathways (510(k), De Novo, PMA, Breakthrough), PCCP for AI/ML model updates, IEC 80601 robot-specific compliance, risk classification table
- **Part IV: Privacy and Data Governance**: HIPAA Safe Harbor (18 identifiers), differential privacy (epsilon-delta), RBAC implementation, 21 CFR Part 11 electronic records, federated learning privacy with FedAvg/FedProx/SCAFFOLD
- **Part V: Robot Qualification and Certification**: USL methodology (4 dimensions x 25% weight), USL score bands table (5 bands), baseline scores for all 9 evaluated robots, qualification tiers by trial phase, re-qualification requirements
- **Part VI: Pharmaceutical Sponsor Implementation Guide**: 3-tier adoption pathways (observer/pilot/full integration), commercial value proposition, development stage integration, CRO partnership model
- **Part VII: Clinical Site Readiness Criteria**: Computational/network/physical infrastructure requirements, staffing table (7 roles), patient education framework, 8 e-stop implementations, 6-stage federation onboarding
- **Part VIII: Industry Milestone Roadmap**: Phase 1 (2026), Phase 2 (2027), Phase 3 (2028+), cross-repository dependency table (4 repositories)
- **Custom style package** (`paiotis.sty`): Adapted from UTB thesis template with Times Roman, PAIBlue color scheme, custom normative commands
- **Bibliography** (`references.bib`): 24 BibTeX entries covering all 4 repositories, ICH E6(R3), FDA guidance, ISO/IEC standards, RFC 2119, simulation frameworks
- **Compiled PDF** (`paiotis_v1.pdf`): 25-page compiled document
- **Source archive** (`paiotis_v1.zip`): .tex, .sty, .bib, and .pdf files
- **Prompts archive** (`unification/industry/prompts.md`): v2.3.0 development prompt
- **No em dashes**: Entire document uses hyphens and "to" ranges per style requirements
- **Cover page**: Title, date "13 March 2026", DOI hyperlink, CEO Kevin Kawchak, ChemicalQDevice, San Diego California, Claude Code attribution

## Contributors
@kevinkawchak
@claude

## Notes
- DOI: 10.5281/zenodo.18445179 (repository)
- Adapted from Overleaf UTB thesis template by Edwin Puertas (CC BY 4.0)
- RFC 2119 normative language used throughout (SHALL, SHOULD, MAY)
- All 9 USL-evaluated robots included with baseline scores
- Four repositories unified: physical-ai-oncology-trials, TrialMCP, national-mcp-pai-oncology-trials, pai-oncology-trial-fl
- No Python code changes -- documentation-only release
- Development by Claude Code Opus 4.6
- License: MIT (repository code), CC BY 4.0 (LaTeX style adaptation)

---

End-to-End Physical AI Oncology Clinical Trial Unification Guidance
v2.2.0 - March 12, 2026

## Summary

Publishes the **End-to-End Physical AI Oncology Clinical Trial Unification** guidance, a comprehensive LaTeX document adapting the prior ICH E6(R3) regulation for physical AI oncology clinical trials. The guidance covers Sections 1 through 4 (Principles, Investigator Responsibilities, Sponsor Responsibilities, Data Governance), Appendices A through C (Physical AI System Documentation, Clinical Trial Protocol, Essential Records), and a specialized Glossary with 30 physical AI-specific definitions. The document integrates USL scoring (v1.4.0 through v1.8.0) for all 9 evaluated robot platforms, references all simulation frameworks (NVIDIA Isaac Lab v2.3.1, MuJoCo v3.4.0, Gazebo v10.0.0, PyBullet v3.2.5), AI/ML categories (generative, agentic, RL, self-supervised, supervised), digital twin capabilities, federated learning, and privacy/regulatory compliance tools from the repository. Throughout the guidance, the prior ICH E6(R3) regulation is consistently referenced as the baseline being adapted. The repository README, regulatory README, CHANGELOG, and other documentation are updated for v2.2.0.

## Features

- **Complete LaTeX guidance document** (`regulatory/adaption-ich-e6r3/source/main.tex`): 4 major sections, 3 appendices, glossary, and bibliography adapted from prior ICH E6(R3) for physical AI oncology trials
- **Section 1: Principles of Physical AI Clinical Practice**: Foundational principles, robot classification (7 categories), AI/ML framework requirements (5 types), simulation and digital twin requirements, USL framework overview
- **Section 2: Investigator Responsibilities**: Qualifications, resources, medical care, IRB communication, informed consent for physical AI interactions, safety reporting, oversight
- **Section 3: Sponsor Responsibilities**: Quality management, regulatory submission, monitoring, noncompliance, safety assessment, data handling, clinical trial reports
- **Section 4: Data Governance**: Blinding in physical AI systems, data lifecycle (capture, metadata, review, corrections, transfer, finalisation, retention, destruction), computerised systems (procedures, training, security, validation, system failure, user management)
- **Appendix A: Physical AI System Documentation**: System description, specifications, safety studies, clinical experience (analogous to Investigator's Brochure)
- **Appendix B: Clinical Trial Protocol**: Protocol template adapted for physical AI trials with B.1 through B.16 sections
- **Appendix C: Essential Records**: Physical AI essential records criteria and table with 20 record categories
- **Glossary**: 30 physical AI-specific definitions (Agentic AI, Cobot, Digital Twin, Federated Learning, USL, VLA Model, etc.)
- **Updated style package** (`ich_guideline_style.sty`): Adapted headers, metadata, and hyperlink colors for physical AI guidance
- **Updated bibliography** (`references.bib`): 18 references covering ICH E6(R3), repository, USL paper, patient instructions, NASA TRL, MLTRL, simulation frameworks, AI frameworks, and regulatory standards
- **Prompts archive** (`regulatory/adaption-ich-e6r3/prompts.md`): v2.2.0 development prompt
- **Updated regulatory README**: Added adaption-ich-e6r3 directory to structure, updated version
- **Updated source README**: Build instructions, version info, DOI reference
- **Cover page**: Title, adaption line, guideline name, Modified E6(R3), draft release date, Zenodo DOI hyperlink, CEO attribution, ICH copyright and attribution text
- **Repository version references**: v1.0.0 through v2.2.0 referenced strategically throughout
- **USL scores**: All 9 robots referenced (da Vinci 7.1, Panda 7.4, Atlas 5.8, Gen3 5.7, Hugo 4.5, Digit 4.2, Optimus 3.6, Versius 3.4, xArm 3.4)
- **No em dashes**: Entire document uses hyphens and "to" ranges per style requirements
- **DOI**: 10.5281/zenodo.18973368

## Contributors
@kevinkawchak
@claude

## Notes
- Guidance DOI: [10.5281/zenodo.18973368](https://doi.org/10.5281/zenodo.18973368)
- Adapted from the prior ICH E6(R3) regulation (adopted 06 January 2025)
- Not endorsed or sponsored by ICH
- Development by Claude Code Opus 4.6
- License: MIT (repository code)
- The original .tex is longer than the prior ICH E6(R3) LaTeX reconstruction
- Compiled PDF and source zip included in repository

---

Patient Instructions: Physical AI Oncology Trials -- Paper Content Context Update and Documentation Restructure
v2.1.0 - March 2, 2026

## Summary

Updates the repository documentation to accurately reflect the content of the 10-page *Patient Instructions: Physical AI Oncology Trials* paper. The prior v2.0.0 documentation focused on file relocation to external hosting and mixed in context from v1.9.0 and v1.9.1, without capturing the actual paper content. This release adds page-by-page patient instructions, robot category text diagrams, quantitative patient data tables, procedure time comparisons, cancer type distribution diagrams, source distribution charts, and PDF image descriptions. Tables and text diagrams now focus on the paper's clinical content rather than file management operations. The main README, patients/README.md, and all relevant documentation have been updated to correctly reference the paper title *Patient Instructions: Physical AI Oncology Trials* (generated by ChatGPT, March 1, 2026).

## Features

- **Complete patients/README.md rewrite**: Replaces file-transfer-focused documentation with paper content including:
  - Page layout text diagram showing the consistent structure across all 10 pages
  - Robot type overview table with page numbers, cancer types, estimated times, and sources
  - Robot categories text diagram organizing 10 types into 5 clinical categories (surgical, therapeutic, diagnostic, assistive, rehabilitative)
  - Procedure time comparison bar chart (text diagram) across all 10 robot types
  - Full page-by-page content with introduction sentences and 3-step instructions for each robot type
  - Patient interaction summary text diagram showing the arrival/during/conclusion flow
  - Quantitative patient data table (anesthesia, physical contact, key measurements, recovery)
  - Source distribution text diagram (7 commercial companies, 3 ISO standards)
  - Cancer type distribution text diagram (8 adult cancers, 2 pediatric cancers)
  - PDF image descriptions linking each of the 5 images to their corresponding 2 pages
- **Corrected paper title**: Updated from "Patient-Robot Instructions" to "Patient Instructions: Physical AI Oncology Trials" matching the actual paper
- **Updated main README.md**: v2.1.0 patients section with robot categories text diagram, source column in overview table, and link to detailed documentation
- **Updated repository structure**: patients/ directory description updated to reflect content focus
- **Updated version references**: Badge, citation, and footer updated to v2.1.0
- **Paper access links preserved**: Zenodo DOI and Google Drive links maintained in URL format
  - Paper (PDF): [Zenodo DOI 10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)
  - LaTeX Source Files: [Zenodo DOI 10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)
  - Images: [Google Drive](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax)
- **Updated CHANGELOG.md**: Added v2.1.0 entry
- **Updated CITATION.cff**: Version updated to 2.1.0
- **Updated prompts archive**: Added v2.1.0 prompt to `patients/prompts/prompts.md`

## Contributors
@kevinkawchak
@claude
@openai

## Notes
- Paper DOI: [10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)
- Google Drive images: [Google Drive](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax)
- Paper generated by ChatGPT (March 1, 2026); repository documentation by Claude Code Opus 4.6
- No Python code changes — documentation-only release
- License: CC BY 4.0 (paper and images), MIT (repository code)
- 7 new text diagrams added to patients/README.md (page layout, robot categories, procedure times, interaction summary, quantitative data, source distribution, cancer distribution)
- Development by Claude Code Opus 4.6

---

Patient-Robot Instructions: Physical AI Oncology Trials — Hyperlink-Only References and Site-Wide Documentation Restructure
v2.0.0 - March 2, 2026

## Summary

Major release that transitions the patient-robot instruction materials to hyperlink-only references, reducing repository size by relocating paper PDFs, LaTeX source files, and images to external hosting (Zenodo and Google Drive). Includes a site-wide documentation restructure that moves detailed engineering example sections from the main README into their respective directory READMEs (`agentic-ai/`, `digital-twins/examples-twins/`, `examples/`, `examples-new/`, `tools/`, `federation/`). The main README now provides a consolidated engineering examples table linking to each directory. @kevinkawchak relocated files from v1.9.0 and v1.9.1 into Drive to reduce repository size. This is the second major release milestone, following v1.0.0 (February 2026).

## Features

- **Hyperlink-only patient-robot instructions**: Paper, LaTeX source files, and images are now referenced via hyperlinks only — no binary files in the repository
  - Paper (PDF): [Zenodo DOI 10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)
  - LaTeX Source Files: [Zenodo DOI 10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)
  - Images: [Google Drive](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax)
- **Repository size reduction**: @kevinkawchak relocated paper PDFs, LaTeX source, illustrations, and images from v1.9.0 and v1.9.1 into Google Drive
- **Site-wide documentation restructure**: Engineering example sections relocated from main README to directory-specific READMEs:
  - Agentic AI Engineering Examples → `agentic-ai/README.md` (new file)
  - Digital Twin Engineering Examples → `digital-twins/examples-twins/README.md`
  - Comprehensive Examples → `examples/README.md`
  - Physical Robot Engineering Examples → `examples-new/README.md`
  - Command-Line Tools → `tools/README.md`
  - Multi-Site Federated Oncology Trial Coordination → `federation/README.md`
- **Consolidated examples table**: Main README now links to all 34 examples and 5 CLI tools via a single summary table
- **Updated patients/README.md**: v2.0.0 documentation with paper, LaTeX, and image hyperlinks, prior version history, and updated directory structure
- **Updated version references**: Badge updated to v2.0.0, Citation.cff version updated, Actively Maintained Repositories date range extended to March 2026
- **Regulatory Compliance Framework date updated**: March 2026
- **v1.0.0 reference**: Main README now references both v1.0.0 and v2.0.0 major releases
- **Federation README updated**: Added examples table from main README
- **Updated CHANGELOG.md**: Added v2.0.0 entry
- **Updated prompts archive**: Added v2.0.0 prompt to `patients/prompts/prompts.md`

## Contributors
@kevinkawchak
@claude
@openai

## Notes
- Paper DOI: [10.5281/zenodo.18810541](https://doi.org/10.5281/zenodo.18810541)
- Google Drive images: [Google Drive](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax)
- Second major release (v2.0.0) following v1.0.0 (February 2026)
- No Python code changes — documentation-only release
- License: CC BY 4.0 (paper and images), MIT (repository code)
- Development by Claude Code Opus 4.6

---

Patient-Robot Instructions: AI Oncology Trials — New Images and Streamlined Instructions
v1.9.1 - March 1, 2026

## Summary

Updates the 10-page patient-facing instructional PDF with new images from Google Drive, a streamlined 3-step interaction format with quantitative data (minutes, distances, forces), corrected URLs for all bibliography sources, abbreviated clickable source links, and a reorganized file structure. Each robot type is now paired with a specific cancer type. The v1.9.0 materials (Cairo illustrations, generators) are archived under `patients/research/`. Three PDF versions are provided: full-size, 10 MB, and 5 MB.

## Features

- New images from [Google Drive](https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax) numbered 1--10, occupying the largest portion of each page
- Streamlined instruction format: 1 introductory sentence + 3-item numbered list per page (entering, interacting, concluding)
- Each robot paired with a specific cancer type (prostate, breast, lung, liver, pediatric leukemia, pediatric bone, pancreatic, thyroid, kidney, bone post-surgery)
- Title updated to "Patient-Robot Instructions: AI Oncology Trials - [Robot Type]" with abbreviations for long names
- Fixed all 7 source URLs (Intuitive Surgical, Franka Robotics, Accuray, SoftBank, Boston Dynamics, Varian, Ekso Bionics)
- Single DOI (10.5281/zenodo.18810541) throughout; removed duplicate DOI reference
- "For Demonstration Purposes Only" added to each page
- Three PDF versions: full-size, 10 MB target, 5 MB target
- `patients/images/` directory with numbered images and README with Drive link
- `patients/research/v1.9.0/` archive of prior version materials (SVG/PDF/PNG illustrations, Cairo generators)
- Updated LaTeX source, style, and bibliography (28 references with corrected URLs)
- Updated `patients/README.md` with v1.9.1 changes, new directory structure, and regeneration instructions
- PDF generated with Python reportlab + Pillow (replaces Cairo dependency)

## Contributors
@kevinkawchak
@claude

## Notes
- Paper DOI: 10.5281/zenodo.18810541
- Google Drive images: https://drive.google.com/drive/folders/1Cpe7fz3KlaERIfd6LQz2wmSBQNmB00Ax
- v1.9.0 materials preserved under patients/research/v1.9.0/ (except prompts/)
- License: CC BY 4.0 (paper and images), MIT (generation scripts)
- Development by Claude Code Opus 4.6

---

Patient-Robot Instructions: Physical AI Oncology Trials — Instructional Illustrations
v1.9.0 - February 28, 2026

## Summary

Publishes a **10-page patient-facing instructional PDF** with professional black-and-white portrait illustrations for physical AI oncology clinical trials. Each page is a self-contained instruction sheet for one of 10 robot types, showing a diverse patient interacting with the robot alongside detailed numbered instructions covering home preparation, entering the room, during interaction, concluding the session, and follow-up care. Pages 5 and 6 feature pediatric patients matched to child-appropriate robots (Social Companion Robots, Humanoids). All illustrations are generated using Python Cairo as high-resolution vector graphics and exported in SVG, PDF, and PNG formats. ISO 15223-1, ISO 20417, ISO 7000, IEC 60417, ISO 7010, and ISO 3864-1 standards are referenced for symbols and safety pictograms.

## Features

- `patients/paper/Patient-Robot Instructions: Physical AI Oncology Trials.pdf`: 10-page compiled PDF with header (author, ORCID, email), title with robot type, prominent black-and-white illustration, 5-section numbered instructions, and footer (date, DOI, model, page number, sources)
- `patients/paper/Latex Source Code.zip`: Archive containing 4 LaTeX source files (patient_robot_instructions.tex, patient_robot_instructions.sty, references.bib, README)
- `patients/paper/patient_robot_instructions.tex`: Main LaTeX document (10 pages, article class, 11pt, Times Roman)
- `patients/paper/patient_robot_instructions.sty`: Custom style package (geometry, fancyhdr, TikZ ISO symbols, enumitem)
- `patients/paper/references.bib`: BibTeX bibliography with 35 references (surgical robots, cobots, radiotherapy, needle placement, companion robots, humanoids, motion tracking, imaging, steerable needles, exoskeletons, ISO standards)
- `patients/paper/README`: Compilation instructions and content overview
- `patients/svg/`: 10 individual SVG vector illustrations (one per robot type)
- `patients/pdf/`: 10 individual PDF vector illustrations
- `patients/png/`: 10 individual PNG raster illustrations (3600×4000 pixels)
- `patients/generate_illustrations.py`: Cairo illustration generator for individual SVG/PDF/PNG files
- `patients/generate_pdf.py`: Combined 10-page PDF generator with full layout
- `patients/README.md`: Detailed documentation of the paper, directory structure, robot types, ISO standards, and regeneration instructions
- `patients/prompts/prompts.md`: Development prompt archive for v1.9.0
- Updated `releases.md`: Added v1.9.0 release notes
- Updated `CHANGELOG.md`: Added v1.9.0 entry
- Updated `README.md`: Updated version badge to v1.9.0, added patients section, updated repository structure

## Contributors
@kevinkawchak
@claude

## Notes
- Paper DOI: 10.5281/zenodo.18810541
- 10 robot types selected from 13 candidates; must-include: Cobots, Surgical Robots, Humanoids
- Excluded: Telepresence robots, AMRs, UV disinfection robots (limited direct patient interaction)
- Patient diversity across 10 pages: 9 distinct hair styles, 2 pediatric patients (pages 5--6)
- Quantitative patient guidance: estimated minutes, force values, distances, specific hand/body positions
- ISO standards: ISO 15223-1, ISO 20417, ISO 7000, IEC 60417 (symbols); ISO 7010, ISO 3864-1 (safety)
- Illustrations rendered with Python Cairo; LaTeX source provided as reference/alternative compilation path
- License: CC BY 4.0 (paper and illustrations), MIT (generation scripts)
- No Python module changes — CI lint/format checks addressed with ruff.toml per-file ignores
- Development by Claude Code Opus 4.6

---

Unification Standard Level for Physical AI Oncology Trials — Comprehensive Paper Publication
v1.8.0 - February 26, 2026

## Summary

Publishes the first comprehensive academic paper formalizing the **Unification Standard Level (USL)** framework for evaluating physical AI robot readiness for multi-site oncology clinical trials. The 9-page LaTeX paper covers all nine evaluated robots across three categories (cobots, surgical robots, humanoid robots), with complete quantitative scoring, code analysis, text diagrams, cross-category comparisons, and discussion of findings. All LaTeX source code is included as a zip archive alongside the compiled PDF.

## Features

- `unification/usl/paper/Unification Standard Level for Physical AI Oncology Trials.pdf`: 9-page compiled paper with Abstract, Table of Contents, Introduction, Methods, Results (all 9 robots with dimension-by-dimension score rationale), Discussion, Limitations and Future Work, Conclusion, References (28 citations), Acknowledgments, Ethical Disclosures, Rights and Permissions, and Citation
- `unification/usl/paper/Latex Source Code.zip`: Archive containing 4 LaTeX source files (usl_oncology_trials.tex, usl-oncology.sty, references.bib, README)
- `unification/usl/paper/usl_oncology_trials.tex`: Main LaTeX document (article class, 11pt, Times Roman)
- `unification/usl/paper/usl-oncology.sty`: Custom style package (geometry, colors, section formatting, code listings, TikZ score bars)
- `unification/usl/paper/references.bib`: BibTeX bibliography with 28 references (NASA TRL, MLTRL, TRL complex systems, oncology trials, simulation frameworks, AI frameworks, regulatory standards)
- `unification/usl/paper/README`: LaTeX compilation instructions and file descriptions
- Updated `unification/usl/prompts.md`: Added v1.8.0 USL Paper prompt on top
- Updated `releases.md`: Added v1.8.0 release notes in standard format
- Updated `CHANGELOG.md`: Added v1.8.0 entry
- Updated `README.md`: Updated version badge to v1.8.0, added paper reference in USL section, updated repository structure with paper directory

## Contributors
@kevinkawchak
@claude

## Notes
- Paper DOI: 10.5281/zenodo.18778220
- Paper format: Single-column, 11pt Times Roman, A4, with colored section headers, code listings, and tables
- All USL scores, dimension breakdowns, and robot specifications verified against repository source code
- Includes code snippets from usl_scoring_framework.py showing Dimension A computation
- Includes text diagrams showing cross-category Dim A and Dim D comparisons and phased trial timeline
- References are clickable with DOI links
- License: CC BY 4.0 (paper), MIT (repository code)
- No Python code changes — CI lint/format checks unaffected
- Development by Claude Code Opus 4.6

---

USL Restructure — Category-Specific READMEs and Cross-Category Diagrams
v1.7.0 - February 24, 2026

## Summary

Restructures the **Unification Standard Level (USL)** documentation into category-specific READMEs with dedicated text diagrams for each robot type. The main `unification/usl/README.md` is streamlined to contain only the USL standard overview, directory structure, influences, and references. All robot-specific evaluations, diagrams, and text are moved to new READMEs in `humanoids/`, `surgical/`, and `cobots/` subdirectories. The `unification/README.md` gains a link to USL and three new cross-category text diagrams covering USL results (with score rationale), meaning, and impact on the future of physical AI oncology trials. Each category README adds three new diagrams addressing results, meaning, and impact specific to that robot type, bringing the total from 9 to 18 text diagrams across the USL documentation.

## Features

- `unification/usl/humanoids/README.md`: New category README with 6 text diagrams (3 new: results with score rationale, meaning, impact; 3 moved: general comparison, technical specs, scoring breakdown), full Atlas/Digit/Optimus evaluations, quick start, contributing guide, and directory structure
- `unification/usl/surgical/README.md`: New category README with 6 text diagrams (3 new: results with score rationale, meaning, impact; 3 moved: general comparison, technical specs, scoring breakdown), full da Vinci/Hugo/Versius evaluations, quick start, contributing guide, and directory structure
- `unification/usl/cobots/README.md`: New category README with 6 text diagrams (3 new: results with score rationale, meaning, impact; 3 moved: general comparison, technical specs, scoring breakdown), full Franka/Kinova/xArm evaluations, quick start, contributing guide, and directory structure
- `unification/usl/README.md`: Streamlined to USL standard overview (scoring methodology, score bands, level definitions, robot categories table with links), directory structure (updated with README.md entries), influences, and references — robot-specific content moved to category READMEs
- `unification/README.md`: Added USL link at top, 3 new cross-category text diagrams (results summary with all 9 robots, meaning with key findings, impact with phased future timeline)
- Updated `unification/usl/prompts.md`: Added v1.7.0 USL Restructure prompt on top
- Updated `releases.md`: Added v1.7.0 release notes in new format (title without hashes)
- Updated `CHANGELOG.md`: Added v1.7.0 entry
- Updated `README.md`: Updated version to v1.7.0, updated repository structure to reflect new READMEs and prompts.md location

## Contributors
@kevinkawchak
@claude

## Notes
- Documentation restructure only — no Python code changes, no new modules
- 3 new category READMEs created (humanoids, surgical, cobots) with 9 new text diagrams (3 per category: results, meaning, impact)
- 3 new cross-category diagrams added to `unification/README.md`
- Total text diagrams in USL documentation: 18 (was 9)
- All robot evaluations, scores, and references preserved exactly from v1.6.0
- Quick start and contributing sections distributed to category READMEs
- `prompts.md` location confirmed at `unification/usl/prompts.md` (moved in v1.5.0)
- No Python files changed — CI lint/format checks unaffected
- Development by Claude Code Opus 4.6

---

## Unification Standard Level (USL) — Humanoid Robots
v1.6.0 - February 24, 2026

### Summary

Extends the **Unification Standard Level (USL)** framework to **Humanoid Robots** — a new robot category under `unification/usl/humanoids/`. Three bipedal humanoid robot systems from different manufacturers are evaluated: **Boston Dynamics Atlas (Electric)** (USL 5.8), **Agility Robotics Digit** (USL 4.2), and **Tesla Optimus (Gen 2)** (USL 3.6). Each system is scored across the same four dimensions (A–D) established for cobots and surgical robots: simulation framework switching, generative/agentic AI integration, cross-robot progress sharing, and multi-site clinical trial collaboration.

A new `usl_humanoid_scoring.py` scoring engine is created for humanoid robot evaluation with humanoid-specific criteria (whole-body locomotion, foundation model integration, bipedal navigation safety, hospital logistics tasks). The USL README is restructured to cover general, humanoid, surgical, and cobot information in that order, with 3 new text diagrams for humanoid robots (general, technical, scoring) bringing the total to 9 diagrams. Each humanoid robot has its own directory with comprehensive evaluation code including hardware specifications, kinematic models, locomotion profiles, oncology-specific task definitions, cross-organization sharing interfaces, and USL scoring.

### Features

- `unification/usl/humanoids/usl_humanoid_scoring.py`: USL scoring engine adapted for humanoid robots with `HumanoidType`, `HumanoidSimFramework` (8 frameworks including Drake), and `HumanoidAICapability` (12 capabilities including VLA, foundation model, whole-body control, locomotion/manipulation policy) enums; `HumanoidTask` (8 oncology tasks); `HumanoidDimAScore` through `HumanoidDimDScore` with humanoid-specific scoring criteria (whole-body model formats, locomotion/manipulation sim fidelity, foundation model integration, ISO 13482 alignment, autonomous navigation safety); `HumanoidUSLRating` with weighted score computation, comparison tables, gap analysis, and report generation
- `unification/usl/humanoids/boston_dynamics_atlas/boston_dynamics_atlas_usl.py`: Boston Dynamics Atlas (Electric) evaluation module — `AtlasElectricSpecs` (~1.5 m, ~89 kg, 28 DOF, custom electric actuators, stereo + LiDAR perception), `AtlasKinematics` with joint group definitions (head, torso, arms, legs) and joint limit validation, `AtlasLocomotionConfig` with hospital/logistics/outdoor profiles, `AtlasOncologyTask` definitions (supply transport, specimen delivery, equipment positioning, decontamination), `AtlasCrossOrgSharing` with Drake/BDAII/URDF/ONNX sharing methods; `AtlasUnifiedActionSpace` and `AtlasUnifiedObsSpace` for cross-platform normalization; USL score: 5.8
- `unification/usl/humanoids/tesla_optimus/tesla_optimus_usl.py`: Tesla Optimus (Gen 2) evaluation module — `OptimusGen2Specs` (~1.73 m, ~57 kg, 28 body DOF + 22 hand DOF, FSD-derived perception, Dojo training), `OptimusKinematics` with joint definitions including 11-DOF hands (5 finger types, 4 grasp types), `OptimusDeploymentProjection` timeline model (2025-2027), `OptimusOncologyTask` definitions (pharmacy delivery, linen transport, sample tray handling, equipment staging), `OptimusCrossOrgSharing` documenting fully proprietary ecosystem; USL score: 3.6
- `unification/usl/humanoids/agility_digit/agility_digit_usl.py`: Agility Robotics Digit evaluation module — `DigitSpecs` (~1.75 m, ~65 kg, 20 DOF, backward-bending knees, 16 kg payload, Jetson AGX Orin), `DigitKinematics` with backward-bending knee handling and spring energy computation, `GROOTIntegrationConfig` documenting NVIDIA GR00T N1 foundation model partnership, `DigitLocomotionConfig` with hospital/warehouse/campus profiles, `DigitOncologyTask` definitions (supply tote delivery, specimen courier, pharmacy restocking, waste collection), `DigitCrossOrgSharing` with NVIDIA/Amazon/DeepMind/OSU partnership ecosystem; USL score: 4.2
- `unification/usl/README.md`: Restructured with general USL information first, then humanoid robot evaluation (3 new text diagrams: general comparison, technical specifications, scoring breakdown), then surgical robot evaluation (3 existing diagrams renumbered 4-6), then cobot evaluation (3 existing diagrams renumbered 7-9), updated robot category table, updated directory structure, expanded references
- Updated `prompts.md`: Added v1.6.0 USL Humanoid Robots prompt
- Updated `releases.md`: Added v1.6.0 release notes
- Updated `CHANGELOG.md`: Added v1.6.0 entry
- Updated `unification/README.md`: Updated USL directory structure, added humanoid robot roadmap items
- Updated `README.md`: Added humanoid robot USL section, updated version to v1.6.0

### Contributors
@kevinkawchak
@claude

### Notes
- Three humanoid robots selected for: different manufacturers (Boston Dynamics, Agility Robotics, Tesla), same type (bipedal full-size humanoid), potential oncology logistics and assistive applications, and varying levels of open-source availability and AI integration
- Atlas (Electric) scores highest due to its advanced whole-body dynamics, 4-framework simulation support (Drake + Isaac Lab + MuJoCo + Gazebo), and BDAII research publications — however, its proprietary platform and lack of healthcare deployment limit sharing and clinical trial dimensions
- Digit benefits from GR00T N1 foundation model integration and commercial deployment experience (Amazon), but lacks healthcare-specific safety certifications
- Optimus scores lowest primarily due to its fully proprietary platform with no public SDK, simulation models, or developer ecosystem, despite having the most capable hands (11 DOF) and mass production potential
- All four USL dimensions (A–D) are adapted for humanoid-specific criteria: whole-body locomotion simulation, foundation model integration (GR00T, OpenVLA), bipedal navigation safety, hospital logistics tasks, ISO 13482 personal care robot safety
- All code passes `ruff check` and `ruff format --check` on Python 3.10–3.12
- 4 new Python modules totaling approximately 2,700 lines of code
- Development by Claude Code Opus 4.6

---

## Unification Standard Level (USL) — Surgical Robots
v1.5.0 - February 24, 2026

### Summary

Extends the **Unification Standard Level (USL)** framework to **Surgical Robots** — a new robot category under `unification/usl/surgical/`. Three teleoperated surgical robot systems from different manufacturers are evaluated: **Intuitive Surgical da Vinci (dVRK)** (USL 7.1), **Medtronic Hugo RAS** (USL 4.5), and **CMR Surgical Versius** (USL 3.4). Each system is scored across the same four dimensions (A–D) established for cobots: simulation framework switching, generative/agentic AI integration, cross-robot progress sharing, and multi-site clinical trial collaboration.

The existing `usl_scoring_framework.py` is moved under the `cobots/` directory, and a new `usl_surgical_scoring.py` is created for surgical robot evaluation. The USL README is restructured to cover general, surgical, and cobot information in that order, with 3 new text diagrams for surgical robots (general, technical, scoring). Each surgical robot has its own directory with comprehensive evaluation code including hardware specifications, kinematic models, simulation framework configurations, oncology-specific task definitions, cross-organization sharing interfaces, and USL scoring.

### Features

- `unification/usl/surgical/usl_surgical_scoring.py`: USL scoring engine adapted for surgical robots with `SurgicalSimFramework`, `SurgicalAICapability`, and `SurgicalProcedure` enums; `SurgicalDimAScore` through `SurgicalDimDScore` with surgical-specific scoring criteria (tissue deformation, instrument modeling, haptic feedback, surgical video AI, phase recognition, remote proctoring, IEC 80601 compliance); `SurgicalUSLRating` with weighted score computation, comparison tables, gap analysis, and report generation
- `unification/usl/surgical/intuitive_davinci/intuitive_davinci_usl.py`: Intuitive Surgical da Vinci (dVRK) evaluation module — `DVRKSpecs` with PSM/ECM/MTM configuration (7+1 DOF, 3 PSMs, stereo vision, EndoWrist articulation), `PSMKinematics` with remote center of motion (RCM) model and modified DH parameters (from Kazanzides et al., 2014), `DVRKFrameworkConfig` for 5 simulation frameworks (ORBIT-Surgical/Isaac Lab, SurRoL/PyBullet, AMBF, Gazebo, MuJoCo), `DVRKOncologyTask` definitions (tumor resection, lymph node dissection, suturing, biopsy), `DVRKCrossOrgSharing` with 5 sharing methods and 10 dVRK institution listing; USL score: 7.1
- `unification/usl/surgical/medtronic_hugo/medtronic_hugo_usl.py`: Medtronic Hugo RAS evaluation module — `HugoRASSpecs` with modular cart architecture (7 DOF per arm, open console, 8 mm instruments), `HugoArmKinematics` with DH parameters and joint validation, `TouchSurgeryInterface` with surgical phase recognition, performance metrics, and analytics, `HugoOncologyTask` definitions (colectomy, hysterectomy, prostatectomy, lymph node biopsy), `HugoCrossOrgSharing` with Medtronic ecosystem listing; USL score: 4.5
- `unification/usl/surgical/cmr_versius/cmr_versius_usl.py`: CMR Surgical Versius evaluation module — `VersiusSpecs` with biomimetic modular architecture (7 DOF, ~10 kg arms, 5 mm instruments, portable), `VersiusArmKinematics` with biomimetic DH parameters, `VersiusORSetup` configurations for 3 oncology specialties (gynecologic, colorectal, upper GI), `VersiusOncologyTask` definitions (hysterectomy, colectomy, gastrectomy, omentectomy), `VersiusCrossOrgSharing` with deployment regions; USL score: 3.4
- `unification/usl/README.md`: Restructured with general USL information first, then surgical robot evaluation (3 new text diagrams: general comparison, technical specifications, scoring breakdown), then cobot evaluation (original 3 diagrams preserved), robot category table, updated directory structure, expanded references
- Moved `unification/usl/usl_scoring_framework.py` → `unification/usl/cobots/usl_scoring_framework.py`
- Updated `prompts.md`: Added v1.5.0 USL Surgical Robots prompt
- Updated `releases.md`: Added v1.5.0 release notes
- Updated `CHANGELOG.md`: Added v1.5.0 entry
- Updated `unification/README.md`: Updated USL directory structure, added surgical robot roadmap items
- Updated `README.md`: Added surgical robot USL section, updated version to v1.5.0

### Contributors
@kevinkawchak
@claude

### Notes
- Three surgical robots selected for: different manufacturers, teleoperated MIS architecture, oncology surgical applications, and varying levels of open-source availability
- da Vinci (dVRK) scores highest due to its unique open-source ecosystem (dVRK, ORBIT-Surgical, SurRoL, AMBF) and extensive AI research community — no other surgical robot has comparable simulation and research infrastructure
- Hugo RAS and Versius score lower primarily due to proprietary platforms with limited open-source availability, which limits simulation switching, AI integration, and cross-robot sharing
- All four USL dimensions (A–D) are adapted for surgical robot-specific criteria: tissue deformation simulation, instrument articulation modeling, surgical video AI, phase recognition, remote proctoring, IEC 80601-2-77 compliance
- All code passes `ruff check` and `ruff format --check` on Python 3.10–3.12
- 4 new Python modules totaling approximately 2,400 lines of code
- Development by Claude Code Opus 4.6

---

## Unification Standard Level (USL) for Collaborative Robots
v1.4.0 - February 23, 2026

### Summary

Introduces the **Unification Standard Level (USL)** — a new scoring framework under `unification/usl/` for evaluating how ready physical AI robots are for deployment in unified, multi-site oncology clinical trials. USL scores range from 1.0 to 10.0 (in 0.1 increments) across four weighted dimensions: simulation framework switching, generative/agentic AI integration, cross-robot progress sharing, and multi-site clinical trial collaboration.

This initial release evaluates three state-of-the-art open-source collaborative robot arms from different manufacturers: **Franka Emika Panda** (Franka Robotics, USL 7.4), **Kinova Gen3 7DoF** (Kinova Robotics, USL 5.7), and **UFACTORY xArm 7** (UFACTORY, USL 3.4). Each cobot receives a comprehensive evaluation with hardware specifications, simulation framework configurations, kinematic validation tools, policy transfer interfaces, cross-organization sharing capabilities, and oncology-specific task definitions.

The USL framework is influenced by NASA/DOD TRL (Mankins, 2004), MLTRL (Lavin et al., 2021), TRL for complex systems (Tomaschek et al., 2015), and is inspired by LLM recommendations for oncology trials (Kawchak, 2025; DOI 10.5281/zenodo.17451709).

### Features

- `unification/usl/usl_scoring_framework.py`: Core USL scoring engine with four weighted dimensions (A–D), 10-level classification system, score band categorization, comparison tables, gap analysis, and JSON/text report generation
- `unification/usl/cobots/franka_panda/franka_panda_usl.py`: Franka Emika Panda evaluation module with hardware specs, DH parameters, URDF template generator, kinematic chain validator, policy transfer interface with 4 oncology tasks, cross-organization sharing manager, and simulation framework configurations for MuJoCo/Isaac Lab/Gazebo/PyBullet
- `unification/usl/cobots/kinova_gen3/kinova_gen3_usl.py`: Kinova Gen3 7DoF evaluation module with Kortex API abstraction layer, modified DH kinematic model, actuator module specifications, angular/Cartesian command interfaces, 4 oncology task definitions, and framework configurations for Gazebo/MuJoCo/Isaac Lab/PyBullet
- `unification/usl/cobots/ufactory_xarm7/ufactory_xarm7_usl.py`: UFACTORY xArm 7 evaluation module with xArm Python SDK abstraction, joint specifications with limit validation, error code mapping, 4 oncology lab automation tasks, intra-organization sharing across xArm family, and framework configurations
- `unification/usl/README.md`: Comprehensive USL standard documentation with scoring methodology, 10-level definitions, score bands, three text comparison diagrams (general, technical, scoring), individual cobot evaluations, references to TRL/MLTRL influences, and quick-start guide
- `prompts.md`: Development prompt archive for v1.4.0 USL standard creation
- `releases.md`: Release notes in standardized format
- Updated `unification/README.md`: Added USL directory to structure, added Q1 2026 USL roadmap items
- Updated `README.md`: Added USL section with cobot evaluation table, updated repository structure, updated version to v1.4.0
- Updated `CHANGELOG.md`: Added v1.4.0 entry
- Updated `ruff.toml`: Added per-file ignore for `unification/usl/**/*.py`

### Contributors
@kevinkawchak
@claude

### Notes
- USL framework is specific to this project — "Unification Standard Level" evaluates robot readiness for multi-site oncology trial unification, distinct from general-purpose TRL
- All four USL dimensions derive directly from the existing `unification/` pillars: `simulation_physics/`, `agentic_generative_ai/`, `cross_platform_tools/`, and the `federation/`+`regulatory/` directories
- The three evaluated cobots (Franka Panda, Kinova Gen3, xArm 7) were selected for: open-source availability, different manufacturers, MuJoCo Menagerie models, active ROS 2 support, and potential oncology applications
- All code passes `ruff check` and `ruff format --check` on Python 3.10–3.12
- 4 new Python modules totaling approximately 2,100 lines of code
- Development by Claude Code Opus 4.6
