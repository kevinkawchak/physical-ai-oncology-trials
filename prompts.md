# Prompts Archive

## v2.4.0 - End-to-End Physical AI Oncology Clinical Trial Unification: Adaption of 21 CFR Part 50 (March 16, 2026)

Your goal is to process the main prompt below using kevinkawchak/physical-ai-oncology-trials and kevinkawchak/national-mcp-pai-oncology-trials (make no changes to this second repository). You are responsible for comprehensive understanding and applying all aspects of the repositories to the new work. This includes, where relevant: all code, all code types, machine learning and AI types, different robot types and characteristics, digital twins, examples, patients, physical ai unification, unification standard level (USL), tools, etc. The new incorporations of physical AI must be consistently high quality throughout the entire new work. 

Be sure to refer to the work as an adaption, and not a new and approved regulatory framework. Utilize the same .tex and .sty formatting from kevinkawchak/physical-ai-oncology-trials/tree/main/regulatory/adaption-ich-e6r3/source template throughout the entire new work (source template is for formatting only, don’t utilize information from source template into this work)(don’t modify the source template). Your final deliverable will be the LaTeX files and pdf described in the main prompt. Use your full 1M context length throughout the outputs. Process through the first half of the .tex file. Then, Confirm when you are finished with the first half. I will then type Continue, and you will then finish the second half. Don’t ask questions or go into plan mode throughout this prompt.

Avoid large white empty spaces without text. Where large spacing between words exist throughout: modify raggedright spacing to make positioning between words look equally and properly spaced. Make sure text doesn’t run off the right side of the page anywhere. Avoid lines with a single word. Avoid single lines separate form the paragraph on the next page. Perform the final formatting steps that a senior author would take by correcting white space formatting, removing and/or adding relevant text to make each section and page look properly formatted and self standing by itself. (Don’t overcrowd the page with text, some white space formatting is ok). Make sure to correct all incorrect symbols such as SS into “§” where relevant. Only use em dashes for the CFR title.

“START COVER PAGE”
No logos are allowed throughout the work. Utilize the following text based on the same formatting from the source template.
“END-TO-END PHYSICAL AI ONCOLOGY CLINICAL TRIAL UNIFICATION” (1 Line, Top)
“Adaption: 21 CFR Part 50” (1 Line)
“Protection of Human Subjects” (1 Line)
“Modified ECFR” (1 Line)
“Draft release” (1 Line)
“Released on 16 March 2026” (1 Line)
“10.5281/zenodo.19040707” with hyperlink https://doi.org/10.5281/zenodo.19040707 (1 Line)
“CEO Kevin Kawchak, ChemicalQDevice” (1 Line) 
“The original 21 CFR Part 50 document is a work in the public domain under 17 U.S.C. § 105, and may be used, reproduced, incorporated into other works, adapted, modified, translated or distributed under a public license. 21 CFR Part 50 — Protection of Human Subjects.” “Source ECFR” Hyperlink: (https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-50) “This current work is not endorsed or sponsored by CFR. The original ECFR formatting was reconstructed into Markdown, and adapted further into LaTeX using Claude Code Opus 4.6.” (Several Lines Together, Bottom)
“END COVER PAGE”

Update physical-ai-oncology-trials main readme documentation, repository structures, text diagrams and toc (with correct urls and make sure it is in the right order), a link and explanation to this new GitHub pages, and other affected areas in the repository (this is the only repository that needs to be edited). Make sure the repository is fully up to date with this work regarding badges, content, and context.

Provide a copy of this v2.4.0 prompt under a new main/prompts.md. Be sure to fix and address errors that would cause failed checks for the single pull request (such as for lint and Python environment issues to avoid the following error during final checks): "3 failing checks
x Cl / lint-and-format (3.10) (pull...
x Cl / lint-and-format (3.11) (pull...
x Cl / lint-and-format (3.12) (pull... " Place the new release notes in releases.md under main using the format below. Update other relevant documentation such as project structures. Update the main Readme diagrams, repository structure, etc. where necessary. Provide an updated changelog (v2.4.0).

When you are finished, auto-push the update to GitHub on your own for my review. The user will then review your updates in GitHub prior to finalization.

"FORMAT"
Release title
v2.4.0 - [Fill in Title Here]

## Summary

## Features

## Contributors
@kevinkawchak
@claude

## Notes



“START MAIN PROMPT”
Your goal is to produce a new, complete set of LaTeX files and pdf that adapts 21 CFR Part 50 -- Protection of Human Subjects to incorporate Physical AI requirements throughout the entire regulation. This is not an appendix or addendum. You must base your new LaTeX files on the physical-ai-oncology-trials/regulatory/Adaption-21-CFR-Part-50
/843pm 14Mar26 21_CFR_Part_50.md (do not modify the original file) for every applicable section, definition, and provision of the original regulation so that Physical AI (autonomous robotic systems, digital twins, AI/ML agents, and simulation-validated platforms used in oncology clinical trials) is woven into the regulatory fabric end-to-end.

Source Repositories to Clone and Read
Clone both repositories and use them as your primary knowledge base:
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git 
git clone https://github.com/kevinkawchak/national-mcp-pai-oncology-trials.git 

Files You MUST Read Before Starting
Primary Source (the original regulation to be adapted):
* physical-ai-oncology-trials/regulatory/Adaption-21-CFR-Part-50/843pm 14Mar26 21_CFR_Part_50.md
Physical AI Framework Context (from physical-ai-oncology-trials):
* README.md — Repository overview, robot categories, AI/ML types, framework structure
* regulatory/README.md — Regulatory framework overview, FDA/IRB/ICH-GCP landscape
* regulatory/adaption-ich-e6r3/source/main.tex — The ICH E6(R3) Physical AI adaptation (Sections 1-4, Appendices A-C, Glossary); this is your primary model for how Physical AI language should be incorporated into regulatory text
* regulatory/adaption-ich-e6r3/prompts.md — Prior prompt methodology and formatting directives
* regulatory/human-oversight/HUMAN_OVERSIGHT_QMS.md — Human oversight controls, CRF/AE automation risk tiers, safety gates, QMS integration
* regulatory/fda-compliance/README.md — FDA submission pathways (510(k), De Novo, PMA, Breakthrough Device), PCCP, post-market surveillance
* regulatory/ich-gcp/README.md — ICH E6(R3) compliance verification, RBQM, digital technology provisions
* regulatory/irb-management/README.md — AI-specific IRB protocol review requirements
* regulatory/regulatory-intelligence/README.md — Multi-jurisdiction regulatory monitoring
* unification/usl/README.md — Unification Standard Level (USL) scoring framework, 9 robot evaluations across surgical/cobot/humanoid categories
* unification/README.md — Multi-organization cooperation model
* digital-twins/ — Read all READMEs in subdirectories (patient-modeling, treatment-simulation, clinical-integration)
* privacy/ — Read all READMEs (phi-pii-management, de-identification, access-control, breach-response, dua-templates)
* agentic-ai/ — Read README and examples for agentic AI workflows
* federation/ — Read README for federated learning multi-site coordination
National MCP Standard Context (from national-mcp-pai-oncology-trials):
* README.md — National standard overview
* spec/core.md — Protocol scope, design principles, error handling
* spec/conformance.md — Five cumulative conformance levels (Core through Robot Procedure)
* spec/tool-contracts.md — All 23 MCP tool definitions (inputs, outputs, errors, audit requirements)
* spec/actor-model.md — Six actors (Trial Coordinator, Data Monitor, Auditor, Sponsor, CRO, Robot Agent) and permission matrix
* spec/security.md — Deny-by-default RBAC, token lifecycle, input validation, SSRF prevention
* spec/privacy.md — HIPAA Safe Harbor de-identification, HMAC-SHA256 pseudonymization, minimum necessary standard
* spec/audit.md — Hash-chained immutable audit trail architecture, 21 CFR Part 11 compliance
* spec/provenance.md — DAG-based data lineage tracking, integrity verification
* regulatory/CFR_PART_11.md — 21 CFR Part 11 electronic records mapping to MCP servers
* regulatory/HIPAA.md — HIPAA Privacy/Security Rule mapping, all 18 PHI identifiers
* regulatory/US_FDA.md — FDA AI/ML SaMD classification, PCCP, submission pathways
* regulatory/IRB_SITE_POLICY_TEMPLATE.md — IRB site policy template for Physical AI trials
* profiles/robot-assisted-procedure.md — Level 5 robot procedure profile: robot capability registration, task-order lifecycle, safety matrix, USL integration, six-step robot agent workflow, forbidden operations
* profiles/base-profile.md — Minimum implementation requirements
* profiles/clinical-read.md — FHIR R4 clinical data access profile
* profiles/imaging-guided-oncology.md — DICOM imaging profile with RECIST
* profiles/multi-site-federated.md — Federated learning, differential privacy, cross-site audit
* profiles/country-us-fda.md — FDA-specific regulatory overlays
* docs/glossary.md — Standard terminology (Physical AI, USL, MCP, Robot Agent, FHIR, DICOM, RECIST, etc.)
* docs/architecture.md — Five-server MCP topology
* schemas/robot-capability-profile.schema.json — Robot registration schema
* schemas/task-order.schema.json — Procedure scheduling and lifecycle schema
* schemas/audit-record.schema.json — Hash-chained ledger entry schema
* schemas/consent-status.schema.json — Consent status tracking schema

Adaptation Requirements
1. Structural Approach
* Preserve the complete structure of 21 CFR Part 50 (Subpart A: General Provisions SS 50.1-50.3; Subpart B: Informed Consent SS 50.20-50.25).
* Do NOT simply append a "Physical AI" section at the end. Instead, modify each section in-place so that Physical AI considerations are integral to the regulation throughout.
* Where a section cannot be meaningfully adapted (e.g., pure statutory citations), prepend that section with "Prior 21 CFR Part 50: " to indicate it retains its original form.
* Add a new Subpart C -- Additional Protections for Subjects in Physical AI Clinical Investigations containing Physical AI-specific provisions that do not fit naturally into existing sections.
2. Section-by-Section Modification Directives
SS 50.1 Scope:
* Expand the scope to explicitly include clinical investigations involving Physical AI systems (autonomous surgical robots, therapeutic positioning systems, diagnostic needle-placement platforms, rehabilitative exoskeletons, companion monitoring systems).
* Reference the five robot types defined in the national MCP standard.
* State that this part applies to investigations where Physical AI systems perform, assist, or monitor clinical procedures on human subjects.
* Reference the Unification Standard Level (USL) as the readiness assessment framework for Physical AI platforms.
SS 50.3 Definitions:
* Retain all existing 18 definitions.
* Add new definitions derived from both repositories, including but not limited to:
    * Physical AI system — AI systems operating in the physical world through robotic platforms (surgical robots, therapeutic positioning systems, diagnostic needle-placement platforms, rehabilitative exoskeletons, companion monitoring systems) used in clinical investigations.
    * Robot agent — An autonomous Physical AI system executing clinical procedures within an oncology trial, interacting through MCP-mediated data access.
    * Unification Standard Level (USL) — A scoring framework (1.0-10.0) evaluating Physical AI platform readiness across simulation fidelity, safety controls, and clinical validation.
    * Digital twin — A patient-specific computational model used for treatment simulation, procedure planning, and real-time intraoperative guidance.
    * Simulation framework — A physics-based computational environment (e.g., NVIDIA Isaac Lab, MuJoCo, Gazebo, PyBullet) used for validating Physical AI system behavior before clinical deployment.
    * Robot capability profile — A machine-readable specification of a Physical AI platform's capabilities, safety prerequisites, USL score, and required MCP tools.
    * Task order — A scheduled clinical trial task assigned to a Physical AI system, including procedure type, robot assignment, and safety check requirements.
    * Emergency stop — An immediate halt capability that must be available for any Physical AI system during clinical procedures, with mandatory audit recording.
    * Human oversight — The requirement that qualified clinical personnel maintain supervisory authority over Physical AI systems during all phases of clinical investigation.
    * Hash-chained audit trail — An immutable, tamper-evident record of all Physical AI system actions using SHA-256 hash chains, satisfying 21 CFR Part 11.
    * Model Context Protocol (MCP) — An open protocol for connecting AI agents to external tools and data sources, implemented through five standardized servers (authz, fhir, dicom, ledger, provenance).
    * Deny-by-default authorization — A security model where all Physical AI system access to clinical data is denied unless explicitly permitted by policy.
    * HIPAA Safe Harbor de-identification — The removal or generalization of all 18 PHI identifiers before clinical data is provided to Physical AI systems.
    * Predetermined Change Control Plan (PCCP) — A documented plan for managing modifications to Physical AI systems used as or within medical devices, per FDA AI/ML guidance.
    * Federated learning — Multi-site AI model training without sharing raw patient data, using differential privacy and secure aggregation.
SS 50.20 General Requirements for Informed Consent:
* Incorporate requirements that informed consent for Physical AI investigations must include disclosure of the role of autonomous robotic systems, the degree of human oversight during procedures, simulation validation results, and USL readiness scores.
* Require disclosure of whether a Physical AI system will make autonomous clinical decisions or serve in an assistive capacity.
* Require disclosure of the specific robot type(s) to be used, their safety record, and emergency stop provisions.
* Require disclosure of digital twin usage for treatment planning if applicable.
* Require disclosure of how the subject's data will be processed through MCP servers with de-identification protections.
SS 50.22 Exception from Informed Consent for Minimal Risk:
* Define how minimal risk is assessed when Physical AI systems are involved (companion monitoring and low-risk rehabilitative applications may qualify; surgical and needle-placement robots generally do not).
* Map minimal risk thresholds to USL scores and robot types.
SS 50.23 Exception from General Requirements:
* Adapt life-threatening situation exceptions to address scenarios where a Physical AI system is the only available means of delivering urgent treatment.
* Address the military exception provisions in the context of Physical AI surgical and therapeutic systems deployed in military medical settings.
* Specify that even in emergency situations, Physical AI systems must maintain audit trails and safety monitoring.
SS 50.24 Exception from Informed Consent for Emergency Research:
* Adapt community consultation requirements to include public understanding of Physical AI systems.
* Address data monitoring committee responsibilities for Physical AI system performance metrics, USL score trends, and safety incident patterns.
* Require that family member/legally authorized representative notification includes information about Physical AI system involvement.
SS 50.25 Elements of Informed Consent:
* Modify each of the 8 basic elements to incorporate Physical AI considerations:
    1. Purpose: Include the role of Physical AI in the investigation's objectives.
    2. Procedures: Describe which procedures will involve Physical AI systems, the specific robot types, the degree of autonomy, and human oversight arrangements.
    3. Risks: Disclose risks specific to Physical AI (system malfunction, software errors, simulation-to-reality gaps, cybersecurity risks, force/torque exceedances, emergency stop scenarios).
    4. Benefits: Describe potential benefits of Physical AI (precision, consistency, reduced human fatigue, simulation-validated approaches, digital twin-guided treatment).
    5. Alternatives: Include non-Physical AI alternatives (conventional surgical or therapeutic approaches).
    6. Confidentiality: Explain MCP-mediated data access, HIPAA Safe Harbor de-identification, pseudonymization, deny-by-default authorization, hash-chained audit trails, and federated learning data protections.
    7. Compensation/treatment for injury: Address liability and compensation in cases where Physical AI system malfunction causes injury.
    8. Contact information: Include contacts for Physical AI system safety concerns in addition to standard investigator contacts.
* Add additional elements for Physical AI investigations:
    * The USL readiness score of the robot system assigned to the subject's procedure.
    * Whether the procedure was rehearsed in simulation and the simulation framework used.
    * Whether a digital twin model was created from the subject's data and how it will be used.
    * The six-step robot agent workflow that will govern the procedure.
    * The pre-procedure safety matrix checks that must pass before the procedure begins.
    * The subject's right to request a non-Physical AI alternative at any time.
    * How real-time safety monitoring will be conducted during the procedure.
    * Post-procedure provenance documentation (data lineage from clinical data through procedure to outcomes).
3. New Subpart C -- Additional Protections for Subjects in Physical AI Clinical Investigations
Create the following new sections:
SS 50.30 Physical AI System Safety Requirements:
* Pre-procedure safety matrix (authorization, patient identity verification, clinical data availability, imaging data access, robot readiness/USL threshold, environmental checks, simulation validation, digital twin sync).
* Runtime safety monitoring (continuous authorization, real-time audit recording, emergency stop, force/torque limit monitoring).
* Post-procedure requirements (complete audit trail, provenance chain, outcome recording, safety incident reporting, USL score update).
* Task-order lifecycle (scheduled -> safety_check -> in_progress -> completed/cancelled/failed).
* Forbidden operations (no procedure without safety check, no procedure without valid authorization, no skipping audit, no operation below minimum USL score, no direct clinical system modification by robot, no continuation after emergency stop without new safety check cycle).
SS 50.31 IRB Review of Physical AI Investigations:
* IRBs must evaluate Physical AI system safety profiles, USL scores, simulation validation reports, and human oversight plans.
* IRBs must assess the adequacy of informed consent language regarding Physical AI systems.
* IRBs must review the robot capability profile and verify minimum USL thresholds for the proposed procedure types.
* IRBs must have access to or expertise in Physical AI systems to evaluate protocols.
* Reference the IRB site policy template from the national MCP standard.
SS 50.32 Ongoing Consent and Subject Notification:
* Subjects must be notified of significant changes to Physical AI systems used in their care (software updates, model version changes, USL score changes).
* Subjects must be notified if a different robot platform than originally consented will be used.
* PCCP-governed changes to Physical AI systems that may affect the subject's risk profile require re-consent or notification per IRB-approved procedures.
SS 50.33 Data Protection for Physical AI Investigations:
* All clinical data accessed by Physical AI systems must be de-identified per HIPAA Safe Harbor (all 18 PHI categories).
* Pseudonymization must use HMAC-SHA256 with site-specific salts.
* Physical AI systems must operate under deny-by-default RBAC with explicit permission policies.
* All Physical AI system actions must be recorded in hash-chained immutable audit trails satisfying 21 CFR Part 11.
* Provenance tracking must document the complete data lineage for every robot-assisted procedure.
* Federated learning deployments must use differential privacy and secure aggregation to prevent data leakage across sites.
SS 50.34 Physical AI System Classification and Regulatory Pathways:
* Reference FDA AI/ML SaMD classification for Physical AI components.
* Reference applicable regulatory submission pathways (510(k), De Novo, PMA, Breakthrough Device Designation).
* Require PCCP documentation for adaptive Physical AI systems.
* Reference IEC 80601-2-77 (robot-assisted surgery safety), ISO 14971 (risk management), and ISO 13482 (personal care robot safety).
4. Formatting Requirements
* Maintain the same formatting conventions as the original 843pm 14Mar26 21_CFR_Part_50.md file.
* Include the centered attribution paragraph at the top, updated to reflect this is a Physical AI adaptation.
* Use the same section numbering scheme (SS 50.XX).
* No em dashes anywhere in the work (use double hyphens -- or rewrite).
* Balanced white space; no large empty gaps; no overcrowded sections.
* Each section should be self-standing and properly formatted.
* Include version numbers from both repositories where relevant (physical-ai-oncology-trials v2.3.0, national MCP standard v0.5.0, USL scores for reference robots).
5. Contextual Emphasis
* Throughout the document, make clear that the original 21 CFR Part 50 was the prior/previous regulation.
* Emphasize that this adaptation incorporating Physical AI requirements represents the new regulatory framework for protecting human subjects in investigations involving autonomous robotic systems, digital twins, and AI/ML agents in oncology clinical trials.
* Reference both source repositories with their DOIs where applicable:
    * Physical AI Oncology Trials: DOI 10.5281/zenodo.18445179
    * National MCP-PAI Standard: DOI 10.5281/zenodo.18869776
    * ICH E6(R3) Physical AI Adaptation: DOI 10.5281/zenodo.18973368
    * Federated Learning: DOI 10.5281/zenodo.18840880
6. Physical AI Concepts That Must Be Integrated Throughout
From physical-ai-oncology-trials:
* 7 robot categories, 9 evaluated robots (3 cobots: Franka Panda 7.4 USL, Kinova Gen3 5.7, xArm 7 3.4; 3 surgical: da Vinci dVRK 7.1, Hugo RAS 4.5, Versius 3.4; 3 humanoids: Atlas 5.8, Digit 4.2, Optimus 3.6)
* 5 AI/ML types (generative, agentic, reinforcement learning, supervised, self-supervised)
* 4 simulation frameworks (NVIDIA Isaac Lab 2.3.1, MuJoCo 3.4.0, Gazebo Sim 10.0, PyBullet 3.2.5)
* Digital twin infrastructure (patient modeling, treatment simulation, clinical integration via FHIR/DICOM)
* Human oversight and QMS (risk-tiered CRF automation, AE detection safety gates, CAPA triggers)
* Privacy framework (PHI/PII management, Safe Harbor/Expert Determination de-identification, RBAC, breach response)
* Federated learning (differential privacy, secure aggregation, multi-site coordination)
From national-mcp-pai-oncology-trials:
* Five MCP servers (trialmcp-authz, trialmcp-fhir, trialmcp-dicom, trialmcp-ledger, trialmcp-provenance) with 23 tools
* Five conformance levels (Core, Clinical Read, Imaging, Federated Site, Robot Procedure)
* Six actors (Trial Coordinator, Data Monitor, Auditor, Sponsor, CRO, Robot Agent)
* Deny-by-default RBAC with token-based session management
* HIPAA Safe Harbor de-identification (all 18 PHI identifiers) with HMAC-SHA256 pseudonymization
* Hash-chained immutable audit trails (SHA-256, genesis hash, canonical JSON)
* Six-step robot agent workflow (Authenticate -> Retrieve Clinical Data -> Access Imaging -> Execute Procedure -> Record Audit -> Record Provenance)
* Pre-procedure safety matrix, runtime monitoring, post-procedure requirements
* Task-order lifecycle (scheduled -> safety_check -> in_progress -> completed/cancelled/failed)
* Robot capability profiles with minimum USL scores per procedure type (surgical 7.0, therapeutic positioning 6.0, diagnostic needle placement 6.0, rehabilitative exoskeleton 4.0, companion monitoring 3.0)
* Consent status tracking schema
7. Output
Double check LaTeX final document appearance to avoid lines with one word, and lines separated from paragraphs into new pages. Make sure text doesn’t run off the right side of the page (this needs double checked).

When finished, output a new Physical_AI_21_CFR_Part_50.tex, Physical_AI_21_CFR_Part_50.sty, Physical_AI_21_CFR_Part_50.bib, and README, along with the zip of these files, and the properly compiled pdf as Physical_AI_21_CFR_Part_50.pdf. The files should be placed in kevinkawchak/physical-ai-oncology-trials/tree/main/regulatory/Adaption-21-CFR-Part-50 under appropriate directories. The files should be complete, self-contained, and ready for regulatory review.
