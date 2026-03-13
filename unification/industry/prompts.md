# Industry Specification Prompts

Development prompts for the Physical AI Oncology Trial Industry Specification (PAIOTIS).

---

## v2.3.0 -- PAIOTIS v1.0 Industry Specification (March 13, 2026)

Your goal is to process the prompt below using the four kevinkawchak repositories, and provide the new files to physical-ai-oncology-trials/tree/main/unification/industry. You are responsible for proficiently understanding and applying all aspects of each repository. This includes: all code, all code types, machine learning and AI types, different robot types and characteristics, digital twins, examples, patients, physical ai unification, unification standard level (USL), tools, etc. No em dashes are allowed anywhere in the work. Be careful to not specify too many version numbers across the repositories to limit reader confusion.

Use the LaTeX template from below, but include no images, logos, figures, charts or any other template identifications (keep black and dark blue text). The margins for each of the page sides need to be for a typical academic article: 1.0” top/bottom and 0.75” left/right. Keep the paragraph spacings, font types, and font sizes. Adapt the template to the prompt specifications (all headings, chapters, titles, etc. need adapted to the prompt meant to look more like an academic paper for industry). Remove the following pages: Acknowledge, Abstract, List of Tables, List of Figures, Appendix A, Annex, Annex A. Essentially, make sure the template adapts to the industrial theme of the prompt. Do make sure to attribute the author for their template.
https://www.overleaf.com/latex/templates/thesis-template-masters-degree-in-engineering-utb/crcspmcdpdvj

“START COVER PAGE”
No logos are allowed throughout the work. Replace original cover page text with each of the following segments in order using the same text type and text size:
“Physical AI Oncology Trial Industry Specification and Pharmaceutical Implementation Standard - v1.0” (Top)
“13 March 2026” (1 Line)
“10.5281/zenodo.18994579” with hyperlink https://doi.org/10.5281/zenodo.18994579 (1 Line) 
“CEO Kevin Kawchak” (1 Line)
“ChemicalQDevice” (1 Line) 
“San Diego, California” (1 Line) 
“This work was generated using Claude Code Opus 4.6.” (Bottom)
“END COVER PAGE”

The new work must be consistently high quality throughout the entire work. Important: Use your Full 1M context length. Process through the first half of the .tex file. Then, Confirm when you are finished with the first half. I will then type Continue, and you will then finish the second half.

Avoid large white empty spaces without text. Where large spacing between words exist throughout: modify raggedright spacing to make positioning between words look equally and properly spaced. Make sure text doesn’t run off the right side of the page anywhere. Perform the final formatting steps that a senior author would take by correcting white space formatting, removing and/or adding relevant text to make each section and page look properly formatted and self standing by itself. (Don’t overcrowd the page with text, some white space formatting is ok).

Update physical-ai-oncology-trials main readme documentation, repository structures, text diagrams and toc (with correct urls and make sure it is in the right order), and other affected areas in the repository. Make sure the repository is fully up to date regarding badges, content, and context. When finished, output the new .tex, .sty, .bib, and README, along with the zip of these files, and the properly compiled pdf into physical-ai-oncology-trials/tree/main/unification/industry.

Provide a copy of this v2.3.0 prompt under a new unification/industry/prompts.md. Be sure to fix and address errors that would cause failed checks for the single pull request (such as for lint and Python environment issues to avoid the following error during final checks): "3 failing checks
x Cl / lint-and-format (3.10) (pull...
x Cl / lint-and-format (3.11) (pull...
x Cl / lint-and-format (3.12) (pull... " Place the new release notes in releases.md under main using the format below. Update other relevant documentation such as project structures. Update the main Readme diagrams, repository structure, etc. where necessary. Provide an updated changelog (v2.3.0).

When you are finished, auto-push the update to GitHub on your own for my review. The user will then review your updates in GitHub prior to finalization.

"FORMAT"
Release title
v2.3.0 - [Fill in Title Here]

## Summary

## Features

## Contributors
@kevinkawchak
@claude

## Notes



“START PROMPT”
You are tasked with generating a complete Physical AI Oncology Trial Industry Specification and Pharmaceutical Implementation Standard — a single, authoritative reference document that formally establishes the Physical AI oncology trial industry by unifying all technical, regulatory, operational, and commercial infrastructure developed across four foundational repositories into one executable industry standard that pharmaceutical and biotechnology companies can adopt immediately.
Background — The Foundation That Has Been Built:
Four repositories represent years-equivalent of compressed, intensive development work that has already solved the hardest unsolved problems in this space:
1. physical-ai-oncology-trials (v2.2.0) — The core framework containing 51+ validated Python modules, 69+ documentation files, and 28+ engineering examples spanning: (a) the Unification Standard Level (USL) scoring framework evaluating 9 robots across 3 categories (surgical, cobot, humanoid) on a rigorous 4-dimension, 10-point scale; (b) bidirectional simulation bridges between NVIDIA Isaac Lab/Sim and MuJoCo with full URDF/SDF/MJCF/USD format conversion; (c) a complete digital twin pipeline for patient-specific tumor modeling using reaction-diffusion, Gompertz, and mechanistic models with real-time intraoperative synchronization via Extended Kalman Filters; (d) a federated learning platform implementing FedAvg, FedProx, and SCAFFOLD with differential privacy budget accounting and secure aggregation via additive secret sharing; (e) a comprehensive privacy framework with automated PHI detection for all 18 HIPAA Safe Harbor identifiers, de-identification pipelines, RBAC access control with 21 CFR Part 11 audit trails, and breach response automation; (f) a full ICH E6(R3) adaptation guidance document with 4 sections, 3 appendices, and 30 Physical AI-specific definitions — adapted from the ICH E6(R3) Good Clinical Practice guideline effective in the US as of September 2025 — now updated to include advanced AI and robotics requirements that replace prior oncology trial guidance assumptions; and (g) a 10-page patient instructions publication covering 10 robot types across 5 clinical categories with quantitative procedural parameters. All published with DOIs on Zenodo.
2. mcp-pai-oncology-trials (TrialMCP) — The reference implementation of Model Context Protocol servers for Physical AI oncology trials containing 5 specialized MCP servers (authorization with deny-by-default RBAC, FHIR R4 clinical data with HIPAA Safe Harbor de-identification, DICOM medical imaging with C-FIND/C-MOVE proxy, append-only hash-chained audit ledger for 21 CFR Part 11, and W3C PROV-based data lineage with SHA-256 integrity verification), 39 passing security and integration tests covering SSRF prevention and injection resistance, synthetic clinical datasets, and 3 peer-reviewed papers. This repository proved that MCP can serve as the interoperability protocol connecting autonomous oncology robots to regulated clinical infrastructure.
3. national-mcp-pai-oncology-trials (v1.2.0) — The proposed National MCP Standard for Physical AI Oncology Clinical Trials targeting U.S. clinical sites, trial sponsors, CROs, and technology vendors. Contains a 20-page academic paper, 13 JSON Schema files (Draft 2020-12), 34 integration adapters for hospital EHRs/PACS/identity platforms, 331 automated conformance tests plus 337 unit tests, 8 safety execution boundary and e-stop implementations, and Docker Compose/Kubernetes deployment configurations. This repository established machine-readable governance — every protocol specification expressed as enforceable JSON schemas and conformance profiles.
4. pai-oncology-trial-fl — The federated learning specialization repository demonstrating privacy-preserving multi-site oncology trial coordination, proving that cross-institutional AI model training can proceed without sharing raw patient data across the Physical AI trial network.
What You Must Generate — The Industry Specification:
Create a comprehensive Physical AI Oncology Trial Industry Specification (PAIOTIS) v1.0 structured as a formal industry standard document (comparable in rigor and structure to an ICH, IEEE, or ASTM standard) that accomplishes the following:
Part I — Industry Definition and Scope
* Formally define the Physical AI Oncology Trial Industry with precise technical and commercial boundaries, drawing directly from the USL framework's 4-dimension scoring and the 7 robot categories already validated
* Establish the canonical taxonomy of Physical AI oncology trial participants: robot manufacturers, simulation platform providers, pharmaceutical sponsors, CROs, clinical sites, AI/ML vendors, MCP infrastructure providers, and regulatory bodies
* Define the minimum viable ecosystem requirements — what must exist for a Physical AI oncology trial to be considered operational (reference the specific modules, servers, and frameworks already built)
Part II — Technical Architecture Standard
* Codify the three-layer architecture (Physical AI Layer → MCP Protocol Layer → Clinical Trial Layer) from TrialMCP as the industry reference architecture
* Specify mandatory MCP server conformance requirements using the 13 JSON schemas from the National MCP Standard, the 5 server types from TrialMCP, and the 331 conformance tests as the baseline
* Define simulation interoperability requirements based on the Isaac-MuJoCo bidirectional bridge, establishing that any conforming Physical AI system must demonstrate cross-framework policy transfer
* Standardize digital twin integration protocols using the tumor twin pipeline, treatment simulator, and clinical DT interface patterns already implemented
* Specify federated learning minimum requirements: which aggregation strategies (FedAvg/FedProx/SCAFFOLD), privacy guarantees (epsilon-delta differential privacy bounds), and secure aggregation protocols are mandatory vs. recommended
Part III — Regulatory Compliance Framework
* Integrate the ICH E6(R3) adaptation guidance (all 4 sections, 3 appendices, 30 definitions) as the GCP foundation, explicitly noting this represents the updated guidance that now includes advanced AI and robotics
* Map FDA submission pathways (510(k), De Novo, PMA, Breakthrough) to specific Physical AI system categories with decision trees derived from the regulatory tracker
* Define IRB review requirements specific to Physical AI trials, incorporating the robot-specific informed consent procedures documented in the patient instructions
* Establish the Predetermined Change Control Plan (PCCP) requirements for post-market AI model updates in deployed Physical AI trial systems
* Specify multi-jurisdiction regulatory intelligence requirements (FDA, EMA, PMDA, TGA, Health Canada) based on the regulatory intelligence module
Part IV — Privacy and Data Governance Standard
* Mandate the privacy framework stack: PHI detection (18 identifiers), de-identification (Safe Harbor + Expert Determination), RBAC (21 CFR Part 11), breach response, and DUA templates — all already implemented and validated
* Define cross-site data harmonization requirements: DICOM normalization, ICD-10 to SNOMED-CT mapping, FHIR R4 standardization as implemented in the federation data harmonization module
* Specify consortium reporting standards: DSMB package formats, enrollment dashboards, CTCAE adverse event grading, and privacy-preserving analytics (federated Kaplan-Meier, Cox proportional hazards)
Part V — Robot Qualification and Certification
* Establish the USL scoring framework as the industry standard for robot qualification, with the complete methodology (4 dimensions × 25% weight each, 10-point scale, 5 score bands)
* Publish the baseline USL scores for 9 evaluated robots as reference benchmarks (Franka Panda 7.4, da Vinci dVRK 7.1, Atlas Electric 5.8, Kinova Gen3 5.7, Hugo RAS 4.5, Digit 4.2, Optimus Gen 2 3.6, Versius 3.4, xArm 7 3.4)
* Define qualification tiers: which USL score ranges qualify robots for which trial phases (Phase I safety, Phase II efficacy, Phase III pivotal, Phase IV post-market)
* Specify ongoing re-qualification requirements as robot firmware, AI models, and simulation environments evolve
Part VI — Pharmaceutical Sponsor Implementation Guide
* Provide step-by-step adoption pathways for pharmaceutical companies with existing oncology trial programs to integrate Physical AI, organized by company capability maturity
* Define the commercial value proposition with specificity: how Physical AI trials improve enrollment efficiency, reduce per-patient costs, accelerate timelines, and generate higher-quality endpoint data — grounded in the technical capabilities already demonstrated (digital twins for virtual cohort design, federated learning for multi-site coordination, agentic AI for real-time protocol adaptation)
* Map Physical AI integration to existing pharmaceutical development stages: IND-enabling, first-in-human, dose-finding, registration, and post-market
* Address the CRO partnership model: what capabilities CROs must develop or acquire to support Physical AI trials, using the National MCP Standard conformance requirements as the benchmark
Part VII — Clinical Site Readiness Criteria
* Define infrastructure requirements for clinical sites participating in Physical AI trials: computational, networking, physical space, staffing, and training
* Specify the patient education and informed consent requirements using the validated 10-page patient instruction format (10 robot types, 5 clinical categories, quantitative procedural parameters)
* Establish safety monitoring requirements including the 8 execution boundary and e-stop implementations from the National MCP Standard
* Define multi-site federation onboarding procedures based on the 6 progressive federation examples (from basic 2-site to full 8-site multi-cancer consortium)
Part VIII — Industry Milestone Roadmap
* Define measurable milestones that mark the progression of the Physical AI oncology trial industry from establishment through maturity
* Phase 1 (2026): Industry specification adoption, first conformance certifications, initial pharmaceutical sponsor commitments — enabled by the complete infrastructure already built across all 4 repositories
* Phase 2 (2027): First Physical AI IND submissions, multi-site federation deployments, USL score improvements for proprietary systems
* Phase 3 (2028+): Pivotal Physical AI trials, regulatory pathway precedents, standard-of-care integration
* For each phase, reference the specific technical components from the repositories that enable that milestone
Critical Requirements for This Document:
1. Let the work speak for itself. Every claim must trace directly to implemented code, validated frameworks, published papers, or passing test suites from the four repositories. The pharmaceutical industry will recognize this as legitimate precisely because the technical foundation is already built — not proposed, not theoretical, but implemented with 51+ modules, 668+ tests, 34+ integration adapters, 9 evaluated robots, and published regulatory guidance with DOIs.
2. Write at the level pharmaceutical regulatory affairs VPs, Chief Medical Officers, and Heads of Clinical Operations read. These are the people who will recognize that this specification means the Physical AI oncology trial industry has started. They understand ICH guidelines, FDA submission pathways, GCP requirements, and clinical trial operations. Speak their language while demonstrating the technical depth that makes this credible.
3. Make the difficulty visible. The reason this industry specification is significant is because what it codifies was genuinely hard to build: adapting ICH E6(R3) for robotics required deep regulatory expertise; building bidirectional simulation bridges required solving physics parameter mapping across fundamentally different engines; implementing federated learning with differential privacy for oncology required solving the intersection of ML, cryptography, and clinical trial statistics; creating a national MCP standard required designing machine-readable governance for autonomous medical robots. The specification should make clear — through its technical precision, not through self-congratulation — that this body of work represents the necessary foundation that did not previously exist.
4. This is a formal industry standard, not a white paper. Use normative language (SHALL, SHOULD, MAY per RFC 2119). Include conformance levels. Define certification requirements. This document should be something that a pharmaceutical company's legal and regulatory affairs team can reference in an FDA pre-submission meeting, that a CRO can use to build a Physical AI trial capability, and that a robot manufacturer can use to achieve USL certification.
Generate the complete PAIOTIS v1.0 specification. Include all sections, subsections, normative requirements, informative annexes, and cross-references to the four foundational repositories. This is the document that announces — through the weight and completeness of the work behind it — that the Physical AI oncology trial industry has begun.

This prompt is designed so that the resulting specification would be immediately recognizable to pharmaceutical industry leaders as a legitimate industry inflection point — not because it makes grand claims, but because the technical infrastructure it codifies (across all four repositories) is comprehensive enough, validated enough, and regulatory-aware enough that it could only exist if the foundational work had actually been done. The hard work speaks for itself.

Development by Claude Code Opus 4.6.
