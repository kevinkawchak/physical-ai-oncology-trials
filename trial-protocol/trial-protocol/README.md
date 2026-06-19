# README — NIH-FDA Phase 2/3 IND/IDE Clinical Trial Protocol Template (v1.0, 7 Apr 2017)

## Purpose of this package

This package contains the complete body of the **NIH-FDA Phase 2 and 3 IND/IDE Clinical Trial Protocol Template (Version 1.0, 7 April 2017)**, split verbatim into **10 sequential Markdown chunks**. The source was a single Word document (`Protocol-Template-Version-1_0-040717.docx`). The text in the chunks is reproduced **word-for-word**, with no abbreviation, no summarization, and **no added section headings** beyond what already existed in the source. The only structural change versus the original `.docx` is the partition into 10 files at major-section boundaries; every original heading, instructional/explanatory note (in italics), example text (in `[regular font]`/`<angle brackets>`), table, and footnote is preserved in place.

This README is written for **Claude Code (Opus 4.8, 1M context, Max)** to use as an index and reasoning aid when processing the chunked files to draft a **new physical AI oncology trial submission**. It explains (1) what each chunk contains, (2) how each chunk correlates to the others, and (3) how the whole set hangs together as one regulatory instrument.

### Source provenance and important conventions

- **Origin:** This is a U.S. government / NIH-FDA template intended for Phase 2 and 3 trials conducted under an FDA Investigational New Drug (IND, 21 CFR Part 312) or Investigational Device Exemption (IDE, 21 CFR Part 812) application, aligned to ICH E6 Good Clinical Practice.
- **It is a TEMPLATE, not a completed protocol.** The text contains three kinds of content that must be handled differently when authoring a real submission:
  1. **Instruction/explanatory text** — appears in *italics* (and in footnotes). It tells the author what to write and is meant to be **deleted** in the final protocol.
  2. **Example text** — appears in `[regular font]` / square brackets. It is illustrative and should be tailored or deleted.
  3. **Fill-in placeholders** — appear in `<angle brackets>` (e.g., `<Protocol Title>`, `<x.x>`, `<Insert text>`, `<DD Month YYYY>`). These mark where study-specific content must be inserted.
- **Numbering:** The template's native section numbering (Sections 1–11 with sub-numbering like 8.3.6) is the authoritative cross-reference scheme. The chunk filenames (01–10) are an overlay for navigation and **do not** always map 1:1 to the template's Section numbers (see the note on chunk 01/02 below).
- **Preface/Resources/Table of Contents:** The source's removable Preface, Resources list, and auto-generated Table of Contents precede the protocol body. The 10 chunks begin at the **Statement of Compliance**, which is the first true protocol-body element.

---

## The 10 chunks at a glance

| # | File | Template sections covered | Approx. words |
|---|------|---------------------------|---------------|
| 01 | `01_statement_of_compliance_protocol_summary_introduction.md` | Statement of Compliance; §1 Protocol Summary (Synopsis, Schema, Schedule of Activities); §2 Introduction (Study Rationale, Background, Risk/Benefit Assessment) | ~2,680 |
| 02 | `02_objectives_and_endpoints.md` | §3 Objectives and Endpoints | ~1,150 |
| 03 | `03_study_design_and_study_population.md` | §4 Study Design; §5 Study Population | ~1,800 |
| 04 | `04_study_intervention.md` | §6 Study Intervention | ~1,920 |
| 05 | `05_intervention_and_participant_discontinuation.md` | §7 Study Intervention Discontinuation and Participant Discontinuation/Withdrawal | ~1,030 |
| 06 | `06_study_assessments_and_procedures.md` | §8 Study Assessments and Procedures (Efficacy; Safety; Adverse Events/SAEs; Unanticipated Problems) | ~6,560 |
| 07 | `07_statistical_considerations.md` | §9 Statistical Considerations | ~1,940 |
| 08 | `08_supporting_documentation_regulatory_oversight.md` | §10.1 Regulatory, Ethical, and Study Oversight Considerations | ~5,530 |
| 09 | `09_additional_considerations_abbreviations_amendment_history.md` | §10.2 Additional Considerations; §10.3 Abbreviations; §10.4 Protocol Amendment History | ~630 |
| 10 | `10_references_and_footnotes.md` | §11 References; trailing document footnotes [^2]–[^8] | ~870 |

> **Boundary note (chunks 01 ↔ 02):** The template's §2 INTRODUCTION (Study Rationale, Background, Risk/Benefit) physically resides at the **end of chunk 01**, not in chunk 02. Chunk 02 begins precisely at §3 OBJECTIVES AND ENDPOINTS. The filenames reflect this. When Claude Code needs the rationale/background, look in **chunk 01**; when it needs objectives/endpoints, look in **chunk 02**.

---

## Detailed description of each chunk

### Chunk 01 — Statement of Compliance, Protocol Summary, Introduction
**File:** `01_statement_of_compliance_protocol_summary_introduction.md`
**Template content:** Statement of Compliance (two alternative compliance statements, "select one"); §1 PROTOCOL SUMMARY containing §1.1 Synopsis, §1.2 Schema (a visual study-flow diagram described in text/table form), and §1.3 Schedule of Activities (SoA); §2 INTRODUCTION containing §2.1 Study Rationale, §2.2 Background, §2.3 Risk/Benefit Assessment (with §2.3.1 Known Potential Risks, §2.3.2 Known Potential Benefits, §2.3.3 Assessment of Potential Risks and Benefits).
**Role:** This is the high-level "front matter" of the protocol. The Synopsis is a condensed mirror of the entire protocol; the Schema and SoA are graphical/tabular summaries of design and visit structure; the Introduction supplies the scientific justification and the risk-benefit framing that everything downstream must remain consistent with.
**For the oncology AI submission:** This chunk is where the GCP/CFR compliance attestation, the one-page study synopsis, the schematic of the trial arms/timeline, the per-visit assessment grid, and the scientific rationale (including why a physical-AI device/intervention is justified in the oncology indication) will be authored.

### Chunk 02 — Objectives and Endpoints
**File:** `02_objectives_and_endpoints.md`
**Template content:** §3 OBJECTIVES AND ENDPOINTS — typically presented as a three-column table aligning Primary/Secondary/Exploratory/Tertiary **Objectives** with their corresponding **Endpoints** and a justification column.
**Role:** Defines what the trial is trying to establish and exactly how each objective is measured. This is the conceptual hinge of the protocol: objectives flow up from the Introduction (chunk 01) and flow down into Assessments (chunk 06) and Statistics (chunk 07).
**For the oncology AI submission:** Where primary efficacy (e.g., tumor response, PFS/OS, or device-performance metric) and safety objectives are codified, each tied to a measurable endpoint.

### Chunk 03 — Study Design and Study Population
**File:** `03_study_design_and_study_population.md`
**Template content:** §4 STUDY DESIGN (§4.1 Overall Design, §4.2 Scientific Rationale for Study Design, §4.3 Justification for Dose, §4.4 End of Study Definition); §5 STUDY POPULATION (§5.1 Inclusion Criteria, §5.2 Exclusion Criteria, §5.3 Lifestyle Considerations, §5.4 Screen Failures, §5.5 Strategies for Recruitment and Retention).
**Role:** Specifies the trial architecture (phase, randomization/blinding intent, arms, duration) and defines exactly who may enroll. Design choices here must satisfy the objectives in chunk 02 and are operationalized statistically in chunk 07.
**For the oncology AI submission:** Eligibility for the oncology population, dose/exposure justification for the device or agent, and definition of study completion.

### Chunk 04 — Study Intervention
**File:** `04_study_intervention.md`
**Template content:** §6 STUDY INTERVENTION (§6.1 Administration with §6.1.1 Description and §6.1.2 Dosing and Administration; §6.2 Preparation/Handling/Storage/Accountability with §6.2.1–§6.2.4; §6.3 Measures to Minimize Bias: Randomization and Blinding; §6.4 Study Intervention Compliance; §6.5 Concomitant Therapy with §6.5.1 Rescue Medicine).
**Role:** Full operational description of the intervention itself — what it is, how it is given, how it is sourced/handled, how bias is controlled, how adherence is tracked, and what other therapies are allowed or prohibited.
**For the oncology AI submission:** The physical-AI device/intervention's technical description, operating/dosing parameters, handling and accountability, and the randomization/blinding mechanics that chunk 07 will analyze.

### Chunk 05 — Intervention and Participant Discontinuation/Withdrawal
**File:** `05_intervention_and_participant_discontinuation.md`
**Template content:** §7 (§7.1 Discontinuation of Study Intervention, §7.2 Participant Discontinuation/Withdrawal from the Study, §7.3 Lost to Follow-Up).
**Role:** Defines the rules and procedures for stopping the intervention for a participant, for a participant leaving the study, and for handling participants lost to follow-up. These rules interact directly with the analysis populations and missing-data handling in chunk 07 and with safety reporting in chunk 06.
**For the oncology AI submission:** Stopping rules and withdrawal handling appropriate to an oncology device/agent trial.

### Chunk 06 — Study Assessments and Procedures
**File:** `06_study_assessments_and_procedures.md` *(largest chunk)*
**Template content:** §8 (§8.1 Efficacy Assessments; §8.2 Safety and Other Assessments; §8.3 Adverse Events and Serious Adverse Events — including §8.3.1 AE definition, §8.3.2 SAE definition, §8.3.3 Classification [Severity, Relationship to Study Intervention, Expectedness], §8.3.4 Time Period/Frequency for Assessment and Follow-Up, §8.3.5 AE Reporting, §8.3.6 SAE Reporting, §8.3.7 Reporting Events to Participants, §8.3.8 Events of Special Interest, §8.3.9 Reporting of Pregnancy; §8.4 Unanticipated Problems — §8.4.1 Definition, §8.4.2 Reporting, §8.4.3 Reporting to Participants).
**Role:** The operational heart of data capture and safety surveillance. It defines every measurement performed (which realizes the endpoints from chunk 02 on the schedule from chunk 01's SoA) and the entire pharmacovigilance/safety-reporting machinery.
**For the oncology AI submission:** Tumor-response and safety assessment methods, the full AE/SAE grading and causality framework (oncology trials commonly use CTCAE), and unanticipated-problem and IND/IDE safety reporting workflows.

### Chunk 07 — Statistical Considerations
**File:** `07_statistical_considerations.md`
**Template content:** §9 (§9.1 Statistical Hypotheses; §9.2 Sample Size Determination; §9.3 Populations for Analyses; §9.4 Statistical Analyses — §9.4.1 General Approach, §9.4.2 Primary Efficacy Endpoint(s), §9.4.3 Secondary Endpoint(s), §9.4.4 Safety Analyses, §9.4.5 Baseline Descriptive Statistics, §9.4.6 Planned Interim Analyses, §9.4.7 Sub-Group Analyses, §9.4.8 Tabulation of Individual Participant Data, §9.4.9 Exploratory Analyses).
**Role:** Converts the objectives/endpoints (chunk 02) and design (chunk 03) into formal hypotheses, a powered sample size, defined analysis populations (which depend on discontinuation rules in chunk 05), and the analysis plan for the data captured in chunk 06.
**For the oncology AI submission:** Hypotheses, powering, analysis populations (ITT/per-protocol/safety), interim-analysis/stopping logic, and the primary/secondary/safety analysis methods.

### Chunk 08 — Supporting Documentation: Regulatory, Ethical, and Oversight
**File:** `08_supporting_documentation_regulatory_oversight.md` *(second-largest chunk)*
**Template content:** §10.1 (§10.1.1 Informed Consent Process incl. consent/assent documents and consent procedures/documentation; §10.1.2 Study Discontinuation and Closure; §10.1.3 Confidentiality and Privacy; §10.1.4 Future Use of Stored Specimens and Data; §10.1.5 Key Roles and Study Governance; §10.1.6 Safety Oversight; §10.1.7 Clinical Monitoring; §10.1.8 Quality Assurance and Quality Control; §10.1.9 Data Handling and Record Keeping incl. data management responsibilities and records retention; §10.1.10 Protocol Deviations; §10.1.11 Publication and Data Sharing Policy; §10.1.12 Conflict of Interest Policy).
**Role:** The governance, ethics, and operational-compliance backbone. It makes the Statement of Compliance (chunk 01) concrete and defines safety oversight bodies (DSMB/SMC/ISM/safety-assessment committee — defined in the footnotes in chunk 10) that act on the safety data from chunk 06.
**For the oncology AI submission:** Consent, privacy/HIPAA, specimen/data future use, governance roles, DSMB and monitoring plans, QA/QC, data handling and retention, deviation handling, and publication/data-sharing and COI policies.

### Chunk 09 — Additional Considerations, Abbreviations, Amendment History
**File:** `09_additional_considerations_abbreviations_amendment_history.md`
**Template content:** §10.2 Additional Considerations; §10.3 Abbreviations (a glossary table); §10.4 Protocol Amendment History (a version-tracking table).
**Role:** Supplementary catch-all, the master abbreviation glossary used throughout all chunks, and the formal change-control log. The amendment history operationalizes the version-control discipline described in the source's Preface.
**For the oncology AI submission:** Where any not-otherwise-covered considerations are placed, where every acronym used across chunks 01–08 is defined, and where each protocol version/amendment is recorded.

### Chunk 10 — References and Footnotes
**File:** `10_references_and_footnotes.md`
**Template content:** §11 REFERENCES (citation-format guidance and examples), plus the document's trailing **footnotes [^2]–[^8]**. These footnotes carry substantive definitions referenced from earlier chunks — notably the regulatory definition of Phase 2/3 ([^3], from 21 CFR 312.21), and the definitions of the **Safety Monitoring Committee** [^4], **Data and Safety Monitoring Board (DSMB)** [^5], **safety assessment committee** [^6], **Independent Safety Monitor (ISM)** [^7], and **Quality Assurance** [^8].
**Role:** Bibliography plus the authoritative definitions that anchor oversight terms used in chunk 08 and design terms used in chunk 03.
**For the oncology AI submission:** Reference list and the canonical definitions to keep oversight/QA language precise and regulation-aligned.

---

## How the chunks correlate (cross-file dependency map)

The protocol is a single tightly cross-referenced instrument. The most important correlations Claude Code should respect when editing any one chunk:

- **Synopsis (01) ⟷ everything.** The §1.1 Synopsis in chunk 01 is a compressed restatement of objectives (02), design/population (03), intervention (04), endpoints/assessments (06), and statistics (07). Any change to those chunks must be reflected back into the Synopsis, Schema, and SoA in chunk 01.
- **Introduction/Risk-Benefit (01) ⟷ Objectives (02) ⟷ Statistics (07).** Rationale and risk-benefit justify the objectives; objectives become hypotheses and powering. These three must tell one consistent story.
- **Objectives & Endpoints (02) ⟷ Assessments (06) ⟷ Statistics (07).** Every endpoint in chunk 02 must have (a) a defined measurement procedure in chunk 06 and (b) a defined analysis in chunk 07. This is the single most important triangle to keep synchronized.
- **Schedule of Activities / SoA (01) ⟷ Assessments (06) ⟷ Intervention (04).** The SoA grid in chunk 01 must list exactly the visits, intervention administrations (04), and assessments (06) defined elsewhere, at the correct timepoints.
- **Study Design & Population (03) ⟷ Intervention (04) ⟷ Statistics (07).** Randomization/blinding is introduced in design (03), specified operationally in §6.3 (04), and analyzed per analysis populations in §9.3 (07).
- **Discontinuation/Withdrawal (05) ⟷ Analysis Populations & Missing Data (07) ⟷ Safety Reporting (06).** Withdrawal and lost-to-follow-up rules in chunk 05 determine ITT/per-protocol/safety population definitions and missing-data handling in chunk 07; discontinuations triggered by safety events link to chunk 06's AE/SAE machinery.
- **Safety Assessments/AE-SAE (06) ⟷ Safety Oversight (08) ⟷ Oversight definitions (10).** AE/SAE/UP detection and reporting (06) feed the DSMB/SMC/ISM/safety-assessment-committee oversight functions (08), whose formal definitions live in the footnotes in chunk 10. Interim-analysis stopping logic in chunk 07 also feeds this oversight loop.
- **Statement of Compliance (01) ⟷ Regulatory/Ethical/Oversight (08) ⟷ References/Footnotes (10).** The compliance attestation (01) is made operational by consent, governance, monitoring, QA/QC, data-handling, deviation, publication, and COI policies (08), grounded in the CFR/ICH definitions cited in chunk 10.
- **Abbreviations (09) ⟷ all chunks.** Every acronym introduced in chunks 01–08 should resolve to an entry in the §10.3 glossary in chunk 09; the Amendment History (09) governs version control across the whole set.

### Suggested processing order for Claude Code
For drafting the physical-AI oncology submission, a productive order is: **02 (objectives) → 03 (design/population) → 04 (intervention) → 06 (assessments) → 07 (statistics) → 05 (discontinuation) → 08 (regulatory/oversight) → 01 (synopsis/schema/SoA/introduction, authored last so it accurately summarizes the rest) → 09 (abbreviations/amendments) → 10 (references).** Author the body first, then reconcile the Synopsis/Schema/SoA in chunk 01 against the finished body, then finalize the glossary and references.

### Editing/authoring reminders (carried from the template's own conventions)
1. **Delete italic instruction text and footnoted instructions** before finalizing; they are guidance, not protocol content.
2. **Resolve every `<angle-bracket>` placeholder** (title, version, dates, names, numbers) with study-specific values.
3. **Tailor or delete `[bracketed]` example text**; do not ship it verbatim unless it genuinely fits.
4. **Keep section order and numbering intact** — the template requires all sections present and in order; mark non-applicable sections "not applicable" rather than removing them.
5. **Maintain version control** via the footer version/date and the §10.4 Protocol Amendment History (chunk 09) on every revision.
6. **Preserve cross-references** — when you change one chunk, walk the correlation map above and update every dependent chunk.

---

## Integrity statement

The 10 chunks together reproduce the entire protocol body (Statement of Compliance through References and trailing footnotes), **word-for-word**, partitioned only at major-section boundaries. Total body word count ≈ **24,105 words**, conserved exactly across the 10 files (no text added, removed, abbreviated, or re-headed). Document Preface, Resources, and the auto-generated Table of Contents that preceded the body in the original `.docx` are intentionally not included, as they are removable front matter per the template's own instructions.


### BibTeX Entries (match to relevant chunk file) 

% 1. NIH Protocol Template URL
@online{nih_protocol_template,
  author       = {{National Institutes of Health}},
  title        = {NIH Clinical Trial Protocol Template},
  url          = {https://grants.nih.gov/policy-and-compliance/policy-topics/clinical-trials/protocol-template},
  year         = {n.d.}
}

% 2. Journal Citation
@article{veronesi2007tamoxifen,
  author       = {Veronesi, Umberto and Maisonneuve, Patrick and Decensi, Andrea},
  title        = {Tamoxifen: an enduring star},
  journal      = {JNCI: Journal of the National Cancer Institute},
  volume       = {99},
  number       = {4},
  pages        = {258--260},
  year         = {2007},
  month        = {Feb},
  doi          = {10.1093/jnci/djk072},
  url          = {https://doi.org/10.1093/jnci/djk072}
}

% 3. Whole Book Citation
@book{belitz2004food,
  author       = {Belitz, Hans-Dieter and Grosch, Werner and Schieberle, Peter},
  title        = {Food Chemistry},
  edition      = {3rd revised},
  translator   = {Burghagen, M. M.},
  publisher    = {Springer},
  address      = {Berlin},
  year         = {2004},
  pages        = {1070},
  doi          = {10.1007/978-3-662-07279-0},
  url          = {https://doi.org/10.1007/978-3-662-07279-0}
}

% 4. Chapter in a Book Citation
@inbook{riffenburgh2006statistics,
  author       = {Riffenburgh, Robert H.},
  title        = {Regression and correlation methods},
  booktitle    = {Statistics in Medicine},
  edition      = {2nd},
  chapter      = {24},
  pages        = {447--486},
  publisher    = {Elsevier Academic Press},
  address      = {Amsterdam (Netherlands)},
  year         = {2006},
  doi          = {10.1016/B978-0-12-384864-2.00025-1},
  url          = {https://doi.org/10.1016/B978-0-12-384864-2.00025-1}
}

% 5. Web Site Citation
@online{mdanderson2007cimer,
  author       = {{University of Texas, M.D. Anderson Cancer Center}},
  title        = {Complementary/Integrative Medicine [Internet]},
  address      = {Houston},
  year         = {2007},
  note         = {[cited 2007 Feb 21]},
  url          = {http://www.manderson.org/departments/CIMER/}
}



