# README — Chunked Source Files for Physical AI Oncology Trial Paper

**Prepared for:** Claude Code Opus 4.7 (1M context)
**Source document:** Deep-Research-3.docx
**Purpose:** This README describes the structure, content, and cross-file relationships of four chunked Markdown files derived from a single deep-research baseline document on U.S. oncology trial law, AI incorporation, and patient control over cancer. These files collectively form the legal, regulatory, clinical-evidence, and bibliographic foundation for developing a new physical AI oncology trial paper.

---

## Document Overview

The source document is titled **"Oncology trial laws, AI incorporation, and patient control over cancer: relevance-ranked baseline."** Its central argument is that U.S. oncology-trial law has not evolved as a single "AI oncology trial statute." Instead, patient control has been built through a layered legal stack — transparency, insurance coverage, Medicaid coverage, health information access, patient-experience evidence, decentralized trial authority, diversity action planning, and software/algorithm transparency — with AI arriving later as an implementation layer. Thirteen ranked entries (eight passed laws, two regulatory/guidance documents, two clinical-AI evidence studies, and one state law) plus one supporting CMS policy document are analyzed. Every entry is evaluated across three consistent analytical axes: (1) relevance and legal/regulatory mechanism, (2) the specific AI type(s) implicated, and (3) patient control metrics.

The document concludes with a forward-looking section comparing what law has achieved at baseline versus what future AI and robotics systems must achieve, with a proposed benchmark framing called "patient time-to-control."

---

## File Inventory

| File | Content Scope | Citation Range | Word Count (approx.) |
|---|---|---|---|
| `chunk_01_intro_ranks1-4.md` | Introduction + Ranks 1–4 (federal laws, core access stack) | [1] [2] [3] [4] [8] [9] [10] [11] [12] [13] [14] | ~1,150 words |
| `chunk_02_ranks5-9.md` | Ranks 5–9 (trial infrastructure, patient pathways, state law, algorithm transparency) | [5] [6] [7] [8] [9] [12] [13] | ~900 words |
| `chunk_03_ranks10-13_future.md` | Ranks 10–13 (CDS guidance, diversity plans, AI evidence) + "Baseline versus future AI and robotics" synthesis section | [1] [2] [3] [4] [5] [9] [10] [11] [12] [13] | ~950 words |
| `chunk_04_bibtex.md` | All 14 BibTeX entries corresponding to citation keys [1]–[14] | All | ~900 words |

---

## File-by-File Detail

### chunk_01_intro_ranks1-4.md

**What it contains:**

This file opens with the document's two introductory paragraphs, which establish the layered legal-stack thesis and define "patient control" in operational rather than biological terms. It then covers the four highest-ranked entries in the baseline:

- **Rank 1 — 21st Century Cures Act (PL 114-255):** The foundational law for AI oncology trials. Establishes interoperability (FHIR-based data exchange), information-blocking rules, novel trial design guidance, real-world evidence evaluation, and the non-device clinical decision support software carve-out. This law is the legal prerequisite for NLP, LLM extraction, oncology knowledge graphs, and FHIR-based trial-matching systems. BibTeX key: `rank1_cures_act_2016` → citation [1].

- **Rank 2 — Consolidated Appropriations Act 2021 / CLINICAL TREATMENT Act (PL 116-260):** Mandates Medicaid coverage of routine patient costs for qualifying clinical trials from January 1, 2022. Directly addresses the financial barrier for low-income patients. AI trial-matching tools become practically actionable only when Medicaid coverage makes the match financially usable. BibTeX key: `rank2_clinical_treatment_act_2020` → citation [2]. Supporting implementation document: CMS SMD 21-005 → citation [14].

- **Rank 3 — Patient Protection and Affordable Care Act (PL 111-148):** PHS Act Section 2709 creates the national private-insurance baseline requiring non-grandfathered plans to cover routine costs and not discriminate against trial participants. Enables payer-benefit automation and coverage-feasibility filtering in AI matching tools. BibTeX key: `rank3_aca_clinical_trials_2010` → citation [3].

- **Rank 4 — Consolidated Appropriations Act 2023 / FDORA / FDA Modernization Act 2.0 (PL 117-328):** The most forward-looking law in the stack. Requires diversity action plans, mandates FDA guidance on decentralized clinical studies, and authorizes in silico and computer-model alternatives to animal testing (Section 3209). Enables ML toxicity prediction, digital twins, synthetic controls, and decentralized trial analytics. BibTeX key: `rank4_fdora_fda_modernization_2022` → citation [4].

**Key analytical themes introduced in this file:**
- The layered legal-stack model (laws build on each other, AI is the implementation layer on top)
- The definition of "patient control" as operational/decisional, not biological
- Introduction of the three-axis analytical framework: (1) legal mechanism, (2) AI type, (3) patient control metrics
- The distinction between laws that create rights and AI that determines whether those rights are exercised in time

**Cross-references to other chunks:**
- All four ranked laws in this file are cited again in `chunk_03_ranks10-13_future.md` in the synthesis section on "Baseline versus future AI and robotics," where they anchor the description of what the legal baseline has and has not achieved.
- Citation [14] (CMS SMD 21-005) supports Rank 2 here and is defined in `chunk_04_bibtex.md`.
- The AI types introduced here (NLP, LLM extraction, FHIR, knowledge graphs, in silico modeling) reappear with additional evidence in `chunk_03_ranks10-13_future.md` (Ranks 12 and 13).
- The concept of "patient time-to-control" introduced as a metric concept in this file is formally defined and elaborated in `chunk_03_ranks10-13_future.md`.

---

### chunk_02_ranks5-9.md

**What it contains:**

This file covers five ranked entries that build out the trial infrastructure, alternative access pathways, patient-experience evidence layer, state-level precedent, and algorithm-transparency regulation:

- **Rank 5 — Food and Drug Administration Amendments Act of 2007 (PL 110-85):** FDAAA Section 801 creates the ClinicalTrials.gov legal infrastructure — the trial registry and results reporting corpus that all modern AI oncology matching systems depend on. Without structured trial records, NLP and LLM extraction have nothing to parse. BibTeX key: `rank5_fdaaa_2007` → citation [5].

- **Rank 6 — Right to Try Act of 2017 (PL 115-176):** Creates the legal pathway for terminally ill patients to access investigational drugs outside a trial after phase 1 completion. AI can support this pathway by identifying products, checking phase 1 status, and comparing Right to Try versus expanded access versus trial options. BibTeX key: `rank6_right_to_try_2018` → citation [6].

- **Rank 7 — FDA Reauthorization Act of 2017 (PL 115-52):** Reauthorizes FDA user-fee programs and reinforces patient-focused drug development. Patient-experience data and patient-reported outcomes affect trial design and regulatory review. Downstream AI types include NLP of patient narratives, ML analysis of PROs, and digital symptom monitoring. BibTeX key: `rank7_fdara_2017` → citation [7].

- **Rank 8 — California SB 37 (2001):** The key state-law precedent for cancer clinical trial routine-cost coverage, requiring many California health plans to cover routine care for cancer patients in phase I–IV trials. Predates current AI but established the practical coverage environment that makes AI-surfaced trial matches financially actionable. BibTeX key: `rank8_california_sb37_2001` → citation [8].

- **Rank 9 — ONC HTI-1 Final Rule (2023):** The first regulatory document in the baseline (not a law). Operationalizes 21st Century Cures Act health IT policy and establishes transparency requirements for predictive decision support interventions in certified health IT, including AI and algorithmic tools. Defines "predictive decision support intervention" broadly to include risk stratification, trial-screening recommendations, toxicity prediction, recurrence prediction, and referral prioritization. BibTeX key: `rank9_onc_hti1_2023` → citation [9].

**Key analytical themes in this file:**
- Transition from core access laws (Chunk 1) to infrastructure, alternative pathways, and transparency regulation
- The ClinicalTrials.gov registry as the structural data layer that all AI matching systems depend on
- The importance of state-level and alternative-pathway laws for filling gaps in the federal stack
- Introduction of the algorithm-transparency regulatory framework that will be carried forward in Rank 10 (Chunk 3)

**Cross-references to other chunks:**
- Rank 5 (FDAAA / ClinicalTrials.gov) is cited in `chunk_03_ranks10-13_future.md` in both Rank 13 (LLM biomarker matching relies on ClinicalTrials.gov records) and the synthesis section.
- Rank 8 (California SB 37) is also cited in `chunk_01_intro_ranks1-4.md` at Rank 4 (Section 3209 authorizes "computer models," which parallels CA SB 37's practical access role at state level).
- Rank 9 (ONC HTI-1) is one of the most cross-referenced entries in the document; it is also cited at Ranks 11 and 12 in `chunk_03_ranks10-13_future.md` and throughout the "Baseline versus future AI and robotics" synthesis section. The transparency requirements it establishes are the regulatory backstop for all AI systems discussed in Ranks 10–13.
- BibTeX definitions for all five ranked entries are in `chunk_04_bibtex.md`.

---

### chunk_03_ranks10-13_future.md

**What it contains:**

This is the most analytically dense file. It covers the four lowest-ranked entries — two FDA guidance documents and two clinical-evidence studies — and then presents the document's synthesis and forward-looking section:

- **Rank 10 — FDA Clinical Decision Support Software Guidance (2026):** Implements Cures Act Section 3060 and clarifies when CDS is excluded from FDA device regulation versus when it remains a regulated device. Defines the non-device CDS boundary for oncology trial recommendation software. BibTeX key: `rank10_fda_cds_guidance_2026` → citation [10].

- **Rank 11 — FDA Diversity Action Plans Draft Guidance (2024):** Issued under FDORA Sections 3601 and 3602. Specifies form, content, timing, and evaluation of diversity action plans. Directly addresses oncology trials' historical underrepresentation by race, ethnicity, age, geography, and income. AI tools (predictive analytics, site-selection algorithms, epidemiologic modeling) can help set and meet enrollment goals, but biased models risk reproducing inequities. BibTeX key: `rank11_fda_diversity_action_plans_2024` → citation [11].

- **Rank 12 — Neuro-symbolic multi-agent AI oncology trial matching (ESMO, 2026):** The strongest current clinical-evidence document. Prospective evaluation in 3,804 patients. F1 score 0.82, screening time reduced from ~120 minutes to ~30 minutes, processed >157,000 clinical pages, no demographic subgroup F1 gap >10 percentage points. Architecture: LLM-based extraction + ontology-grounded deterministic eligibility reasoning + oncology knowledge graph + expert-curated corpora. BibTeX key: `rank12_neurosymbolic_multiagent_ai_oncology_trial_matching_2026` → citation [12].

- **Rank 13 — LLM biomarker-based oncology trial matching (npj Digital Medicine, 2025):** Evaluates LLM-based extraction and matching for biomarker-driven precision oncology trials. Focuses on extracting eligibility information from unstructured text and matching genomic biomarkers to trial criteria. BibTeX key: `rank13_llm_biomarker_oncology_trial_matching_2025` → citation [13].

- **"Baseline versus future AI and robotics" synthesis section:** This is the document's forward-looking conclusion. It articulates the gap between legal access rights and automated execution, proposes "patient time-to-control" as the correct benchmark metric (time from scan/molecular result to understandable options → time to consent → time to first trial visit → time from progression to next match), and describes the architecture of future systems: LLMs + knowledge graphs + EHR interoperability + genomic interpretation + claims/coverage logic + decentralized trial logistics + remote monitoring + digital twins + robotics for biopsy/imaging/drug delivery/surgery/home sample collection. It closes by insisting future systems be judged on calibrated accuracy, subgroup fairness, explainability, patient comprehension, coverage feasibility, missed-trial reduction, travel burden, retention, adverse event reduction, and decisional agency.

**Key analytical themes in this file:**
- The regulatory boundary between AI-as-CDS and AI-as-medical-device
- Fairness, subgroup performance, and the risk of algorithmic bias reproducing demographic inequities
- Quantified AI performance benchmarks (F1 0.82, 4x screening speedup) as the current state-of-the-art anchor
- The architecture of neuro-symbolic AI as superior to pure LLM or pure rules-based approaches
- The "patient time-to-control" metric framework as the evaluative standard for the new paper
- The full scope of physical AI and robotics integration (robotics for biopsy, imaging, drug delivery, surgery, home sample collection)

**Cross-references to other chunks:**
- This file cites every other major law and guidance document from Chunks 1 and 2: citations [1], [2], [3], [4], [5], [9], [10] all appear in the synthesis section.
- Rank 12 and Rank 13 are the primary clinical-evidence anchors. They translate the legal baseline from Chunks 1 and 2 into measurable patient outcomes.
- The "patient time-to-control" concept is first referenced in `chunk_01_intro_ranks1-4.md` and formally defined here.
- The neuro-symbolic architecture described in Rank 12 is the recommended AI architecture for any new physical AI oncology trial paper built on this baseline.
- BibTeX definitions for all four ranked entries are in `chunk_04_bibtex.md`.

---

### chunk_04_bibtex.md

**What it contains:**

All 14 BibTeX entries for the source document, in order of citation key. Entry types used: `@legislation` (federal and state statutes), `@regulation` (ONC HTI-1 Final Rule), `@misc` (FDA guidance documents and CMS policy letter), and `@article` (peer-reviewed journal articles). Each entry includes: title, bill or author, public law number or journal/volume/number, date, institution/publisher, URL, and a note field containing the relevance rank and key supplementary URLs.

**Citation key to bracket-reference mapping:**

| Citation Key | Bracket Ref | Type | Subject |
|---|---|---|---|
| `rank1_cures_act_2016` | [1] | @legislation | 21st Century Cures Act |
| `rank2_clinical_treatment_act_2020` | [2] | @legislation | CLINICAL TREATMENT Act / Medicaid trial coverage |
| `rank3_aca_clinical_trials_2010` | [3] | @legislation | ACA Section 2709 / private insurance trial coverage |
| `rank4_fdora_fda_modernization_2022` | [4] | @legislation | FDORA / FDA Modernization Act 2.0 / diversity + in silico |
| `rank5_fdaaa_2007` | [5] | @legislation | FDAAA / ClinicalTrials.gov registry |
| `rank6_right_to_try_2018` | [6] | @legislation | Right to Try Act |
| `rank7_fdara_2017` | [7] | @legislation | FDA Reauthorization Act 2017 / patient-focused endpoints |
| `rank8_california_sb37_2001` | [8] | @legislation | California SB 37 / state cancer trial coverage |
| `rank9_onc_hti1_2023` | [9] | @regulation | ONC HTI-1 / algorithm transparency in certified health IT |
| `rank10_fda_cds_guidance_2026` | [10] | @misc | FDA CDS Software Guidance / device vs. non-device boundary |
| `rank11_fda_diversity_action_plans_2024` | [11] | @misc | FDA Diversity Action Plans draft guidance |
| `rank12_neurosymbolic_multiagent_ai_oncology_trial_matching_2026` | [12] | @article | Neuro-symbolic multi-agent AI trial matching (ESMO 2026) |
| `rank13_llm_biomarker_oncology_trial_matching_2025` | [13] | @article | LLM biomarker trial matching (npj Digital Medicine 2025) |
| `rank14_cms_smd_21_005_2021` | [14] | @misc | CMS SMD 21-005 / Medicaid routine-cost coverage implementation |

**Cross-references to other chunks:**
- Every bracket citation `[1]` through `[14]` appearing in Chunks 1, 2, and 3 resolves to a BibTeX entry in this file.
- Note [14] (CMS SMD 21-005) appears in `chunk_01_intro_ranks1-4.md` as the implementing document for Rank 2 but has no standalone ranked section of its own; it is a supporting reference.
- Note [8] (California SB 37) appears in both `chunk_01_intro_ranks1-4.md` (Rank 4 note reference) and `chunk_02_ranks5-9.md` (Rank 8 main entry).
- Note [9] (ONC HTI-1) is the most widely cited non-statutory entry, appearing in Ranks 9, 11, and the synthesis section across `chunk_02_ranks5-9.md` and `chunk_03_ranks10-13_future.md`.

---

## Cross-File Correlation Map

The following table summarizes which source documents (citation keys) are substantively discussed in each chunk:

| Citation / Entry | chunk_01 | chunk_02 | chunk_03 | chunk_04 |
|---|---|---|---|---|
| [1] Cures Act | ✅ Main | — | ✅ Synthesis | ✅ BibTeX |
| [2] CLINICAL TREATMENT Act | ✅ Main | — | ✅ Synthesis | ✅ BibTeX |
| [3] ACA Section 2709 | ✅ Main | — | ✅ Synthesis | ✅ BibTeX |
| [4] FDORA / FDA Mod Act 2.0 | ✅ Main | — | ✅ Synthesis | ✅ BibTeX |
| [5] FDAAA / ClinicalTrials.gov | — | ✅ Main | ✅ Rank 13 + Synthesis | ✅ BibTeX |
| [6] Right to Try | — | ✅ Main | — | ✅ BibTeX |
| [7] FDARA 2017 | — | ✅ Main | — | ✅ BibTeX |
| [8] CA SB 37 | ✅ Ref [4] | ✅ Main | — | ✅ BibTeX |
| [9] ONC HTI-1 | ✅ Ref [1] | ✅ Main | ✅ Ranks 11–12 + Synthesis | ✅ BibTeX |
| [10] FDA CDS Guidance | ✅ Ref [1] | — | ✅ Main + Synthesis | ✅ BibTeX |
| [11] FDA Diversity Action Plans | ✅ Ref [4] | — | ✅ Main + Synthesis | ✅ BibTeX |
| [12] Neuro-symbolic AI (ESMO 2026) | ✅ Ref [1] | ✅ Ref [5] | ✅ Main + Synthesis | ✅ BibTeX |
| [13] LLM biomarker (npj 2025) | ✅ Ref [1] | ✅ Ref [5] | ✅ Main + Synthesis | ✅ BibTeX |
| [14] CMS SMD 21-005 | ✅ Ref [2] | — | — | ✅ BibTeX |

---

## Architectural Logic for a New Physical AI Oncology Trial Paper

Based on the document's own synthesis section, these chunks collectively support a new paper structured around the following argument chain:

1. **Legal foundation (Chunks 1–2):** The U.S. legal stack gives patients rights over trial access, coverage, data, and consent — but rights alone do not guarantee timely exercise of those rights. The critical gap is operational latency.

2. **Regulatory enablers and constraints (Chunks 2–3):** ClinicalTrials.gov (FDAAA), algorithm transparency (HTI-1), CDS device boundary (FDA CDS Guidance), and diversity action plans (FDORA + FDA guidance) define the operating environment for any AI oncology trial system. Any physical AI system must be designed to operate within and leverage these regulatory structures.

3. **Current AI performance anchor (Chunk 3):** The neuro-symbolic multi-agent system (Rank 12, F1 0.82, 4x speedup) and LLM biomarker matching (Rank 13) are the current best-in-class evidence. A new paper must benchmark against or extend these results.

4. **Physical AI extension:** The synthesis section explicitly lists robotics for biopsy, imaging, drug delivery, surgery, and home-based sample collection as the physical layer that can close the remaining gap between digital trial-matching and actual patient participation. A new paper should operationalize "patient time-to-control" across the full physical-digital pipeline.

5. **Evaluation framework:** New paper metrics should span: calibrated accuracy, subgroup fairness (no F1 gap >10pp across demographics), explainability, patient comprehension, coverage feasibility, missed-trial reduction, travel burden, retention, avoidable adverse events, and self-reported decisional agency.

**Recommended reading order for Claude Code Opus:**
`chunk_01_intro_ranks1-4.md` → `chunk_02_ranks5-9.md` → `chunk_03_ranks10-13_future.md` → `chunk_04_bibtex.md`

All bracket citations `[N]` in chunks 1–3 resolve to BibTeX keys in `chunk_04_bibtex.md` using the mapping table above.

---

*End of README*
