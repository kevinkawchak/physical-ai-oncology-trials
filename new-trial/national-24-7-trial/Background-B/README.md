# README — AI Oncology Clinical Trial Knowledge Base
## Processing Guide for Claude Code Opus 4.7 (1M Context)

**Prepared for:** Authoring a new physical AI oncology trial paper
**Source document:** `Deep_Research_B_Oncology_Clinical_Trials.docx`
**Chunked into:** 3 Markdown files + this README
**Date compiled:** 2026-05-02

---

## Overview

This README describes three Markdown chunks derived from a deep-research document on the current baseline of AI-driven patient outcome prediction in oncology clinical trials. The chunks were created to:

1. Organize the body content into two thematically coherent halves (Chunks 1 & 2)
2. Isolate all BibTeX citations into a standalone reference library (Chunk 3)
3. Provide this README as a navigation and synthesis guide for building a new physical AI oncology trial paper

The goal of the new paper is to design and propose a **prospective, physically embedded AI oncology trial** — one where AI predictions are integrated in real time into clinical decision-making, not merely validated retrospectively. The chunks below define the evidence base, benchmarks, and gaps that the new trial must address.

---

## File Inventory

| File | Role | Sections Covered |
|------|------|-----------------|
| `chunk_01_baseline_and_prediction_domains.md` | **Evidence Base — Part 1** | What current AI is; survival/death prediction; adverse event and hospitalization prediction |
| `chunk_02_response_metrics_conclusions.md` | **Evidence Base — Part 2** | Response/progression prediction; metric interpretation; baseline conclusions; open questions and gaps |
| `chunk_03_bibtex_references.md` | **Citation Library** | All 9 BibTeX entries; key map table; source URL index |
| `README.md` *(this file)* | **Navigation & Synthesis Guide** | Cross-file correlations, authoring instructions, trial design implications |

---

## Chunk 1 — `chunk_01_baseline_and_prediction_domains.md`

### What It Contains

**Section 1: What Current AI in This Area Actually Is**
Establishes the architectural and methodological reality of today's oncology prediction AI: narrow supervised models (ridge Cox, SVM, random survival forests, XGBoost, elastic net), not general-purpose reasoning systems. Key inputs are routine bloods, clinical variables, metastatic burden, adverse-event logs, ePROs, passive sensors, and CT-derived features. The dominant validation paradigm is retrospective, not prospective.

**Section 2: Survival and Death Prediction**
Details four landmark studies:
- **SCORPIO** (Yoo 2025, `[1]`): 9,745 ICI patients, 21 cancer types, 33-variable ensemble. Time-dependent AUC 0.72–0.78 across test sets. Hazard ratios of 0.25 (low vs. high risk) to 0.48 (moderate vs. high). Validated on 12 phase III trial arms. → **Primary benchmark for survival prediction.**
- **Sun 2024 MYSTIC/durvalumab** (`[6]`): 60-day on-treatment AE signature predicts long-term OS. Meta-HR 0.83 in durvalumab arms vs. 1.02 in chemotherapy/placebo. → Demonstrates value of *early on-treatment signals* over pretreatment scores.
- **PBMF** (Arango-Argoty 2025, `[7]`): Contrastive learning for predictive (not merely prognostic) biomarker discovery. 15% improvement in survival risk in trial enrichment retrospective analysis. → Establishes AI's role in trial *design*, not just monitoring.
- **Xu 2025 HCC Multimodal Fusion** (`[8]`): CT deep learning + random survival forest. External C-index 0.74 (OS) and 0.69 (PFS). → Benchmark for imaging-integrated survival models.

**Section 3: Adverse Event and Hospitalization Prediction**
Details four studies across two adverse-event paradigms:
- **DREAM Challenge / Seyednasrollah 2017** (`[2]`): 34 teams, 61 models, docetaxel discontinuation in mCRPC. Best ensemble AUPRC 0.230 vs. random baseline 0.104. → **Primary benchmark for rare-event toxicity prediction in phase III data.**
- **Deng 2020** (`[3]`): DREAM follow-on; tabular, feature-engineered ML integrating survival status + AE severity. → Reinforces that tabular approaches dominate practical trial AI.
- **Iivanainen 2021** (`[4]`): Weekly ePRO + XGBoost, 34 patients, irAE detection AUC 0.99, irAE *onset* AUC 0.93. High performance but very small N. → Proof-of-concept only; not a field-wide benchmark.
- **Brouwer 2025** (`[5]`): Smartphone step count, 76 patients. 7-day hospitalization AUC 0.88; treatment modification AUC 0.28–0.51. → **Benchmark for passive-sensor short-horizon prediction; also establishes hard limits for heterogeneous endpoints.**

### Cross-References Within Chunk 1
- SCORPIO (`[1]`) and Xu 2025 (`[8]`) share survival modeling but differ in modality (blood/clinical vs. CT imaging) — compare them when selecting the input architecture for the new trial.
- Sun 2024 (`[6]`) and PBMF (`[7]`) both use ICI trial cohorts but serve different functions: `[6]` is a monitoring tool, `[7]` is a trial enrichment/design tool.
- DREAM Challenge (`[2]`) and Iivanainen 2021 (`[4]`) both target adverse events but differ radically in sample size, endpoint granularity, and reported metrics — they should be cited as *complementary* evidence, not competing.

---

## Chunk 2 — `chunk_02_response_metrics_conclusions.md`

### What It Contains

**Section 1: Response and Progression Prediction**
Covers binary response prediction (Zhao 2023 breast radiomics, `[9]`: AUC 0.961 validation) and PFS prediction (Xu 2025, `[8]`: C-index 0.69). Establishes that high-AUC response papers are often small and retrospective, while multimodal PFS predictions at 0.66–0.70 are more practically generalizable.

**Section 2: What the Quality Metrics Really Mean**
Provides a metric-selection guide critical for the new trial's statistical analysis plan:
- **Survival / PFS endpoints with censoring** → C-index, time-dependent AUC
- **Rare binary events (e.g., toxicity discontinuation, ~10% prevalence)** → AUPRC (not AUROC)
- **Imbalanced onset detection (e.g., irAE emergence)** → F1 and MCC as secondary metrics
- **Risk stratification tools** → Hazard ratio and Kaplan-Meier separation remain valid and clinically legible

Also introduces the **three-tier prediction horizon taxonomy**:
- Short-horizon (next ~7 days): hospitalization, acute AE
- Medium-horizon (first 60 days → long-term): treatment-specific survival signal
- Long-horizon (6 months–3 years): OS/PFS probability windows

**Section 3: Baseline Conclusion for Future AI Comparison**
Provides a performance benchmark table for the new trial's primary and secondary endpoints:
- Broad survival (multi-cohort, externally validated): time-dependent AUC / C-index 0.72–0.78
- Rare-event discontinuation (phase III): AUPRC ~0.23
- 7-day hospitalization (passive data): AUC ~0.88
- Binary response (small retrospective): AUC >0.90 — *interpret cautiously*

Also specifies the four requirements a future AI system must meet *simultaneously* to advance the field: external validation, cross-tumor calibration, actionable timing (not just rank order), and prospective outcome improvement.

**Section 4: Open Questions and Limitations**
Enumerates five gaps the new trial must close (sparse prospective deployment, calibration inconsistency, retrospective reliance, training/deployment mismatch, no prospective outcome-improvement evidence) and presents them in a structured table with direct implications for new trial design.

### Cross-References Within Chunk 2
- The metric section directly applies to all nine studies cited in Chunk 1 — it is the interpretive lens for all performance numbers.
- The baseline conclusion table in Chunk 2 synthesizes data from Chunk 1 studies (`[1]`, `[2]`, `[5]`, `[8]`, `[9]`) — do not read the table without the underlying evidence.
- The open questions section identifies gaps that collectively motivate the new trial's design; every gap maps back to one or more Chunk 1 studies.

---

## Chunk 3 — `chunk_03_bibtex_references.md`

### What It Contains

A complete, ready-to-use BibTeX library for all 9 studies cited across Chunks 1 and 2. Includes:
- A **reference key map** table mapping inline citation numbers `[1]`–`[9]` to BibTeX keys, first authors, years, and topic summaries
- All **9 full BibTeX entries** in a fenced code block, ready for `.bib` file insertion
- A **source URL index** with direct hyperlinks to each paper

### BibTeX Keys Quick Reference
- `Yoo2025SCORPIO` — SCORPIO, pan-cancer ICI, Nature Medicine 2025
- `Seyednasrollah2017Docetaxel` — DREAM challenge, docetaxel mCRPC, JCO CCI 2017
- `Deng2020mCRPC` — Treatment stratification mCRPC, iScience 2020
- `Iivanainen2021irAE` — ePRO + XGBoost irAE, BMC Med Inf 2021
- `Brouwer2025StepCount` — Step count ML prediction, JCO CCI 2025
- `Sun2024HealthSignals` — AE signature / durvalumab survival, iScience 2024
- `ArangoArgoty2025PBMF` — PBMF contrastive learning, Cancer Cell 2025
- `Xu2025MMF` — Multimodal fusion HCC ICI, npj Precision Oncology 2025
- `Zhao2023BreastRadiomics` — Radiomics anti-PD-1 breast, JITC 2023

---

## Cross-File Correlations

### How the Three Chunks Relate to Each Other

```
chunk_01 (Evidence base, Part 1)
    └──► Provides raw study data, performance numbers, and experimental
         designs for all survival and adverse-event prediction benchmarks.
         All [1]–[9] inline citations map to chunk_03 for full details.

chunk_02 (Evidence base, Part 2)
    └──► Interprets, synthesizes, and evaluates the studies from chunk_01.
         The metric section is the analytical key to understanding whether
         numbers in chunk_01 are strong or weak. The gaps table in chunk_02
         directly motivates the new trial design.

chunk_03 (Reference library)
    └──► Supports both chunk_01 and chunk_02. Every [N] inline tag in
         either body chunk resolves to a BibTeX entry here. The key map
         table in chunk_03 should be kept open while reading both body
         chunks.
```

### Thematic Correlation Map (for New Paper Authoring)

| New Paper Section | Primary Chunk | Key Studies / Keys |
|---|---|---|
| Introduction / Motivation | chunk_02 (Open Questions) | All 9 |
| Related Work — Survival Prediction | chunk_01 §2 | `[1]`, `[6]`, `[7]`, `[8]` |
| Related Work — AE & Toxicity | chunk_01 §3 | `[2]`, `[3]`, `[4]`, `[5]` |
| Related Work — Response / PFS | chunk_02 §1 | `[8]`, `[9]` |
| Methods — Metric Selection | chunk_02 §2 | `[1]`, `[2]`, `[4]`, `[8]` |
| Methods — Baseline Comparators | chunk_02 §3 | `[1]`, `[2]`, `[5]`, `[8]`, `[9]` |
| Discussion — Field Limitations | chunk_02 §4 | `[1]`, `[2]`, `[7]`, `[8]` |
| Bibliography | chunk_03 | All 9 BibTeX entries |

---

## Authoring Instructions for New Physical AI Oncology Trial Paper

The following guidance is intended for Claude Code Opus 4.7 processing this README alongside the three chunks.

### Trial Design Imperatives (derived from chunk_02 §4 gap table)

1. **Prospective deployment is mandatory.** The literature gap of "sparse prospective deployment" (chunk_02) means the new trial must embed AI predictions *during* patient care — the AI must influence a pre-specified decision point, not merely be applied post-hoc to frozen data.

2. **Define prediction horizons before enrollment.** The horizon taxonomy from chunk_02 §2 must map directly onto primary and secondary endpoints. Specify whether each endpoint is short-, medium-, or long-horizon and select the corresponding evaluation metric accordingly.

3. **Multi-site, multi-tumor enrollment from day one.** The external validation gap (chunk_02 §4) means single-site studies are insufficient. Enrollment design must include at least three independent sites and ideally two tumor types to enable internal cross-validation as a minimum standard.

4. **Calibration reporting is non-negotiable.** Unlike most prior studies in chunk_01, the new trial must report calibration curves (Brier score or reliability diagrams) alongside discrimination metrics for every AI model used prospectively.

5. **Intervention arm required.** The new trial must include a clinical-action arm — patients (or oncologists) in one arm receive AI risk scores and act on them; the control arm does not. This is the only way to close the "no prospective outcome-improvement evidence" gap.

### Benchmark Performance Targets (from chunk_02 §3 table)

When claiming the new AI improves over the state of the art, use these as minimum comparison thresholds:

- Survival (broad, externally validated): beat C-index / time-dependent AUC **> 0.78**
- Rare-event toxicity (phase III data): beat AUPRC **> 0.23**
- Short-horizon hospitalization (≤7 days, passive data): beat AUC **> 0.88**
- For narrower, single-modality response classification: contextualize AUC > 0.90 with sample size and external validation status

### Key Models to Cite as Prior Art

In order of primary relevance to a new prospective physical AI trial:

1. SCORPIO (`Yoo2025SCORPIO`) — the most broadly validated survival baseline; must be cited as primary prior art
2. DREAM Challenge (`Seyednasrollah2017Docetaxel`) — the clearest adverse-event benchmark from real phase III data
3. Sun 2024 (`Sun2024HealthSignals`) — establishes the medium-horizon on-treatment signal paradigm the new trial should extend
4. PBMF (`ArangoArgoty2025PBMF`) — most relevant if the new trial incorporates trial enrichment or adaptive biomarker selection
5. Brouwer 2025 (`Brouwer2025StepCount`) — most relevant if passive monitoring (wearables, smartphones) is part of the trial protocol

### Suggested Paper Structure

```
1. Abstract
2. Introduction
   - Clinical unmet need
   - Gap: no prospective physical AI trial evidence (chunk_02 §4)
3. Background / Related Work
   - Survival prediction (chunk_01 §2)
   - AE and hospitalization prediction (chunk_01 §3)
   - Response and PFS prediction (chunk_02 §1)
4. Methods
   - Trial design (prospective, interventional, multi-site)
   - AI architecture and input features
   - Prediction horizons and endpoint definitions
   - Evaluation metrics (chunk_02 §2 as justification)
   - Pre-specified baseline comparators (chunk_02 §3 table)
5. Results
6. Discussion
   - Comparison to existing benchmarks
   - Limitations and calibration analysis
   - Implications for trial enrichment (PBMF paradigm)
7. Conclusion
8. References (chunk_03 BibTeX library + new trial citations)
```

---

## Notes on Inline Citation Numbering

The inline tags `[1]`–`[9]` used throughout Chunks 1 and 2 correspond **directly and exclusively** to the 9 BibTeX entries in Chunk 3. There are no additional references beyond these nine. The source document used higher inline numbers (e.g., `[10]`–`[36]`) that were internal cross-reference anchors within the original Word document, not additional bibliography entries. All unique sources resolve to the 9 keys listed in chunk_03.

---

*End of README*
