# README — AI Oncology Deep Research Corpus

> **Prepared for:** Claude Code Opus 4.7 (1M context)  
> **Purpose:** Navigation guide and semantic map for processing chunked research files in service of drafting a new **physical AI oncology trial paper**  
> **Source document:** `Deep_Research_A_Oncology.docx`  
> **Generated:** 2026-05-02  
> **Total chunks:** 3 markdown files + this README

---

## 1. File Inventory

| File | Role | Sections | Primary Focus |
|---|---|---|---|
| `chunk_01_baseline_and_short_horizon.md` | Body Part 1 | §1–3 | Baseline landscape + short/medium-horizon prediction models |
| `chunk_02_multimodal_and_limitations.md` | Body Part 2 | §4–6 | Immunotherapy models, measurement standards, open questions |
| `chunk_03_bibtex.md` | Reference library | All 17 refs | Full BibTeX entries + cross-reference index |
| `README.md` | This file | — | Corpus navigation and authoring guidance |

---

## 2. Chunk-by-Chunk Detail

---

### 2.1 `chunk_01_baseline_and_short_horizon.md`

**What it contains:**

This file establishes the empirical state of the art as of 2025–2026 for the first half of the research landscape. It opens with the critical framing premise: current oncology AI is not a unified system but a heterogeneous collection of narrow supervised models, and despite widespread claims, machine learning does not consistently outperform traditional Cox regression on structured real-world survival data (SMD in AUC/C-index = 0.01, 95% CI −0.01 to 0.03). This null result is the anchor benchmark for the entire corpus.

The file then covers three concrete evidence domains:

1. **Short-horizon EHR-based mortality prediction** — The Manz 2020 gradient-boosting model (AUC 0.89, 24,582 patients, 4–8 day prospective lead time) represents the strongest documented baseline for near-term death prediction from full EHR data. Patient-reported-outcome-only models yield AUROCs of 0.69–0.76.

2. **Adverse event prediction during treatment** — SHIELD-RT is the gold standard for operational deployment: a prospective RCT (Hong 2020) that reduced acute-care events 45% and costs 48%, externally validated across 22,000+ courses (Elia 2025) at AUROC 0.756–0.770. Chemotherapy cardiotoxicity (Li 2022, XGBoost AUC 0.816) and neoadjuvant breast cancer toxicity (Cai 2024, AUROC → 0.75 with dose-intensity features) extend this domain.

3. **Medium/long-horizon survival prediction** — AIM-LCpro (multimodal NSCLC, C-index 0.785–0.804 internal / 0.693–0.749 external) and PROGPATH (pancancer ViT foundation model, C-index 0.713–0.805 across 17 external cohorts) represent the strongest current baselines. The lung-cancer image-AI meta-analysis (Yuan 2025) provides a literature-level upper bound at AUC 0.90, with publication bias acknowledged.

**Key numerical benchmarks in this file:**

| Domain | Model | Metric | Value |
|---|---|---|---|
| 180-day mortality (EHR) | Manz 2020 gradient boosting | AUC | 0.89 |
| 180-day mortality (PRO) | Ovarian voting ensemble | AUROC | 0.76 |
| RT acute-care events | SHIELD-RT (external) | AUROC | 0.756–0.770 |
| Chemo cardiotoxicity | XGBoost (Li 2022) | AUC | 0.816 |
| HN radiation toxicity | Meta-analysis (Ugwu 2026) | Pooled AUROC | 0.76 |
| NSCLC 5-yr DFS (external) | AIM-LCpro | C-index | 0.693–0.749 |
| Pancancer survival (external) | PROGPATH | C-index | 0.713–0.805 |
| Lung image AI (literature) | Meta-analysis (Yuan 2025) | AUC | 0.90 (upper bound) |

**Citations used:** [1][2][3][4][5][6][7][8][9][10][11][15]

**Leads into:** `chunk_02_multimodal_and_limitations.md` (continues the body)

---

### 2.2 `chunk_02_multimodal_and_limitations.md`

**What it contains:**

This file covers the second half of the body, organized around three themes: immunotherapy and multimodal precision medicine (Section 4), the correct measurement framework for benchmarking future systems (Section 5), and the limitations that define what "genuine advance" would look like (Section 6).

**Section 4 — Immunotherapy & Multimodal Models:**

Three distinct model lineages are documented:

- **Vanguri 2022 (multimodal PD-(L)1 NSCLC):** Radiology + pathology + genomics integration reaches AUC 0.80 vs PD-L1 alone (0.73) and TMB alone (0.61). Establishes multimodality as the performance ceiling for ICI response prediction.
- **SCORPIO (Yoo 2025):** Operationally practical model using only routine blood tests and clinical data. Median time-dependent AUC 0.763 (hold-out) to 0.725 (external). HR 0.25 for low vs high risk. Clinical benefit separation: 55.96% vs 12.12%.
- **Captier 2025 (late-fusion NSCLC pembrolizumab):** Clinical + PET + pathology + RNA-seq across 317 patients. C-index 0.75 OS; AUC 0.81 (1-year death); but AUC only 0.67 (6-month progression) — illustrating that short-run progression remains harder to model than long-run survival.

**Section 5 — Measurement Baseline:**

Defines the correct metric stack for any new trial paper:
- **AUROC** (binary outcomes), **time-dependent AUC** (survival at horizons), **C-index** (censored survival), **Brier score** (calibration), **PPV / sensitivity / specificity** (clinical deployment thresholds)
- Consolidated benchmark table by domain: 0.75–0.82 (adverse events), ~0.89 (EHR mortality), 0.69–0.76 (PRO mortality), 0.70–0.81 (multimodal survival)
- **Four endpoint families** any new system must address: (1) imminent adverse events, (2) fixed-horizon mortality, (3) fixed-horizon recurrence/survival, (4) treatment-specific benefit

**Section 6 — Open Questions and Limitations:**

Three structural gaps that a new trial paper should directly address:
1. **Retrospective/single-center bias** — most literature lacks external validation and is subject to data shift
2. **Surrogate endpoint misalignment** — OS predicted well; clinical benefit and short-run progression are harder; dynamic, multi-endpoint models are absent
3. **Internal validation ≠ clinical impact** — SHIELD-RT is the only current example of an externally validated, prospectively randomized, clinically impactful model

**Key numerical benchmarks in this file:**

| Domain | Model | Metric | Value |
|---|---|---|---|
| ICI response (multimodal) | Vanguri 2022 | AUC | 0.80 |
| ICI OS prognosis (blood only) | SCORPIO (hold-out) | Time-dep AUC | 0.763 |
| ICI OS prognosis (external) | SCORPIO | Time-dep AUC | 0.725 |
| ICI benefit rate (low risk) | SCORPIO | Clinical benefit | 55.96% |
| NSCLC pembrolizumab OS | Captier 2025 late-fusion | C-index | 0.75 |
| NSCLC 1-yr death | Captier 2025 | AUC | 0.81 |
| NSCLC 6-mo progression | Captier 2025 | AUC | 0.67 |

**Citations used:** [1][2][3][4][5][6][7][8][9][10][11][12][13][14][15][16][17]

**Continues from:** `chunk_01_baseline_and_short_horizon.md`  
**References resolved in:** `chunk_03_bibtex.md`

---

### 2.3 `chunk_03_bibtex.md`

**What it contains:**

All 17 BibTeX entries extracted from the source document, each annotated with:
- Its **role in the research narrative** (which sections cite it, what claim it supports)
- Its **study design type** (`[META-ANALYSIS]`, `[PROSPECTIVE]`, `[RANDOMIZED]`, `[EXTERNAL VALIDATION]`, `[RETROSPECTIVE]`, `[REPORTING GUIDELINE]`)
- Full **BibTeX-formatted citation block** ready for `.bib` file insertion

Additionally, the file includes a **17×13 citation-to-section cross-reference matrix** mapping each reference to each section of Chunks 01 and 02.

**Structural classification of the 17 references:**

| Type | Count | Citation Keys |
|---|---|---|
| Meta-analysis / systematic review | 4 | [1][2][9][11] |
| Prospective / RCT | 3 | [3][4][5] (Manz, Sidey-Gibbons, Hong) |
| External multi-institutional validation | 3 | [6][10][13] (Elia, Li AIM-LCpro, SCORPIO) |
| Retrospective model development | 3 | [7][8][12] (Li cardiotoxicity, Cai breast, Vanguri) |
| Prospective multimodal cohort | 2 | [14][15] (Captier, PROGPATH) |
| Reporting guideline | 2 | [16][17] (TRIPOD+AI, CREMLS) |

**Critical pair for new trial paper compliance:**  
References [16] (TRIPOD+AI, Collins 2024, BMJ) and [17] (CREMLS, El Emam 2024, JMIR) define the reporting standards any new physical AI oncology trial paper **must** follow. These should be consulted first when drafting the Methods section.

---

## 3. Cross-File Correlations

### 3.1 Narrative Continuity Between Chunks 01 and 02

Chunks 01 and 02 together form a single continuous argument structured as:

```
[Chunk 01]
  § 1 → Problem statement: AI has not improved on Cox regression
  § 2 → Short-horizon domain: what works, what the numbers are
  § 3 → Long-horizon domain: image models, foundation models
[Chunk 02]
  § 4 → Immunotherapy domain: multimodal, operational, late-fusion
  § 5 → Synthesis: the right metric stack + benchmark table
  § 6 → Gaps: three failure modes the new trial must address
```

The logical flow is: **establish the null result → document best-in-class narrow models → show the immunotherapy ceiling → define the measurement standard → identify what a new system must overcome.**

### 3.2 Thematic Threads Spanning All Three Chunks

| Theme | Chunk 01 | Chunk 02 | Chunk 03 |
|---|---|---|---|
| **External validation gap** | AIM-LCpro internal vs external drop (§3.1) | SCORPIO external AUC decay (§4.2); limitation (§6.1) | [6][10][13][15] all involve external validation cohorts |
| **Multimodality as ceiling** | PROGPATH integrates WSI + clinical (§3.3) | Vanguri/Captier integrate 3–4 modalities (§4) | [12][14][15] are all multimodal architecture papers |
| **Calibration & clinical impact** | SHIELD-RT Brier < 0.06 (§2.3) | Benchmark table includes Brier; SHIELD-RT as gold standard (§5, §6.3) | [5][6] are RCT/external; [16][17] mandate calibration reporting |
| **Surrogate endpoint difficulty** | Short-run progression harder than OS noted implicitly | Explicitly: Captier AUC 0.67 (6-mo) vs 0.81 (1-yr) (§4.3, §6.2) | [13][14] directly illustrate this |
| **Reporting standards** | Invoked for benchmark framing (§1) | Fully specified in §5 | [16][17] are the complete standards |

### 3.3 Citation Overlap Analysis

The most-cited references across both body chunks (indicating their structural importance):

| Ref | Chunks | Key Claim Supported |
|---|---|---|
| **[1]** Huang 2025 | §1, §2 baseline, §5, §6 | ML = Cox null result — foundational benchmark anchor |
| **[16]** Collins TRIPOD+AI | §1, §5, §6 | Reporting standard — mandatory for new trial methods |
| **[17]** El Emam CREMLS | §1, §5, §6 | Reporting standard — mandatory for new trial methods |
| **[6]** Elia SHIELD-RT external | §2.3, §5, §6.3 | Only externally validated clinical-impact model in corpus |
| **[13]** Yoo SCORPIO | §4.2, §5, §6 | Operational ICI baseline with clinical stratification |

---

## 4. Authoring Instructions for Claude Code Opus 4.7

The following guidance is for the downstream authoring agent that will use this corpus to draft or scaffold a new physical AI oncology trial paper.

### 4.1 What This Corpus Provides

This corpus is a **comprehensive empirical baseline literature review** covering the state of AI-driven oncology prediction as of 2025–2026. It documents:
- The null result for ML vs Cox regression at the field level
- Best-in-class models for four distinct clinical prediction domains
- The correct metric stack for fair comparison
- Three structural limitations any new paper must address or acknowledge

### 4.2 What This Corpus Does Not Provide

This corpus does **not** contain:
- Details of the new physical AI system or trial protocol being designed
- Patient enrollment criteria, IRB language, or regulatory framing
- Clinical trial phase classification (Phase I/II/III feasibility)
- Statistical power calculations for the new trial
- Specific intervention protocols (dosing, hardware, scheduling)

These elements must be supplied separately or generated from the new trial design.

### 4.3 Recommended Usage Pattern

When drafting a new trial paper, use the chunks in this order:

1. **Start with `chunk_03_bibtex.md`** — Ingest the cross-reference matrix to understand which references are load-bearing for which claims. Import all BibTeX entries into the new paper's `.bib` file immediately.

2. **Use `chunk_02_multimodal_and_limitations.md` §6** as the **gap analysis** that motivates the new trial. Sections 6.1–6.3 directly define the three deficiencies the new system must address.

3. **Use `chunk_02_multimodal_and_limitations.md` §5** (benchmark table) as the **primary comparison target table** in the Results section. Any new model's metrics should be presented in the same format.

4. **Use `chunk_01_baseline_and_short_horizon.md`** for the **Related Work / Background** section. Organize by prediction domain (adverse events → survival → immunotherapy) using the section structure provided.

5. **Cite [16] and [17] in the Methods section** to declare TRIPOD+AI and CREMLS compliance. These are non-negotiable for modern oncology ML trial reporting.

### 4.4 Key Claims to Position Against in the New Paper

The new paper must explicitly position itself against these documented baselines:

| Claim to Beat | Source | Current Value |
|---|---|---|
| Best EHR short-horizon mortality | Manz 2020 [3] | AUC 0.89 |
| Best RT adverse event prevention | SHIELD-RT [5][6] | 45% event reduction; AUROC 0.756–0.770 |
| Best ICI OS prognosis (routine data) | SCORPIO [13] | Time-dep AUC 0.763 |
| Best multimodal ICI response | Vanguri 2022 [12] | AUC 0.80 |
| Best pancancer survival (external) | PROGPATH [15] | C-index 0.713–0.805 |
| Literature ceiling (single-center) | Yuan 2025 meta [11] | AUC 0.90 (with bias caveat) |

### 4.5 Structural Gaps the New Trial Should Address

The three gaps from Section 6 of Chunk 02 should map directly to trial design choices:

| Gap | Design Response |
|---|---|
| Retrospective / single-center bias | Prospective, multi-site enrollment with pre-registered analysis plan |
| Surrogate endpoint misalignment | Dual primary endpoints: OS **and** clinical benefit; secondary endpoint: short-run progression |
| Internal validation ≠ clinical impact | Prospective randomized or stepped-wedge design with clinical workflow integration (cf. SHIELD-RT model) |

---

## 5. Quick Reference — Metric Benchmarks by Domain

| Prediction Domain | Best Current Externally Validated Performance | Reference |
|---|---|---|
| Short-horizon mortality (EHR, 180-day) | AUC **0.89** | [3] Manz 2020 |
| Short-horizon mortality (PRO-only) | AUROC **0.69–0.76** | [4] Sidey-Gibbons 2022 |
| RT acute-care adverse events | AUROC **0.756–0.770** | [6] Elia 2025 |
| Chemo cardiotoxicity (30-day) | AUC **0.816** | [7] Li 2022 |
| HN radiation toxicity (pooled) | AUROC **0.76** | [9] Ugwu 2026 |
| NSCLC multimodal DFS (5-yr, external) | C-index **0.693–0.749** | [10] Li 2025 |
| Lung image AI (literature upper bound) | AUC **0.90** | [11] Yuan 2025 |
| ICI response prediction (multimodal) | AUC **0.80** | [12] Vanguri 2022 |
| ICI OS (routine blood, external) | Time-dep AUC **0.725** | [13] Yoo 2025 |
| ICI OS (late-fusion multimodal) | C-index **0.75** | [14] Captier 2025 |
| Pancancer survival (foundation model, external) | C-index **0.713–0.805** | [15] Yuan 2025 PROGPATH |

---

*End of README*
