# README — AI in Oncology Trials and Patient Outcomes: Chunked Research Files

## Purpose of This Document

This README is provided to Claude Code Opus 4.7 (1M context) as a navigation and processing guide for three chunked Markdown files derived from a single deep-research document titled **"AI in Oncology Trials and Patient Outcomes"**. The full document is a rigorously cited evidence synthesis intended to serve as the evidentiary baseline for a new physical AI oncology trial paper. All three chunk files, together with this README, should be loaded simultaneously into context before any drafting, synthesis, or analytical work begins.

---

## File Inventory

| File | Role | Sections Covered |
|---|---|---|
| `part1_scope_and_radiotherapy.md` | Body — Part 1 | Scope and baseline; How AI has improved radiotherapy safety and efficacy |
| `part2_systemic_surgical_conclusions.md` | Body — Part 2 | How AI has improved systemic therapy selection and dosing; What current surgical and robotic-adjacent evidence shows; Baseline conclusions for future AI and robotics; Open questions and limitations |
| `part3_bibtex_references.md` | Bibliography | All 13 BibTeX entries in citation-key order [1]–[13] |

---

## Part 1 — `part1_scope_and_radiotherapy.md`

### What This File Contains

This file opens the source document. It establishes the evidentiary landscape for AI in oncology as of 2025–2026, and then presents the most thoroughly validated clinical domain in the literature: AI-assisted radiotherapy planning and adaptive delivery.

**Section: Scope and baseline**
Defines the overarching claim — AI has produced measurable benefit, but unevenly. Introduces the critical analytical distinction between *process improvements* and *patient-health improvements*, which is the methodological backbone for evaluating every study cited across all three files. Establishes the safety and efficacy metric vocabulary used throughout the document (grade 3–4 toxicity rates, Dice-Sørensen coefficients, distant metastasis, RECIST response, etc.). Cites references [1] and [2].

**Section: How AI has improved radiotherapy safety and efficacy**
Presents four primary studies in chronological order (2021 → 2022 → 2025 → 2026):

- **Cha et al. 2021** [6] — Prostate MR-based autosegmentation, 173 patients, 28-minute median contouring, 65% minor-edit-only cases; establishes the need for continued clinician oversight.
- **Hosny et al. 2022** [5] — NSCLC deep-learning CT segmentation, 2,208 patients, 65% faster segmentation, 32% less interobserver variability, expert-equivalent dose coverage; first multicenter generalizability evidence.
- **Niu et al. 2026** [7] — Prospective multicenter thoracic OAR study, 500 patients, 5 centers; Dice 0.902 vs 0.857, HD95 5.20 mm vs 8.01 mm, 81.63% time reduction; highest-quality radiotherapy contouring evidence in the corpus.
- **Preziosi et al. 2025** [8] — Online adaptive prostate radiotherapy, 31 patients; PTV V95% gains of 10.4–11.8%, CTV D98% gains of 2.6–2.9%, workable session times (25–32 min); demonstrates shift from contour-assistance toward real-time treatment adaptation.

### Cross-File Dependencies from Part 1

- The safety/efficacy metric framework introduced in the **Scope and baseline** section is the analytical lens applied to every study in both Part 1 and Part 2. Claude Code should treat this framework as a shared schema.
- References [5], [6], [7], [8] introduced in Part 1 are cited again in the **Baseline conclusions** section of Part 2. Their BibTeX entries are in Part 3.
- Reference [2] (Macheka 2024 systematic review) appears first in Part 1 and recurs throughout Part 2 as a methodological anchor; its BibTeX entry is in Part 3.

---

## Part 2 — `part2_systemic_surgical_conclusions.md`

### What This File Contains

This file covers the remaining four sections of the source document and carries the analytical weight of the paper's argument about where oncology AI stands today and what a future trial must demonstrate to exceed it.

**Section: How AI has improved systemic therapy selection and dosing**
Presents five primary studies or analyses on AI-guided drug dosing and patient-stratification for hormone therapy and targeted agents:

- **Pantuck et al. 2018** [3] — N-of-1 CURATE.AI case (ZEN-3694 + enzalutamide in mCRPC); proof-of-possibility for patient-specific AI dosing; quadratic dose-response model.
- **Blasiak et al. 2025** [4] — PRECISE CURATE.AI feasibility trial, 49 dosing events in capecitabine-treated solid tumor patients; all 36 relevant recommendations delivered on time; 35/36 physician adherence; no grade 4 toxicities; controlled personalization under physician supervision.
- **Spratt et al. 2023** [9] — AI predictive model for short-term ADT benefit in localized prostate cancer; model-positive subgroup (34%, n=543): hazard ratio 0.34 for distant metastasis; model-negative subgroup (66%, n=1,051): hazard ratio 0.92; simultaneous intensification and de-intensification logic.
- **Armstrong et al. 2025** [10] — Phase III-trial-based validation for long-term vs short-term ADT, 6 phase III trials; 15-year distant-metastasis risk difference 14% in biomarker-positive men, 0% in biomarker-negative men.
- **Parker et al. 2025 / UCL 2025** [11, 12] — STAMPEDE-based MMAI model for abiraterone selection; biomarker-positive (25% of cohort): 5-year mortality 17% → 9%; biomarker-negative: 7% → 4% (not significant); conference-stage at time of writing.

**Section: What current surgical and robotic-adjacent evidence shows**
Presents one primary study:

- **Li et al. 2025** [13] — Non-contact AI-assisted intraoperative 3D navigation in lung cancer segmentectomy, prospective randomized; navigation time 50–107s vs 120–234s; operation time 84.23 vs 101.84 minutes; sterility-preserving gesture control; no hard oncologic endpoint.

**Section: Baseline conclusions for future AI and robotics**
Synthesizes the entire corpus into a three-phase historical arc (single-patient dosing → radiotherapy → biomarker-guided selection) and states the fairest present-day baseline as *augmented oncology under clinician supervision*. Lists the specific outcome benchmarks any future AI or robotic system must exceed: grade ≥3 toxicity rates, unnecessary systemic treatment rates, patient-reported quality of life, local-failure and distant-metastasis rates, overall survival on intensification, major-edit/override rates, and session/operating time sustainability.

**Section: Open questions and limitations**
Identifies four standing limitations: (1) predominance of surrogate endpoints; (2) retrospective application of AI to archived phase III material rather than prospective randomization; (3) the conference-stage status of the abiraterone result; (4) the clinician-in-the-loop dependency of all currently safe AI deployments.

### Cross-File Dependencies from Part 2

- All inline citations [2]–[13] reference BibTeX keys in Part 3.
- References [5]–[8] from Part 1 are explicitly cited again in the **Baseline conclusions** section of Part 2; Claude Code must treat those studies as already introduced.
- References [9]–[12] introduced in Part 2 are prostate cancer treatment-selection studies that form a logical cluster: [9] established the ADT benefit-prediction logic, [10] validated it for long-term ADT, and [11]/[12] extended it to abiraterone — the three should be read as a sequential evidentiary chain.
- References [3] and [4] form a CURATE.AI dosing pair: [3] (2018 N-of-1) is the proof-of-concept and [4] (2025 PRECISE trial) is the first prospective feasibility validation.

---

## Part 3 — `part3_bibtex_references.md`

### What This File Contains

All 13 BibTeX entries for the studies cited in Part 1 and Part 2, listed sequentially by citation number [1]–[13]. Each entry includes the citation key, author list, title, journal or venue, year, volume, pages where available, DOI, and URL.

### BibTeX Key Index

| Citation Number | BibTeX Key | First Author | Year | Domain |
|---|---|---|---|---|
| [1] | `Verma2025AIOncologyTrials` | Verma | 2025 | Epidemiology / registry review |
| [2] | `Macheka2024ProspectiveEvaluation` | Macheka | 2024 | Systematic review / methodology |
| [3] | `Pantuck2018CURATEAIProstate` | Pantuck | 2018 | Systemic therapy / AI dosing |
| [4] | `Blasiak2025PRECISE` | Blasiak | 2025 | Systemic therapy / AI dosing |
| [5] | `Hosny2022NSCLCRadiotherapy` | Hosny | 2022 | Radiotherapy / NSCLC segmentation |
| [6] | `Cha2021ProstateAutosegmentation` | Cha | 2021 | Radiotherapy / prostate autoseg |
| [7] | `Niu2026ThoracicOAR` | Niu | 2026 | Radiotherapy / thoracic OAR |
| [8] | `Preziosi2025AIOART` | Preziosi | 2025 | Radiotherapy / adaptive prostate |
| [9] | `Spratt2023ADTBenefit` | Spratt | 2023 | Treatment selection / ADT short-term |
| [10] | `Armstrong2025LongTermADT` | Armstrong | 2025 | Treatment selection / ADT long-term |
| [11] | `Parker2025STAMPedeARPI` | Parker | 2025 | Treatment selection / abiraterone |
| [12] | `UCL2025AbirateroneAI` | UCL News | 2025 | Treatment selection / abiraterone (inst.) |
| [13] | `Li2025LungSurgeryNavigation` | Li | 2025 | Surgical navigation |

### Notes on Part 3

- Entry [12] (`UCL2025AbirateroneAI`) is an institutional news summary, not a peer-reviewed manuscript. It corresponds to the same STAMPEDE-based ASCO 2025 result as [11] but from UCL's institutional communications. Both should be cited together for the abiraterone claim; neither alone is sufficient as a primary source.
- Entry [7] (`Niu2026ThoracicOAR`) carries a 2026 publication year, reflecting the document's most recent evidence; this is intentional and not an error.
- DOIs for all entries where available should be treated as the canonical identifier for retrieval. URLs are supplementary.

---

## How All Three Files Correlate to Each Other

### Logical Architecture

The three files together constitute a complete evidence synthesis structured as:

```
Part 1 (foundation + radiotherapy evidence)
    ↓ shares metric framework and radiotherapy references [5–8]
Part 2 (systemic + surgical + synthesis + limitations)
    ↓ draws on all citations from Part 1 plus adds [3,4,9–13]
Part 3 (bibliography)
    ↑ provides resolvable identifiers for every inline citation in Part 1 and Part 2
```

### Shared Analytical Framework

The safety/efficacy distinction introduced in Part 1 (Scope and baseline) governs the structure of every study summary in both Part 1 and Part 2. When Claude Code drafts new trial methodology, hypotheses, or outcome criteria, this framework — *what counts as a safety metric versus an efficacy metric in AI oncology trials* — should be applied consistently.

### Thematic Clusters Across Parts 1 and 2

Three thematic clusters span the body files and should be treated as unified evidence threads when writing the new trial paper:

**Cluster A — Radiotherapy AI (Part 1 exclusively):**
References [5], [6], [7], [8] — deep learning for contouring, segmentation, and adaptive delivery. Collectively they demonstrate a progression from single-center proof-of-concept to multicenter prospective validation. The outcome benchmark set here (Dice score, HD95, contouring time, session time, DVH metrics) is the radiotherapy standard any new trial arm should be measured against.

**Cluster B — AI-Guided Systemic Dosing (Part 2, systemic therapy section):**
References [3] and [4] — the CURATE.AI pair. The 2018 case is the hypothesis generator; the 2025 PRECISE trial is the first clinical feasibility test. For a new physical trial, this cluster defines the minimum bar for a patient-specific AI dosing arm: physician-constrained recommendations, on-time delivery, adherence tracking, grade 3–4 toxicity monitoring, and RECIST/tumor-marker efficacy assessment.

**Cluster C — AI-Guided Treatment Selection in Prostate Cancer (Part 2, systemic therapy section):**
References [9], [10], [11], [12] — the multimodal digital pathology biomarker thread. This cluster represents the most clinically mature AI evidence in the document, with hard endpoints (distant metastasis, mortality) and phase III trial validation. For a new trial, this cluster sets the expectation for what a well-validated predictive biomarker should demonstrate: statistically significant benefit in the positive subgroup, absence of benefit in the negative subgroup, and a clinical utility argument (avoid toxicity or intensify appropriately).

**Cluster D — Surgical and Robotic Navigation (Part 2, surgical section):**
Reference [13] only — the lung surgery navigation study. This is the thinnest cluster in the document (single study, no survival endpoint) but the most directly relevant to any trial involving robotic or navigation-assisted surgical oncology. It defines the current efficiency and sterility-preservation baseline without proving oncologic superiority.

### Limitations Applicable to New Trial Design (from Part 2)

The Open Questions section of Part 2 lists four methodological gaps that the new physical AI oncology trial paper should explicitly address in its design:

1. **Surrogate endpoint reliance** — new trial should include patient-reported outcomes and survival arms, not only dosimetric or contour metrics.
2. **Retrospective AI validation** — new trial should prospectively randomize patients to AI-guided versus standard care rather than applying AI retrospectively to archived data.
3. **Conference-stage evidence** — any abiraterone-selection or STAMPEDE-derived claim requires the full peer-reviewed publication before being cited as definitive in the new paper.
4. **Clinician-in-the-loop dependency** — new trial design should explicitly specify the degree of AI autonomy being tested and distinguish augmented from autonomous AI arms.

---

## Instructions for Claude Code Opus 4.7

1. Load all three chunk files and this README into context simultaneously before beginning any drafting or analysis.
2. Treat Part 3 as the authoritative reference list; resolve all inline citations [1]–[13] appearing in Parts 1 and 2 against the BibTeX keys in Part 3.
3. Use the safety/efficacy metric framework from Part 1 (Scope and baseline) as the analytical schema for structuring any new trial methodology, endpoints, or outcome reporting.
4. Apply the thematic clusters (A, B, C, D) as the organizational logic for literature review sections of the new paper.
5. When generating new trial design elements, benchmark against the specific quantitative thresholds documented in Parts 1 and 2 (e.g., Dice ≥0.90, HD95 ≤5.2 mm for thoracic OAR; hazard ratio thresholds from the ADT selection models; adherence rates and toxicity ceilings from the PRECISE trial).
6. Flag the four limitations from Part 2 (Open questions) explicitly in the new paper's limitations and future directions sections, and demonstrate how the new trial design addresses each one.
7. Do not treat reference [12] (UCL News) as a primary source; pair it with [11] (Parker et al. 2025) whenever the abiraterone claim is cited.
