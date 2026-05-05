# README — Deep Research Chunk Set: AI and Patient Control in Oncology Trials

## Purpose of This README

This README is written for **Claude Code Opus 4.7 (1M context)** and provides a complete orientation to the three chunked Markdown files derived from the source document `Deep-Research-1.docx`. The document is a deep-research synthesis titled **"AI and Patient Control in Oncology Trials"**. It surveys the published clinical trial evidence on how AI-based digital health systems have changed patients' practical sense of control over their cancer experience, from early rule-based symptom monitors through to multimodal predictive platforms and autonomous coaching agents.

The three chunk files, taken together, constitute the complete, unabridged source text. All content is reproduced word for word from the original document with no paraphrasing, summarization, or added headings. The chunking is structural, not editorial.

The intended downstream task is the drafting of a **new physical AI oncology trial paper** — a prospective, human-subjects clinical trial protocol (or companion methods paper) that introduces a physically embodied AI system (e.g., a robotic or wearable-robotic agent) into an oncology care setting and must situate that system against the existing software-only evidence base catalogued here.

---

## File Inventory

| File | Sections Covered | Citation Keys Used | Approx. Word Count |
|---|---|---|---|
| `chunk_1_foundations.md` | What the evidence means by patient control; Early baseline from rule-based symptom AI; EHR-integrated adverse-event AI and the move toward explicit self-efficacy; Predictive and multimodal survivorship AI | [1][2][3][4][5][6][7][8][9][10][11] | ~1,350 words |
| `chunk_2_advanced_systems.md` | AI-generated survivorship plans and AI coaching; Baseline for future AI and robotics; Open questions and limitations | [9][10][11][12][13][14][15][16][17][18][19] | ~1,050 words |
| `chunk_3_references_bibtex.md` | BibTeX entries [1]–[11]; Source URL reference list | All 11 citation keys | ~650 words |

---

## Detailed Description of Each File

### `chunk_1_foundations.md`

**Role in the corpus:** Establishes the definitional and historical foundation of the entire review. This file must be read first because it defines the four-part taxonomy of patient control (informational, behavioral, relational, and decisional) that every subsequent section and all future trial endpoints should be mapped against.

**Sections and their function:**

**"What the evidence means by patient control"** — Provides the core working definition used throughout the review. States that patient control has been measured along a developmental arc: toxicity containment (earliest studies) → self-efficacy and health-status instruments (middle generation) → patient activation, personalized survivorship planning, multimodal monitoring, and behavior-change metrics (newest generation). Explicitly names three validated measurement instruments — the Communication and Attitudinal Self-Efficacy scale for cancer (CASE-cancer), the Patient Activation Measure (PAM), and self-efficacy scales for chronic disease management — that recur as comparators across chunks 1 and 2 and should anchor the primary and secondary endpoints of any new trial.

**"Early baseline from rule-based symptom AI"** — Describes the ASyMS/eSMART program (citations [1][2][3]) as the first convincing baseline. ASyMS combined mobile symptom reporting, rule-based clinical decision support, CTCAE-linked logic, automated self-care advice, and clinician alerts. Key finding: the system reduced toxicity burden and improved anxiety, quality of life, self-efficacy, and unmet supportive care needs. Establishes the mechanism of control gain as "shrinking the gap between symptom onset and action." This is the earliest quantified baseline that any new physical AI trial must exceed or reference.

**"EHR-integrated adverse-event AI and the move toward explicit self-efficacy"** — Describes the eRAPID program (citations [4][5]). eRAPID was a phase III RCT that integrated patient-reported adverse events directly into the EHR, generated severity-dependent self-management advice, and prompted hospital contact for serious symptoms. Key quantitative finding: self-efficacy improved significantly at 18 weeks (mean advantage +0.48, 95% CI 0.13–0.83, p=0.0073). This section defines the middle-generation control benchmark: EHR-integrated symptom management measurably improves patient confidence in managing treatment side effects. This is the minimum software-only threshold that a new embodied-AI trial should frame its hypothesis against.

**"Predictive and multimodal survivorship AI"** — Describes the PERSIST program (citations [6][7][8]). PERSIST used a mobile health app, big-data predictive models, smart-band sensing, software-based mood estimation, and a Multimodal Risk Assessment and Symptom Tracking framework fusing vital signs, PROMs, PREMs, and linguistic/vocal/facial cues. Primary endpoint: CASE-cancer (perceived self-efficacy). Secondary endpoint: PAM. Key sobering finding: no statistically significant changes in CASE-cancer or PAM were found at the trial level, even though qualitative data showed patients felt more in control. This ceiling-effect finding is critical for new trial design: patients with already-high baseline activation may not show large instrument-level gains, requiring careful stratification or novel endpoints.

**Key citation bridge to chunk 2:** Citations [6][7][8] appear in both chunks. The PERSIST discussion begun in chunk 1 informs the "Baseline for future AI and robotics" section in chunk 2.

---

### `chunk_2_advanced_systems.md`

**Role in the corpus:** Synthesizes the most recent evidence (late-generation systems) and draws the direct implications for future trial design — including trial design for a physical AI system. This file is the primary source for hypothesis framing, comparator arm specification, endpoint selection, and the articulation of what a next-generation system must demonstrably exceed.

**Sections and their function:**

**"AI-generated survivorship plans and AI coaching"** — Covers two late-generation systems:

- **QOL+ (Korean AI survivorship care plan, citation [9]):** An AI service for posttreatment breast cancer survivors that collected survivor-reported outcomes via mobile, auto-generated a survivorship care plan, and supported face-to-face counseling. Patient control was measured via patient activation, CASE-cancer, patient-provider interaction, and quality of life. Pilot metrics: participation 70.7%, retention 89.5%, satisfaction 88.2%; gains in self-efficacy (p=0.040, effect size 0.31), patient activation (p=0.051, effect size 0.29), and quality of life (p=0.010, effect size 0.42). This provides the most recent single-arm pilot effect size estimates for power calculations in a new trial.

- **PATH trial AI coaching (citation [11]):** Tested MyCoach (unsupervised goal-based model via smart speaker) and SmartText (supervised goal-based model via text) in cancer survivors. Patient control was operationalized behaviorally through objectively captured daily steps. MyCoach produced +3568.9 steps/day vs. control; 61% of MyCoach person-days reached ≥10,000 steps vs. 28% in control. Companion citation [10] (random forest QoL prediction model, accuracy 0.85, F1 0.90) documents the predictive ML layer associated with the QOL+ program.

**"Baseline for future AI and robotics"** — This is the most important section for new trial protocol framing. States explicitly that any next-generation system (including physical AI/robotics) should outperform the current software-only baseline on: (a) validated agency measures (CASE-cancer, PAM, disease-management self-efficacy); (b) objective management metrics (symptom-burden reduction, adherence, behavior execution); and (c) disease-relevant endpoints (treatment continuity, preventable emergency use, longer-term survivorship quality of life). States that the current literature is "overwhelmingly about software, sensing, and decision support, not robotics," and that any robotics claim must be judged against this software baseline.

**"Open questions and limitations"** — Identifies three structural gaps in the current evidence that a new physical AI trial can directly address: (1) direct measurement of "sense of control" is still uncommon — most trials use adjacent constructs; (2) the most technically advanced systems remain pilots or survivorship-only studies; (3) algorithmic transparency is incomplete, making exact cross-study comparisons difficult. A new trial that directly measures a validated global control construct, recruits patients during active treatment, and publishes its algorithmic stack in full will address all three gaps simultaneously.

**Key citation bridges:**
- Citations [3][4][5][9][11] are shared with chunk 1 and anchor the multi-generation progression argument.
- Citations [15][16][17][18][19] are inline reference markers in the body text that correspond to BibTeX entries [2][4][5][3][5] respectively (see mapping table below).

---

### `chunk_3_references_bibtex.md`

**Role in the corpus:** Provides the complete, machine-readable reference layer for the entire review. Contains all 11 BibTeX entries and the original source URL reference list. This file is the authoritative citation source for any new paper generated from this material.

**BibTeX entry inventory and cross-reference:**

| BibTeX Key | Inline Cite Numbers | Trial / Program | Journal / Publisher | Year |
|---|---|---|---|---|
| `Kearney2009ASYMS` | [1] | ASyMS original trial | Supportive Care in Cancer | 2009 |
| `Maguire2017eSMART` | [2] | eSMART RCT protocol | BMJ Open | 2017 |
| `McCann2024ASYMSQualitative` | [3] | ASyMS qualitative post-trial | JMIR | 2024 |
| `Absolom2021eRAPID` | [4] | eRAPID phase III RCT | Journal of Clinical Oncology | 2021 |
| `Velikova2022eRAPIDProgramme` | [5] | eRAPID full programme | NIHR Journals Library | 2022 |
| `Mlakar2021PERSISTProtocol` | [6] | PERSIST protocol | BMC Medical Informatics | 2021 |
| `Arioz2025PERSISTAppliedSciences` | [7] | PERSIST outcomes | Applied Sciences | 2025 |
| `Arioz2024PERSISTPreprint` | [8] | PERSIST preprint | JMIR Preprints | 2024 |
| `Kim2025QOLPlus` | [9] | QOL+ survivorship AI | Journal of Cancer Survivorship | 2025 |
| `Choe2024QoLPrediction` | [10] | ML QoL prediction model | Supportive Care in Cancer | 2024 |
| `Hassoon2021PATH` | [11] | PATH AI coaching trial | npj Digital Medicine | 2021 |

The **Source URL Reference List** at the bottom of chunk 3 maps inline hyperlink markers ([1]–[19]) to their resolved web URLs. Note that inline markers [12]–[19] are secondary citation pointers within the body text that resolve to URLs already associated with BibTeX entries [1]–[11]; they are not separate references.

---

## Cross-File Correlation Map

### Narrative thread across all three files

The three files together follow a single argumentative arc:

```
chunk_1_foundations.md
    │
    │  Establishes:
    │  - Four-part taxonomy of patient control
    │  - Rule-based baseline (ASyMS/eSMART) → citations [1][2][3]
    │  - EHR-integrated baseline (eRAPID) → citations [4][5]
    │  - Multimodal predictive platform (PERSIST) → citations [6][7][8]
    │
    ▼
chunk_2_advanced_systems.md
    │
    │  Extends to:
    │  - AI survivorship planning (QOL+) → citation [9]
    │  - ML prediction layer → citation [10]
    │  - Autonomous coaching (PATH) → citation [11]
    │  - Synthesizes all into the explicit software-only baseline
    │  - Identifies three open gaps for new trial design
    │
    ▼
chunk_3_references_bibtex.md
    │
    │  Grounds all claims in:
    │  - 11 fully resolved BibTeX entries
    │  - 14 source URLs confirming online accessibility
    │
    ▼
    NEW PHYSICAL AI ONCOLOGY TRIAL PAPER
```

### Citation overlap between chunk 1 and chunk 2

The following citations appear in both body chunks and are therefore central load-bearing references for the entire synthesis. Any new paper must cite all of these:

| Citation | Chunk 1 role | Chunk 2 role |
|---|---|---|
| [2] (Maguire2017eSMART) | Defines eSMART protocol and patient-agency measurement intent | Part of multi-generation progression summary |
| [3] (McCann2024ASYMSQualitative) | Reports qualitative patient themes; anchors reassurance finding | Supports "faster symptom recognition" gain claim |
| [4] (Absolom2021eRAPID) | Primary eRAPID phase III RCT results | Baseline comparator for future agency measures |
| [5] (Velikova2022eRAPIDProgramme) | Self-efficacy result (p=0.0073); EQ-5D findings | Defines minimum software-only threshold |
| [6] (Mlakar2021PERSISTProtocol) | PERSIST architecture and CASE-cancer/PAM endpoints | Informs "more sophisticated AI ≠ larger gains" warning |
| [9] (Kim2025QOLPlus) | Pilot feasibility metrics and effect sizes | Most recent benchmark for new trial power calculation |
| [11] (Hassoon2021PATH) | PATH trial step-count outcomes | Anchors behavioral-control vs. attitudinal-control distinction |

### Measurement instrument recurrence across chunks

The following validated instruments are mentioned across both body chunks and should be considered candidate endpoints for the new physical AI trial:

| Instrument | Abbrev. | Appears in chunk 1 | Appears in chunk 2 | BibTeX primary source |
|---|---|---|---|---|
| Communication and Attitudinal Self-Efficacy – cancer | CASE-cancer | Yes (definition section; PERSIST) | Yes (baseline section) | [6] Mlakar2021PERSISTProtocol |
| Patient Activation Measure | PAM | Yes (definition section; PERSIST) | Yes (QOL+; baseline section) | [6] Mlakar2021PERSISTProtocol |
| FACT-G Physical Well-Being | FACT-G PWB | Yes (eRAPID primary endpoint) | Yes (baseline section) | [4] Absolom2021eRAPID |
| EQ-5D visual analogue scale | EQ-5D VAS | Yes (eRAPID secondary) | Yes (baseline section) | [5] Velikova2022eRAPIDProgramme |
| Daily step count (objective) | — | No | Yes (PATH trial) | [11] Hassoon2021PATH |

---

## Instructions for Claude Code Opus Processing

When using these three files to assist in drafting the new physical AI oncology trial paper, apply the following logic:

1. **Load all three files simultaneously.** The argumentation is distributed across files; chunk 2's synthesis statements depend on the foundational definitions in chunk 1, and all citation validation requires chunk 3.

2. **Use chunk 1 for background and related-work sections.** The four-part patient-control taxonomy and the ASyMS/eSMART and eRAPID program descriptions provide the most citable historical context. Map every cited AI system in the new paper's background to at least one of the three generational stages described in chunk 1.

3. **Use chunk 2 for hypothesis framing and comparator specification.** The "Baseline for future AI and robotics" section directly names the threshold the new physical system must exceed. The three open gaps identified in "Open questions and limitations" are the justification for the new trial's existence and should be addressed in the Introduction or Rationale section of the new paper.

4. **Use chunk 3 for all bibliography construction.** All BibTeX keys are stable and complete. Import them directly into the new paper's .bib file. Do not reconstruct references from memory; use these entries as ground truth. Note that `Velikova2022eRAPIDProgramme` is typed as `@book` (NIHR monograph), not `@article`.

5. **Preserve effect size anchors.** The following quantitative benchmarks from the source text should be cited precisely in the new paper's power-calculation or expected-effect-size discussion:
   - eRAPID self-efficacy gain: mean +0.48, 95% CI 0.13–0.83, p=0.0073 (source: [5])
   - QOL+ self-efficacy effect size: d=0.31, p=0.040 (source: [9])
   - QOL+ patient activation effect size: d=0.29, p=0.051 (source: [9])
   - QOL+ quality of life effect size: d=0.42, p=0.010 (source: [9])
   - PATH MyCoach step gain: +3568.9 steps/day vs. control; 61% person-days ≥10,000 steps (source: [11])
   - PERSIST CASE-cancer/PAM: no statistically significant change (source: [7][8])

6. **Respect the attitudinal vs. operational control distinction.** Chunk 2 explicitly distinguishes attitudinal control (self-efficacy, activation, decisional confidence) from operational/behavioral control (step count, adherence, task completion). The new physical AI trial paper should specify which type of control its primary endpoint captures and justify that choice against the existing evidence.

7. **The ceiling-effect warning is design-critical.** PERSIST showed that patients entering with already-high CASE-cancer and PAM scores produced no statistically significant instrument-level gains. The new trial should pre-specify how it will handle baseline activation stratification and whether it will recruit patients who are earlier in the treatment pathway (where floors are lower) to avoid the same ceiling effect.

8. **No BibTeX entries exist for physical/robotic AI in oncology.** This is deliberate: the source document explicitly states the current literature is about software, sensing, and decision support, not robotics. The absence of robotic-AI citations in chunk 3 is a finding, not a gap to be filled by hallucination. New references for physical AI systems should be sourced and added to the .bib file independently during the new paper's literature review phase.
