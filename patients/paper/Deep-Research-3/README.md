# README — Chunked Source Files for Physical AI Oncology Trial Paper

**Source document:** `Deep-Research-4.docx`
**Title:** *Oncology Trial Laws for Patient Control in the AI Era*
**Chunked:** 2026-05-05
**Intended consumer:** Claude Code Opus 4.7 (1M context) — processing all files together to assist in drafting a new physical AI oncology trial paper.

---

## Overview

The source document is a legal-policy deep-research report analyzing U.S. oncology clinical-trial law through the lens of patient control, AI, and robotics. It has been split into **four markdown files** that together reproduce the complete original content word for word. No text has been abbreviated, paraphrased, or restructured; no additional headings have been inserted beyond those present in the source. The split points follow the document's own major section boundaries.

The four files are:

| File | Sections covered | Primary content type |
|------|-----------------|----------------------|
| `part1_legal_baseline.md` | Core thesis; Current legal baseline and what it misses | Policy framing, regulatory inventory, gap analysis |
| `part2_ai_robotics_legislation.md` | AI and robotics most likely to expand patient control; Ranked document types and bill summaries | Evidence synthesis, technology evaluation, legislative taxonomy |
| `part3_metrics_guardrails.md` | Metrics for a patient-control baseline; Implementation guardrails; Open questions and limitations | Measurement framework, governance principles, evidentiary caveats |
| `part4_bibtex.md` | BibTeX entries [1]–[27] | Full bibliographic records for all 27 cited sources |

---

## File-by-File Detail

### `part1_legal_baseline.md`

**Sections:** Core thesis · Current legal baseline and what it misses

**What this file establishes:**
This is the argumentative foundation of the entire document. It defines the operative meaning of "patient control" across six functional dimensions: trial discovery, enrollment comprehension, location choice, data/biospecimen governance, toxicity warnings, and AI-exclusion contestability. It then surveys what existing U.S. law already provides (Common Rule consent protections, broad-consent permissions for secondary biospecimen use, Cures Act/HIPAA data-access rights, FDA decentralized-trial guidance) and precisely identifies what those authorities fail to deliver (payment parity for decentralized components, machine-readable eligibility outputs, algorithm transparency, formal appeal rights). The participation-rate statistic (approximately 7% in modern datasets) and the note that eligibility data is largely unstructured are key motivating facts cited throughout later files.

**Key citations used:** [1] 42 USC 300gg-8, [2] CMS NCD 310.1, [3] 45 CFR 46, [4] FDA informed-consent guidance 2023, [5] FDA/OHRP key-information draft 2024, [6] OHRP broad-consent 2017, [7] FDA DCT final guidance 2024, [8] FDA OCE advancing oncology DCTs, [9] FDA AI regulatory-decision draft 2025, [10] ONC Cures Act final rule, [11] ASTP/ONC patient portals 2024, [12] ASTP/ONC hospital APIs 2024, [13] HTI-1 DSI fact sheet 2023, [14] Federal Register information-blocking disincentives 2024, [15] HHS HIPAA right-to-access 2025.

**Role in a new oncology trial paper:** Part 1 supplies the regulatory baseline against which any proposed physical-AI trial design must be evaluated. Any new paper section describing trial governance, consent architecture, or eligibility workflows should be cross-referenced against the gaps enumerated here.

---

### `part2_ai_robotics_legislation.md`

**Sections:** AI and robotics most likely to expand patient control · Ranked document types and bill summaries

**What this file establishes:**
This is the evidence core of the document. It reviews five categories of AI/robotics technology through the specific lens of patient-control impact, with quantitative performance data for each:

- **LLM-based trial matching** — TrialGPT [16], PRISM [17], TrialMatchAI [18]: recall rates, criterion-level accuracy, screening-time reduction, NDCG scores.
- **AI notification trials** — Dana-Farber randomized trial [19]: null enrollment result, lessons about pathway dependency.
- **Predictive remote monitoring** — electronic PRO symptom monitoring [20], Moffitt Bayesian-network wearable model [22]: hospitalization risk reduction, AUC trajectories.
- **Computational pathology** — Virchow foundation model [21]: pan-cancer AUC, rare-cancer AUC, biomarker task performance.
- **Robotics** — TARGET robotic bronchoscopy [26], Mayo Clinic Cancer CARE Beyond Walls home chemotherapy [27]: diagnostic yield, complication rates, feasibility data.

The second half ranks seven legislative vehicle types (federal authorizing statute, state insurance code, federal health-data rights amendment, state telehealth/delegation amendment, federal biomarker/biospecimen statute, appropriations bill, model state act) in order of near-term patient-control impact, with a summary objective for each.

**Key citations used:** [1][2][5][6][7][8][9][10][11][12][13][14][15][16][17][18][19][20][21][22][23][24][25][26][27] — this file draws on nearly the entire reference corpus.

**Role in a new oncology trial paper:** Part 2 is the technology and legislative evidence base. A new paper's Methods, Discussion, or Policy Implications sections should draw on the quantitative performance figures here. The ranked bill summaries define the legislative architecture that a new physical-AI trial would need to operate within or propose changes to.

---

### `part3_metrics_guardrails.md`

**Sections:** Metrics for a patient-control baseline · Implementation guardrails · Open questions and limitations

**What this file establishes:**
This file operationalizes the policy framework into measurable outcomes and enforceable constraints. It proposes a four-domain public patient-control dashboard:

1. **Access** — trial-offer rate, days from pathology/genomic result to trial shortlist, machine-readable eligibility report coverage, real-time data release rates, travel/work hours saved.
2. **Decision quality** — consent comprehension, decisional-conflict scores, goal-trial concordance, AI-exclusion appeal success rate, explanation/provenance coverage.
3. **Control-in-action** — hospitalization rate, urgent-care/ED use, alert response time, false-alert burden, serious adverse events in home/robotics settings.
4. **Fairness and technical quality** — declared context of use, external validation, subgroup performance, calibration monitoring, update-control disclosure, revalidation after model changes.

It then specifies three implementation guardrails: (1) no patient denied trial consideration by automated output alone; (2) capability-neutral statutory rights that follow any AI architecture; (3) mandatory annual public reporting, auditability, and sunset review. It closes with an honest limitations section identifying where the evidence base is weakest (autonomous robotics claims) and where federal-state interoperability questions remain unresolved.

**Key citations used:** [1][4][5][7][8][9][10][11][12][13][14][16][17][18][20][21][22][23][24][25][26][27]

**Role in a new oncology trial paper:** Part 3 supplies the trial design standards and outcome measurement framework. Any new paper proposing a physical AI oncology trial should use the four-domain dashboard as a checklist for primary and secondary endpoints, and should treat the three guardrails as protocol-level constraints. The limitations section directly informs the new paper's own limitations and future-work discussion.

---

### `part4_bibtex.md`

**Sections:** BibTeX entries [1]–[27]

**What this file contains:**
All 27 bibliographic references cited across parts 1–3 in standard BibTeX format. The entries cover: 9 U.S. regulatory guidance documents and federal statutes, 3 ONC/ASTP data briefs and rule summaries, 3 peer-reviewed LLM trial-matching studies (Nature Communications, npj Digital Medicine), 2 JAMA Network Open randomized/nonrandomized clinical trials, 1 Nature Medicine computational pathology study, 1 JCO Clinical Cancer Informatics Bayesian-network study, 1 CHEST robotic bronchoscopy trial, 1 NEJM Catalyst home-chemotherapy study, 2 FDA program pages and press announcements, 1 NIH funding opportunity announcement, and 1 CMS National Coverage Determination.

**BibTeX key index:**

| Key | Ref # | Subject |
|-----|-------|---------|
| `usc_300gg8_clinical_trials` | [1] | 42 USC 300gg-8 clinical trial coverage |
| `cms_ncd_3101_routine_costs` | [2] | Medicare NCD 310.1 routine costs |
| `hhs_45cfr46_protection_human_subjects` | [3] | 45 CFR 46 Common Rule |
| `fda_informed_consent_guidance_2023` | [4] | FDA informed-consent guidance 2023 |
| `fda_ohrp_key_information_draft_2024` | [5] | FDA/OHRP key-information draft 2024 |
| `ohrp_broad_consent_attachment_c_2017` | [6] | OHRP broad-consent recommendations 2017 |
| `fda_dct_guidance_2024` | [7] | FDA decentralized clinical trials final guidance 2024 |
| `fda_oce_advancing_oncology_dct_2024` | [8] | FDA OCE advancing oncology DCTs 2024 |
| `fda_ai_regulatory_decision_draft_2025` | [9] | FDA AI regulatory decision-making draft guidance 2025 |
| `onc_cures_act_final_rule_page` | [10] | ONC Cures Act final rule |
| `astp_onc_patient_portals_apps_2024` | [11] | ASTP/ONC patient portals & apps 2024 |
| `astp_onc_hospital_apis_2024` | [12] | ASTP/ONC hospital APIs 2024 |
| `hti1_dsi_fact_sheet_2023` | [13] | HTI-1 DSI fact sheet 2023 |
| `federal_register_info_blocking_disincentives_2024` | [14] | Information-blocking disincentives final rule 2024 |
| `hhs_right_to_access_2025` | [15] | HHS HIPAA right-to-access 2025 |
| `jin2024trialgpt` | [16] | TrialGPT — LLM trial matching, Nature Communications 2024 |
| `gupta2024prism` | [17] | PRISM — EHR trial matching, npj Digital Medicine 2024 |
| `abdallah2026trialmatchai` | [18] | TrialMatchAI — end-to-end matching, Nature Communications 2026 |
| `mazor2025ai_notifications_trial` | [19] | Dana-Farber AI notification RCT, JAMA Network Open 2025 |
| `rocque2025remote_symptom_monitoring` | [20] | Remote ePRO symptom monitoring, JAMA Network Open 2025 |
| `vorontsov2024virchow` | [21] | Virchow pathology foundation model, Nature Medicine 2024 |
| `gonzalez2025bayesian_urgent_care_nsclc` | [22] | Bayesian-network urgent-care prediction NSCLC, JCO CCI 2025 |
| `fda_rtct_press_2026` | [23] | FDA real-time clinical trials announcement April 2026 |
| `fda_dhts_drug_development_2026` | [24] | FDA DHTs for drug development 2026 |
| `nih_par25_170_dht_endpoints` | [25] | NIH PAR-25-170 DHT endpoints in cancer trials |
| `target2025robotic_bronchoscopy` | [26] | TARGET robotic bronchoscopy trial, CHEST 2025 |
| `dronca2026cancer_care_beyond_walls` | [27] | Cancer CARE Beyond Walls home chemotherapy, NEJM Catalyst 2026 |

**Role in a new oncology trial paper:** Part 4 is the import-ready reference list. All in-text citation numbers in parts 1–3 map directly to the numbered entries here. When drafting the new paper, these BibTeX entries can be used as-is or supplemented with additional sources. Note that refs [16]–[22] and [26]–[27] are the primary empirical evidence sources; refs [1]–[15] and [23]–[25] are regulatory and policy authorities.

---

## Cross-File Citation Map

The table below shows which reference numbers appear in each file, enabling efficient lookup when drafting sections of a new paper that span multiple themes.

| Ref # | Part 1 | Part 2 | Part 3 | BibTeX |
|-------|--------|--------|--------|--------|
| [1] | ✓ | ✓ | ✓ | ✓ |
| [2] | ✓ | ✓ | — | ✓ |
| [3] | ✓ | — | — | ✓ |
| [4] | ✓ | — | ✓ | ✓ |
| [5] | ✓ | ✓ | ✓ | ✓ |
| [6] | ✓ | ✓ | — | ✓ |
| [7] | ✓ | ✓ | ✓ | ✓ |
| [8] | ✓ | ✓ | ✓ | ✓ |
| [9] | ✓ | ✓ | ✓ | ✓ |
| [10] | ✓ | ✓ | ✓ | ✓ |
| [11] | ✓ | ✓ | ✓ | ✓ |
| [12] | ✓ | ✓ | ✓ | ✓ |
| [13] | ✓ | ✓ | ✓ | ✓ |
| [14] | ✓ | ✓ | ✓ | ✓ |
| [15] | ✓ | ✓ | — | ✓ |
| [16] | — | ✓ | ✓ | ✓ |
| [17] | ✓ | ✓ | ✓ | ✓ |
| [18] | — | ✓ | ✓ | ✓ |
| [19] | — | ✓ | ✓ | ✓ |
| [20] | — | ✓ | ✓ | ✓ |
| [21] | — | ✓ | ✓ | ✓ |
| [22] | — | ✓ | ✓ | ✓ |
| [23] | — | ✓ | ✓ | ✓ |
| [24] | ✓ | ✓ | ✓ | ✓ |
| [25] | ✓ | ✓ | ✓ | ✓ |
| [26] | — | ✓ | ✓ | ✓ |
| [27] | — | ✓ | ✓ | ✓ |

---

## Thematic Threads Across All Files

The following thematic threads run across multiple files and are important for a new oncology trial paper to address coherently:

**1. Patient control as operational definition (Parts 1, 2, 3)**
Part 1 defines it; Part 2 provides the technology evidence for each dimension of it; Part 3 proposes how to measure it. A new paper should maintain this operational definition consistently rather than using "patient control" as a rhetorical phrase.

**2. The AI-enrollment paradox (Parts 2, 3)**
Part 2 introduces the Dana-Farber null RCT result [19] showing that AI notifications alone did not improve enrollment. Part 3 elevates this into the first implementation guardrail (no automated-only decisions). A new trial design must explicitly address the full pathway from AI identification to human navigation to patient decision.

**3. Regulatory nonbinding problem (Parts 1, 2, 3)**
Part 1 establishes that FDA guidance is nonbinding [7]. Part 2's legislative taxonomy is explicitly structured as a rights-creating response to that nonbinding status. Part 3's guardrails assume statutory force. A new paper proposing a physical AI trial must address where its protocol sits in this binding/nonbinding spectrum.

**4. Decentralization as both technology and legal gap (Parts 1, 2, 3)**
Part 1 identifies decentralized participation costs as unprotected [1][2][7][8]. Part 2 provides the clinical evidence for home chemotherapy [27] and robotic diagnostics [26]. Part 3 requires serious adverse event reporting specifically for home-based/robotics-enabled services. These three files together define the full scope of what a decentralized oncology trial protocol must address.

**5. Biomarker/pathology speed as eligibility gatekeeper (Parts 1, 2, 3)**
Part 1 flags rapid release of eligibility-critical pathology and genomic data as a missing entitlement. Part 2 quantifies the Virchow model's biomarker performance [21] and links it to trial-window expansion. Part 3 includes same-day or near-real-time pathology output release as a first-domain access metric. These passages collectively define a testable hypothesis for a new trial: faster AI-based pathology review expands patient-eligible trial windows.

**6. Technology-neutrality requirement (Parts 2, 3)**
Part 2's bill summaries explicitly call for technology-neutral definitions. Part 3's second guardrail (capability neutrality) extends this to any architecture — LLM, foundation model, agentic system, or robotics platform. A new paper's regulatory framework section should use architecture-agnostic language to future-proof its governance proposals.

**7. Citation clusters by theme**
- Consent and human-subjects protection: [3][4][5][6]
- Decentralized trials and remote care: [7][8][20][24][25][27]
- AI in regulated product development: [9][13]
- Patient data access infrastructure: [10][11][12][14][15]
- LLM trial matching: [16][17][18]
- Clinical AI deployment evidence: [19][20][22]
- Computational pathology: [21]
- Robotics and physical AI: [26][27]
- FDA program and policy evolution: [23][24]

---

## Instructions for Claude Code Opus 4.7

When processing these four files together:

1. **Treat parts 1–3 as a single continuous argument.** The document was split only for context management. Reasoning that spans multiple files (e.g., the nonbinding-guidance thread, the patient-control definition) must be synthesized across files rather than treated as separate claims.

2. **Use part4_bibtex.md as the authoritative reference source.** All numeric citations in parts 1–3 correspond to the numbered BibTeX entries in part 4. Do not fabricate or infer citations; consult part 4 for full bibliographic details.

3. **The participation rate (~7%) and unstructured-notes challenge in Part 1 are motivating constraints** that should inform any new trial's enrollment design and data-pipeline architecture.

4. **The four-domain dashboard in Part 3 is proposed as a statutory requirement.** For a new trial paper, treat these domains as candidate primary/secondary/exploratory endpoints, not as arbitrary metrics.

5. **The null result from ref [19] (Dana-Farber RCT) is a critical methodological caution.** Any new trial design that incorporates AI notifications or matching should specify the complete enrollment pathway, not just the AI component, and should power the trial to detect whole-pathway effects.

6. **The seven bill types in Part 2 define the legislative context** within which a new physical AI oncology trial would operate. Framing the new paper in terms of which gaps it addresses and which bill types it would inform will strengthen its policy relevance section.

7. **The limitations section in Part 3 (Open questions and limitations) is an honest scope statement** about where the source document's evidence base is weakest. These limitations should be acknowledged and addressed in the new paper rather than treated as resolved.
