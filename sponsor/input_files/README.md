# README: Chunked Oncology Clinical Trial Documents for Claude Code Processing
**Prepared for:** Claude Code Opus 4.6 (1M context)
**Source documents:** `2026_sponsor_2.md` (End-to-End Sponsor Playbook) and `2026_organization_2.md` (Sponsor Organization)
**Total chunks:** 15 files (8 sponsor + 7 organization)
**Purpose:** Processing inputs for a new physical AI oncology trial paper

---

## Document Overview

### Document A: End-to-End Sponsor Playbook (`sponsor_*.md`, 8 chunks)
Covers the full what/why of oncology trial execution from a sponsor perspective — strategy, design, operations, data systems, CMC, submission, and KPIs. All sections are modality-agnostic unless explicitly noted. Uses an `[R#]` reference system (e.g., `[R4]`) for regulatory citations plus sequential in-text `[#]` citations. Both systems are fully resolved in `sponsor_08_references.md`.

### Document B: Sponsor Organization (`org_*.md`, 7 chunks)
Covers the who/how — operating model, roles/staffing, governance forums, functional responsibilities across the trial lifecycle (using an Initiate → Operationalize → Handle → Finish framework), decision gates, KPIs, and CRO split. Uses a single sequential `[#]` in-text citation system fully resolved in `org_07_cro_split_references.md`.

---

## Sponsor Chunks: Detailed Descriptions

### `sponsor_01_executive_strategy.md`
**Content:** Executive summary of 2026 oncology development context + full portfolio/indication strategy section.
**Key topics:** Integrated evidence plan; quality-by-design; Project Optimus; Bayesian methods normalization; EU CTR/CTIS in force; EU HTA Joint Clinical Assessments; asset strategy memo structure; PoS as portfolio metric; biomarker funnel math; modality mix trends (ADCs, multispecifics, cell/gene).
**Key regulatory anchors:** ICH E6(R3) [R4], Project Optimus [R1][R5], ICH E9(R1) [R29], FDA accelerated approval [R13][R14], enrichment strategies [R15], IQVIA Global Oncology Trends 2025 [R12].
**Intra-document connections:** Sets the strategic context that all subsequent sponsor chunks execute against. The "integrated evidence plan" framing recurs in chunks 3 (design), 5 (data/safety), and 7 (KPIs). The "dose optimization as a portfolio gate" theme links directly to chunk 2 (preclinical/IND) and chunk 7 (decision gates).

---

### `sponsor_02_preclinical_regulatory.md`
**Content:** Translational/preclinical package design → IND/CTA mechanics → early regulator interaction → global accelerated pathway map.
**Key topics:** ICH S9 nonclinical framework; ctDNA as translational biomarker; IND 30-day review window; EU CTIS transition (full by 31 Jan 2025); UK CTIMPs reform (April 2026); FDA/EMA/PMDA meeting programs; Project Orbis; RTOR; SAKIGAKE; EMA accelerated assessment; EMA conditional MA.
**Key regulatory anchors:** ICH S9 [R17], ctDNA guidance [R18], IND mechanics [R19], UK HRA reform [R20], FDA formal meetings [R21], EMA scientific advice [R22], PMDA consultations [R23], accelerated pathways [R24-R28].
**Intra-document connections:** Nonclinical package feeds chunk 6 (CMC/IMP). Early regulator interactions (FDA meeting types, EMA scientific advice) recur in chunk 6 (submission planning). The IND/CTA activation section pairs with chunk 4 (operations/enrollment) for site startup timing.

---

### `sponsor_03_trial_design_stats.md`
**Content:** Trial design patterns (6-row comparative table) + biomarker/CDx strategy + adaptive/Bayesian/estimand statistics.
**Key topics:** Conventional RCT vs. seamless phases vs. basket/umbrella master protocols vs. platform trials vs. single-arm+RWE; biomarker evidence hierarchy; CDx codevelopment (FDA + EU IVDR); biomarker qualification program; ICH E9(R1) estimand framework; FDA Jan 2026 Bayesian draft guidance; ICH E20 adaptive design draft.
**Key regulatory anchors:** FDA master protocol guidance [R2], enrichment strategies [R15], CDx resources [R16], EU IVDR/EMA IVDR [R10], biomarker qualification [R36], ICH E9(R1) [R29], Bayesian guidance [R6], ICH E20 [R7], FDA CID guidance [R31].
**Intra-document connections:** The 6 design patterns in the table are referenced throughout chunk 4 (enrollment challenges differ by design) and chunk 7 (KPIs differ by design — adaptive gates vs. fixed). Bayesian/estimand topics connect to chunk 7 (pivotal design lock gate).

---

### `sponsor_04_operations_enrollment.md`
**Content:** Why operations fail (SSU data) + site/CRO/vendor management + enrollment ecosystem + decentralized elements + diversity + common failure modes table + Gantt timeline diagram.
**Key topics:** SSU bottlenecks (61-120+ day timelines; NCI vs. industry gap); hybrid sponsor-CRO model; ICH E6(R3) vendor oversight obligation; contract/budget tactics; enrollment ecosystem (central lab, EHR matching, prescreening); selective decentralization; DHT/remote endpoint evidence files; diversity action plans (FDORA); 5-row failure mode mitigation table; Gantt from TPP through launch.
**Key regulatory anchors:** ICH E6(R3) [R4], decentralized elements guidance [R38], DHT guidance [R39], diversity action plans [R40], master protocol (biomarker codevelopment) [R2], WCG site challenges report [41], IQVIA FSP/RSU brief [42].
**Intra-document connections:** SSU cycle-time KPIs feed directly into chunk 7 (IND/CTA green light gate and pivotal execution control KPIs). The Gantt diagram is the visual backbone spanning chunks 2 (IND), 4 (conduct), 6 (submission), and 7 (KPI gates). Decentralized elements link to chunk 5 (electronic systems governance, eConsent).

---

### `sponsor_05_data_safety_quality.md`
**Content:** Data architecture/electronic systems governance + eConsent/ePRO + EHR/RWE/digital endpoints + safety monitoring/DSMB/PV/RBM + entity relationship diagram.
**Key topics:** Inspection-ready data supply chain (eConsent→eSource→EDC→submission); FDA electronic systems Q&A guidance; EMA computerized systems guideline; eConsent Q&A + informed consent guidance; eCOA/ePRO (COA guidance series); FDA RWD registries, EHR/claims, and RWE framework guidances; DARWIN EU; RBQM with KRIs; ICH E2F (DSUR); E2B(R3) April 2026 deadline; entity relationship diagram (Sponsor → CRO/Sites/Vendors/CMC/PV/Regulators).
**Key regulatory anchors:** FDA electronic systems guidance [R41], EMA computerized systems guideline [R42], eConsent Q&A [R43], informed consent guidance [R44], COA guidance [R45], RWD registries [R46], EHR/claims guidance [R47], RWE framework [R48], DARWIN EU [R49], RBM guidance [R50], ICH E2F [R51], E2B(R3)/AEMS [R52].
**Intra-document connections:** The data supply chain described here is the execution layer for the designs in chunk 3 (EDC builds, adaptive access controls, Bayesian interim analyses). E2B(R3) deadline directly affects chunk 4 vendor contracting. RBQM/KRIs connect to chunk 7 (pivotal execution control KPIs).

---

### `sponsor_06_cmc_submission.md`
**Content:** Modality-agnostic CMC workflow (pre-IND through lifecycle) + IMP logistics + submission readiness + eCTD v4.0 + accelerated review mechanisms + global markets table.
**Key topics:** GMP for clinical supply; EU IMP GMP (Reg. 536/2014 + EudraLex Vol. 4); ICH Q12 lifecycle CMC; IRT-driven demand forecasting; depot/cold-chain strategy; FDA standardized study data guidance + eCTD v4.0 (Sept 2024); CDISC SDTM/ADaM; Project Orbis; RTOR; FDA accelerated approval + "underway" guidance; EMA conditional MA; EU HTA JCA (from 12 Jan 2025); Health Canada NOC/c; Swissmedic fast-track; TGA provisional; NMPA conditional.
**Key regulatory anchors:** ICH Q12 [R53], EU GMP regulation [R54], EudraLex Vol. 4 [R55], FDA standardized data/eCTD [R56], Project Orbis [R27], RTOR [R28], accelerated approval [R13][R14], EMA conditional MA [R25], EU HTA JCA [R9], Health Canada [R57], Swissmedic [R58], TGA [R59], NMPA [R60].
**Intra-document connections:** CMC and IMP logistics feed the supply chain failure mode in chunk 4. eCTD v4.0 and SDTM/ADaM readiness connect to chunk 7 (database lock + CSR gate). EU HTA JCA coupling is introduced here and appears again in chunk 7 (launch + postmarketing gate).

---

### `sponsor_07_timelines_kpis.md`
**Content:** Budget benchmarks + daily delay costs + 7-gate decision table with KPIs.
**Key topics:** Tufts CSDD mean oncology protocol budgets (Phase I $14.2M, II $25.5M, III $65.8M); Tufts daily delay costs (Phase III $55,716/day); 7 decision gates from candidate selection through launch; per-gate go-criteria, KPIs, and approval significance.
**Key regulatory anchors:** Project Optimus [3], ICH E6(R3) [92], ICH E9(R1) [91], ICH E3 [93], FDA accelerated approval [94], Tufts CSDD [R61][R62].
**Intra-document connections:** This chunk synthesizes KPIs from all previous sponsor chunks — dose optimization (chunk 2), design lock (chunk 3), SSU/enrollment (chunk 4), RBQM/data integrity (chunk 5), and submission readiness (chunk 6). It is the "scorecard" layer of the playbook.

---

### `sponsor_08_references.md`
**Content:** Complete reference list in two formats — (1) plain [R#] regulatory reference URLs and (2) in-text [#] citation index mapping numbers to URLs.
**Key topics:** Full URL set for all 62 [R#] references + full mapping of all [#] in-text citations across chunks 1–7.
**Processing note:** This chunk is a lookup table only — no narrative content. When Claude Code encounters `[R#]` or `[#]` in any sponsor chunk, resolve here.

---

## Organization Chunks: Detailed Descriptions

### `org_01_executive_regulatory.md`
**Content:** Executive summary + all regulatory/standards drivers (backbone GCP, methods, RWD/RWE, data standards, EU CTR/CTIS).
**Key topics:** Sponsor accountability under 21 CFR 312.50/312.52; ICH E6(R3) modernized oversight; quality-by-design with QTLs; decentralized elements; RBM; electronic systems governance; RWD/RWE; Project Optimus; EU CTR archiving (25 years, sponsor-appointed archive persons); CTIS results timelines.
**Key regulatory anchors:** 21 CFR 312.50 [1], ICH E6(R3) [2][3][10], decentralized elements [4], RBM guidance [5], Part 11/electronic systems [6][13], RWD/RWE [7], Project Optimus [8], EU CTR archiving [9][21][22], adaptive designs [15], master protocols [16], ICH M11 [17], FDA Study Data Technical Conformance Guide [20].
**Intra-document connections:** Sets the regulatory "spine" referenced throughout all org chunks. QTL/pre-specified acceptable ranges theme recurs in org_06. EU CTR archiving obligation recurs in org_05 (closeout) and org_07 (CRO split table).

---

### `org_02_operating_model_roles.md`
**Content:** Two-layer sponsor structure (program/asset + study/trial) + full staffing table (10 functions, mid vs. large pharma scale ranges) + governance forums.
**Key topics:** Asset/program layer vs. study/trial layer; RACI; 10 functional areas with titles, mid-pharma scale (~2–6), large-pharma scale (~4–12); governance bodies (Portfolio Committee, SSC/STG, DMC/IDMC, Safety Management Team, Dose Escalation Committee, Quality/Risk Review Board, Vendor Governance); ICH E6(R3) committee documentation; QTL governance.
**Key regulatory anchors:** ICH E6(R3) [34][38], 21 CFR 312.50 [23][44], DMC guidance [35], ICH E2A [36], Project Optimus staffing [24][37], RBM [25], safety reporting [26][43], CDISC/data standards [27], adaptive/biostatistics [28], CTIS/regulatory [29], QA/electronic systems [30], DCT supply [31], disclosure/CTIS [32], RWD/RWE [33].
**Intra-document connections:** The 10-function staffing table is the organizational backbone for org_03 through org_05. Governance forums recur as the decision infrastructure for all "Handle issues" scenarios in org_03–05. The DMC governance recurs in org_05 (safety). QTL/Risk Review Board connects to org_06 (KPIs).

---

### `org_03_functions_discovery_protocol.md`
**Content:** Functional lifecycle descriptions for (1) Discovery/IND-enabling and (2) Protocol design/scientific leadership — each using Initiate → Operationalize/run → Handle issues → Finish structure.
**Key topics:** IND-enabling core team RACI (tox, DMPK, bioanalytical, CMC, regulatory writing); IB as essential record; unexpected toxicology/CMC signals; IND-ready gate deliverables; Project Optimus pulling dose strategy earlier; ICH M11 structured protocol industrialization; adaptive design governance; master protocol infrastructure; protocol amendment portfolio risk; enrollment feasibility/burden reduction; estimands/QTLs from inception.
**Key regulatory anchors:** ICH M3(R2)/ICH S9 [41], 21 CFR 312.23 [42], Project Optimus [45], ICH M11 [17], FDA adaptive designs [46], FDA master protocols [46], RBM/electronic systems for amendments [47], FDA enrollment guidance [48], adaptive integrity controls [49], ICH E6(R3) QTLs [50].
**Intra-document connections:** IND-enabling output (IB, nonclinical summaries, safety monitoring plan) is the "start" input for org_04 (site selection). Protocol amendment governance recurs in org_04 (startup) and org_06 (decision gates). Estimand lock-in connects to org_06 (pivotal design gate).

---

### `org_04_functions_site_enrollment.md`
**Content:** Functional lifecycle for (1) Site selection/country strategy/study startup and (2) Enrollment/conduct/monitoring/data QC — both using Initiate → Operationalize → Handle → Finish structure.
**Key topics:** RWD-driven feasibility (claims/EHR/registry); CRO-led but sponsor-governed startup; EU CTIS structured submission; startup delays (cancer center activation gap vs. NCI); diversity via site geography/decentralized; system readiness/Part 11; hybrid conduct model (DCT + RBM + centralized analytics); ICH E6(R3) centralized monitoring; data scientists as monitoring roles; centralized monitoring for QTL detection; supply/logistics disruptions; inspection readiness artifacts; LPLV-to-lock cycle times.
**Key regulatory anchors:** FDA RWD guidance [7], CTIS [51], startup delay empirical data [52], Part 11/electronic systems [53], FDA decentralized elements [4][134], ICH E6(R3) centralized monitoring [54], RBM [5], FDA electronic systems [55], enrollment mitigation [56], QTL/KRI/CAPA [50], IMP disposition [57], ICH E6(R3) essential records [58], cycle time benchmarks [59].
**Intra-document connections:** Startup KPIs (FPI gate) connect to org_06 decision gates. Centralized monitoring/RBQM directly feeds org_06 KPIs layer. IMP disposition/logistics issues link to org_07 (CRO split — clinical supply ownership).

---

### `org_05_functions_safety_closeout.md`
**Content:** Functional lifecycle for (1) Safety/PV/signal management and (2) Medical affairs/disclosure/closeout/archiving — using Initiate → Operationalize → Handle → Finish structure.
**Key topics:** Safety strategy, SMT/DMC governance; expedited reporting (21 CFR 312.32, ICH E2A, ICH E2B(R3)); safety signal escalation chain; DMC/HA interactions; decentralized safety monitoring; DHT validation/data integrity; CSR safety modules; post-approval PV planning; ClinicalTrials.gov (FDAAA 801/42 CFR Part 11, 1-year primary completion deadline); CTIS results/lay summaries (1 year from completion; 6 months pediatric; CSR 30 days post-MA); disclosure governance; post-completion inspection readiness; EU CTR 25-year archive obligation; sponsor archive role assignment.
**Key regulatory anchors:** FDA DMC guidance [35][60], ICH E6(R3) [63], 21 CFR 312.32 [61][62], ICH E2A [36], ICH E2B(R3) [61], decentralized elements [64], ICH E6(R3) essential records [71], ClinicalTrials.gov/FDAAA [70][79], CTIS results [22][32], EU CTR archiving [9].
**Intra-document connections:** Safety signal escalation connects to org_02 governance forums (SMT, DMC). PV planning post-approval feeds org_06 (launch/postmarketing gate). EU CTR archive obligations recur in org_07 (CRO split — archiving row) and org_06 (archiving complete gate).

---

### `org_06_gates_timelines_kpis.md`
**Content:** Decision gates across the lifecycle (8-gate list) + timeline context (median development times) + KPIs in 3 linked layers + Mermaid flowchart of sponsor lifecycle milestones.
**Key topics:** 8 lifecycle gates (candidate → IND-ready → FPI readiness → dose optimization → interim/adaptation → database lock → CSR/submission → archiving); median clinical development ~8.3 years (innovative drugs 2010–2020); oncology ~6.7 years (EMA 2010–2019); ClinicalTrials.gov 1-year results deadline; CTIS timelines; EU CTR 25-year archive; 3-layer KPIs (probability-of-approval, execution, inspection/data integrity); Mermaid flowchart from Discovery through Archiving with safety/enrollment/data quality branch loops.
**Key regulatory anchors:** ICH E6(R3) QTLs/KRIs [80][82], Project Optimus [81], RBM guidance [82], 21 CFR 312.23 [72], FDA adaptive design/FPI [47][46], CDISC/data standards [73], ICH E3 CSR [74], EU CTR archiving [9], ClinicalTrials.gov [79], CTIS [22], development time data [75][76][78], startup delay empirical [52].
**Intra-document connections:** This chunk is the synthesis layer for the entire org document. The 8 gates correspond to the functional phases in org_03–05. The 3-layer KPI framework pulls from every functional area. The Mermaid flowchart is a visual representation of the entire org_03–05 lifecycle narrative.

---

### `org_07_cro_split_references.md`
**Content:** Sponsor-owned vs. CRO-managed activities table (8 lifecycle areas) + complete reference list (plain URLs + citation index).
**Key topics:** 8-row sponsor/CRO/must-have split table covering Discovery/IND-enabling, Protocol/design, Regulatory interactions, Site selection/startup, Enrollment/retention, Monitoring, Safety/PV, Data management/biostats, QA, Closeout/archiving, Disclosure; sponsor governance "must-haves" per area; EU CTR archive obligation in closeout row.
**Key regulatory anchors:** 21 CFR 312.52/ICH E6(R3) [83], IB essential records [84], ICH E6(R3) QTLs [85], CTIS [29], electronic systems [86], decentralized oversight [87], centralized monitoring [88], DMC/data integrity [89], CDISC [90], EU CTR archiving [91], ClinicalTrials.gov/CTIS disclosure [92]; plus full plain-URL reference list and numeric citation index.
**Processing note:** The reference list in this chunk resolves all `[#]` citations in org_01 through org_06. The CRO split table summarizes who owns what across the full lifecycle and is the primary cross-document bridge to sponsor_04 (operations) and sponsor_05 (data/safety).

---

## Cross-Document Structural Alignment

The two documents are complementary: the Sponsor Playbook answers **what to do and why**; the Organization document answers **who does it and how it is governed**. They share the same lifecycle phases and regulatory anchors.

| Lifecycle Phase | Sponsor Playbook Chunk | Organization Chunk |
|---|---|---|
| Strategy / portfolio / TPP | sponsor_01 | org_01 (regulatory drivers), org_02 (program layer) |
| Preclinical / IND-enabling | sponsor_02 | org_03 (Discovery/IND section) |
| Protocol design / statistics | sponsor_03 | org_03 (Protocol design section) |
| Site selection / startup | sponsor_04 (operations) | org_04 (Site selection section) |
| Enrollment / conduct / monitoring | sponsor_04 (operations) | org_04 (Enrollment section) |
| Safety / PV | sponsor_05 (safety section) | org_05 (Safety section) |
| Data / electronic systems | sponsor_05 (data section) | org_04 (data QC), org_07 (data mgmt row) |
| CMC / supply chain | sponsor_06 | org_07 (CRO split — clinical supply row) |
| Submission / market entry | sponsor_06 | org_07 (regulatory/disclosure rows) |
| Decision gates / KPIs | sponsor_07 | org_06 |
| References | sponsor_08 | org_07 (second half) |

---

## Shared Regulatory Anchors Across Both Documents

The following regulatory documents are cited in both source files and are foundational across nearly all chunks:

- **ICH E6(R3)** — cited in 12 of 15 chunks; governs GCP, centralized monitoring, QTLs, essential records, and vendor oversight
- **FDA decentralized elements guidance** — cited in sponsor_04/05 and org_01/04
- **FDA RBM guidance** — cited in sponsor_04/05 and org_01/04/06
- **21 CFR 312.50/.52** — cited in org_01/02/07 and sponsor (via ICH E6(R3) references)
- **FDA electronic systems guidance + 21 CFR Part 11** — cited in sponsor_05 and org_01/04/07
- **EU CTR / CTIS** — cited in sponsor_02/06/07 and org_01/05/06/07
- **Project Optimus** — cited in sponsor_01/02/07 and org_01/02/06
- **FDA master protocol guidance** — cited in sponsor_03/04 and org_01/03
- **ICH E9(R1) estimands** — cited in sponsor_03/07 and org_03/06
- **CDISC/FDA Study Data Conformance Guide** — cited in sponsor_06/07 and org_01/07

---

## Diagrams and Structured Tables Inventory

| Diagram/Table | Chunk | Type | Description |
|---|---|---|---|
| Major oncology trial designs | sponsor_03 | 6-row comparative table | Design pattern vs. use case vs. gains vs. risks vs. regulator must-haves |
| Operational failure modes | sponsor_04 | 5-row mitigation table | Failure mode vs. root causes vs. sponsor mitigation |
| Sponsor Gantt timeline | sponsor_04 | Mermaid Gantt | TPP → launch, 4 sections, milestones at FPI and LPLV |
| Sponsor governance ERD | sponsor_05 | Mermaid graph | Sponsor → CRO/Sites/Vendors/CMC/PV/Regulators → downstream |
| Sponsor decision gates + KPIs | sponsor_07 | 7-row table | Gate vs. go-criteria vs. KPIs vs. approval significance |
| Staffing scale table | org_02 | 10-row table | Function vs. titles vs. mid-pharma scale vs. large-pharma scale vs. notes |
| Governance forums list | org_02 | Prose list | 7 governance bodies with charters and scope |
| Sponsor lifecycle flowchart | org_06 | Mermaid flowchart | Discovery through archiving with 3 issue-branch loops |
| Sponsor/CRO split table | org_07 | 11-row table | Lifecycle area vs. sponsor-owned vs. CRO-managed vs. governance must-haves |

---

## Processing Notes for Claude Code

1. **Independent reference systems.** The sponsor file uses `[R#]` + `[#]` dual citations; the org file uses `[#]` only. Do not conflate the two systems. Resolve sponsor references in `sponsor_08_references.md`; resolve org references in `org_07_cro_split_references.md`.

2. **The Initiate → Operationalize → Handle → Finish pattern** in org_03–05 is an intentional AI-integration framework. Each "Handle issues" subsection represents a failure mode that is a candidate for AI-assisted detection, prediction, or intervention in the new paper.

3. **QTL/KRI/RBQM is a recurring cross-document theme.** ICH E6(R3) QTLs are mentioned in sponsor_01 (quality-by-design), sponsor_05 (RBQM), sponsor_07 (KPIs), org_01 (regulatory drivers), org_02 (Quality/Risk Review Board), org_04 (centralized monitoring), and org_06 (KPIs). This is a strong candidate for a dedicated AI paper section.

4. **The Bayesian methods normalization** (FDA Jan 2026 draft guidance) is exclusive to sponsor_03 and sponsor_07. It is not mirrored in the org document but has direct organizational implications (simulation capacity, operating characteristics documentation) relevant to org_02 (biostatistics function).

5. **EU dual obligations** (CTIS transparency + 25-year archive) appear in both documents but in different contexts: the org document emphasizes the organizational role-assignment and access-control obligations (org_01, org_05, org_06, org_07); the sponsor document emphasizes the timeline and submission-readiness implications (sponsor_02, sponsor_06, sponsor_07).

6. **The E2B(R3) April 1, 2026 deadline** is a time-sensitive operational inflection point noted in sponsor_05. This is a 2026-specific regulatory event with direct system and vendor contract implications — high relevance for any paper section describing the 2026 operating environment.

7. **Budget/cost benchmarks** are confined to sponsor_07 (Tufts CSDD). The org document has no cost data. These numbers are directional baselines from 2022 data (inflation-adjusted in the delay-cost figure).

8. **Cross-document pairing guidance for new paper sections.** When drafting any new section, load both the sponsor chunk (what/why) and the corresponding org chunk (who/how) as listed in the Structural Alignment table above.
