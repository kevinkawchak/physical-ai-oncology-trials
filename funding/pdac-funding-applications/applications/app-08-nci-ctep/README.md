# app-08 - NCI Cancer Therapy Evaluation Program (medical oncology perspective)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Recipient](https://img.shields.io/badge/Recipient-NCI%20CTEP-00417A.svg)](.)
[![Perspective](https://img.shields.io/badge/Perspective-Medical%20oncology-3C7DB2.svg)](.)
[![Design](https://img.shields.io/badge/Design-Phase%201%2C%203%2B3%2C%20n%E2%89%A418-6C757D.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-4-6C757D.svg)](.)
[![Prior work DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21097442-blue.svg)](https://doi.org/10.5281/zenodo.21097442)

A concept submission inquiry to **NCI CTEP**, dated **August 3, 2026**.

## Why this recipient

Chapter I of *Science: A New Golden Age* uses cancer as its own example of
scientific triumph, and pancreatic ductal adenocarcinoma is where that claim is
least true. The annexed FY 2028 memorandum asks agencies to prioritise
foundational biological science over the broader life-sciences category. This
application is the most clinically technical of the ten, because a concept review
is a technical review.

## The design point put forward for review

A combined drug and device study in a surgical population carries a hazard a
standard escalation does not: an adverse event may be attributable to the drug,
the operation, or the advisory system, and the dose decision depends on which.
Attribution is adjudicated by a blinded reviewer before the toxicity window
closes. Only the drug branch reaches the dose decision; the other two feed the
feasibility endpoint. Every branch reaches the independent safety monitor
unfiltered.

## The ask

| Item | Value |
|:--|:--|
| Mechanism | CTEP concept review, unaffiliated investigator |
| Design | Phase 1, open-label, single-arm, 3+3, up to 18 treated participants |
| Co-primary | DLT and MTD or RP2D; 30-day device- or procedure-related SAEs |
| Correlatives | Plasma PK, tumour pathway PD, ctDNA clearance, pathologic response, costed inside the budget |
| Direct cost | $700,000 per year, $3,500,000 total |
| Partner site | UC San Diego Moores Cancer Center, feasibility stage only |

## Files

| File | What it is |
|:--|:--|
| `email-app-08-nci-ctep.txt` | Recipients, subject, body, four-line closing, attachment lists, and an instruction to state that no agreement exists with the agent's developer |
| `main.tex` | The concept. Cover variant 8, `\apprecord`: a four-field study-registration header |
| `appstyle.sty`, `references.bib` | Shared style and bibliography, self-contained copies |
| `sections/sec-01-independent-scientist.tex` | Why an unaffiliated investigator is submitting; the concept-stage gap table |
| `sections/sec-02-mechanism-fit.tex` | Escalation as it will be executed. **Figure 1**, mermaid-type decision flowchart |
| `sections/sec-03-evidence.tex` | Pharmacologic case, with each source's limitation in its own row |
| `sections/sec-04-operation-governance.tex` | Adjudication and reporting. **Figures 2 and 3**, plantuml-type adjudication state and graphviz-type reporting dependency |
| `sections/sec-05-budget-site.tex` | Schedule, budget, partner site. **Figure 4**, d2-type schedule of assessments grid |
| `sections/sec-06-backmatter.tex` | Scope of claims, availability, conflicts, two-column references |
| `app-08-nci-ctep-LaTeX.zip` | Overleaf bundle |

## Figures, and why each type was chosen

| Fig | Type | Why this platform |
|:--|:--|:--|
| 1 | mermaid-type decision flowchart | The subject is the **3+3 rule as executed**, including the return edge to the next cohort. A flowchart with decision nodes is the notation clinical reviewers already read |
| 2 | plantuml-type state with guards | The subject is **adjudication**: which branch an event takes and under what guard. Guards written on transitions are what make the rule checkable |
| 3 | graphviz-type reporting dependency | The subject is **where a fixed attribution sends the report**, with four destinations and one monitor that sees all of them |
| 4 | d2-type schedule grid | The subject is a **schedule of assessments**, which is a grid by convention. Filled and empty cells carry the pattern without a legend |

## Files used from other directories (Rule 5)

| Source | Used in |
|:--|:--|
| [`../../../science-golden-age/chunk-02-chapter-one-introduction.md`](../../../science-golden-age/chunk-02-chapter-one-introduction.md) | §1, the report's own cancer framing |
| [`../../../science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md`](../../../science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md) | §1, prioritised foundational biological sciences |
| [`../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md`](../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md) | §1, the $200 billion portfolio and the incumbency finding |
| [`../../../RFA-RM-27-001-v2/LaTeX Source Files.zip`](../../../RFA-RM-27-001-v2) | §2 and §4, endpoint and estimand wording, the retention of all treated participants in safety summaries; §5, budget |
| [`../../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../../supplementary/source-files) | §3, the QSP and digital-twin results **and their stated limitations**, which are the authors' own |
| [`../../../daraxonrasib-llm-story.md`](../../../daraxonrasib-llm-story.md) | §3, the chronology and the three stated differences from RASolute 302 |
| [`../../../potential-partners/UC-San-Diego/README.md`](../../../potential-partners/UC-San-Diego) | §5, positioning constraint |
| [`../appstyle.sty`](../appstyle.sty), [`../references.bib`](../references.bib) | Copied here so the directory compiles standalone |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
