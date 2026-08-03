# app-09 - Convergent Research, FRO programme (medical oncology perspective)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Recipient](https://img.shields.io/badge/Recipient-Convergent%20Research%20FRO-00417A.svg)](.)
[![Perspective](https://img.shields.io/badge/Perspective-Medical%20oncology-3C7DB2.svg)](.)
[![Model](https://img.shields.io/badge/Model-Time--bound%2C%20dissolves%20in%20Y5-6C757D.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](.)
[![Prior work DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20619762-blue.svg)](https://doi.org/10.5281/zenodo.20619762)

An FRO concept to **Convergent Research**, dated **August 3, 2026**.

## Why this recipient

Chapter II of *Science: A New Golden Age* describes focused research
organizations as time-bound nonprofit research startups built to dissolve, and
places them in the mid-scale gap it says no existing institution fills. This
application is the only one of the ten whose organization states its own end
date, and the budget contains no year six.

## The three properties the concept asks to be tested against

1. **A real end date.** The organization closes in year five, and year five is a
   working year with its own deliverables rather than a wind-down.
2. **Releases that do not all depend on success.** Three releases; only the
   specimen dataset requires the trial to succeed. The harness lands in year two
   and is useful even if escalation stops early.
3. **Named recipients agreed in year one.** Each artifact has one recipient
   class fixed at the start, because negotiating recipients in year five is how
   a dissolved organization strands its own outputs.

## The ask

| Item | Value |
|:--|:--|
| Mechanism | Focused research organization, five-year term |
| Direct cost | $700,000 per year, $3,500,000 total |
| Cost share | None requested |
| Deliverable | Specimen-level perioperative RAS pharmacodynamics in resected human PDAC |
| Handoffs | Harness to any qualified site; dataset to a public archive; protocol and IND to a successor sponsor |
| Partner site | UC San Diego Moores Cancer Center, feasibility stage only |

## Files

| File | What it is |
|:--|:--|
| `email-app-09-convergent-fro.txt` | Recipients, subject, body, four-line closing, attachment lists, pre-send checklist |
| `main.tex` | The concept. Cover variant 9, `\apptimeline`: a five-year strip whose fill lightens toward the dissolution year |
| `appstyle.sty`, `references.bib` | Shared style and bibliography, self-contained copies |
| `sections/sec-01-independent-scientist.tex` | Built to dissolve; the four-property table |
| `sections/sec-02-mechanism-fit.tex` | The five-year arc. **Figure 1**, mermaid-type flowchart to dissolution |
| `sections/sec-03-evidence.tex` | The gap the dataset fills. **Figure 2**, d2-type deliverable containers |
| `sections/sec-04-operation-governance.tex` | What survives the organization. **Figure 3**, diagrams-python-type handoff across the dissolution boundary |
| `sections/sec-05-budget-site.tex` | Budget with no year six, and the partner site |
| `sections/sec-06-backmatter.tex` | Scope of claims, availability, conflicts, two-column references |
| `app-09-convergent-fro-LaTeX.zip` | Overleaf bundle |

## Figures, and why each type was chosen

| Fig | Type | Why this platform |
|:--|:--|:--|
| 1 | mermaid-type flowchart | The subject is a **term ending in dissolution**, with three handoff edges that must visibly leave the main line. A flowchart carries both without implying containment |
| 2 | d2-type containers | The subject is **three releases as three bundles**, one of which is conditional on success. Containers show membership; a timeline would imply the wrong dependency |
| 3 | diagrams-python-type handoff | The subject is **what crosses a boundary and to whom**. Glyph tiles in dashed clusters on either side of a dissolution line is the only reading that shows nothing is retained |

No plantuml-type or graphviz-type figure appears here.

## Files used from other directories (Rule 5)

| Source | Used in |
|:--|:--|
| [`../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md`](../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md) | §1, FROs as time-bound nonprofits built to dissolve, the mid-scale definition, and the $200 billion framing |
| [`../../../daraxonrasib-llm-story.md`](../../../daraxonrasib-llm-story.md) | §3, the chronology and the three stated differences from RASolute 302 |
| [`../../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../../supplementary/source-files) | §3, the QSP result |
| [`../../../supplementary/Physical AI Oncology Trial Founding Documents.md`](../../../supplementary) | The `.txt` attachment list |
| [`../../../RFA-RM-27-001-v2/LaTeX Source Files.zip`](../../../RFA-RM-27-001-v2) | §4, trial parameters; §5, the five-year budget frame |
| [`../../../potential-partners/UC-San-Diego/README.md`](../../../potential-partners/UC-San-Diego) | §5, positioning constraint |
| [`../appstyle.sty`](../appstyle.sty), [`../references.bib`](../references.bib) | Copied here so the directory compiles standalone |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
