# app-01 - NIH Common Fund, Director's Pioneer Award (surgical perspective)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Recipient](https://img.shields.io/badge/Recipient-NIH%20Common%20Fund%20HRHR-00417A.svg)](.)
[![Perspective](https://img.shields.io/badge/Perspective-Surgical-3C7DB2.svg)](.)
[![Pages](https://img.shields.io/badge/Compiled-5%20pages-6C757D.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](.)
[![Dated](https://img.shields.io/badge/Dated-August%203%2C%202026-6C757D.svg)](.)
[![Prior work DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21317266-blue.svg)](https://doi.org/10.5281/zenodo.21317266)

An inquiry and application draft to the **NIH Director's Pioneer Award**
(High-Risk, High-Reward Research, NIH Common Fund), written in Kevin Kawchak's
name as an independent scientist and dated **August 3, 2026**.

## Why this recipient

*Science: A New Golden Age* names this award by title in its Summary of the
Report: agencies should "bet on people, not just projects" by "scaling
long-horizon grants for the best and brightest modeled on National Institutes of
Health (NIH) Director's Pioneer Award." It is the only funding mechanism in the
report named after an existing program, which makes it the natural first
recipient for an application whose entire premise is the individual scientist.

## The ask

| Item | Value |
|:--|:--|
| Mechanism | Director's Pioneer Award, person-based, long horizon |
| Term | 5 years, fully funded in year one |
| Direct cost | $700,000 per year, $3,500,000 total |
| Cost share | None requested |
| Partner site | UC San Diego Moores Cancer Center, feasibility stage only |
| Study | Phase 1, open-label, single-arm, 3+3, up to 18 treated participants |

## Files

| File | What it is |
|:--|:--|
| `email-app-01-nih-pioneer-award.txt` | Recipients, subject, body, four-line closing, compiled and manual attachment lists, and a pre-send checklist |
| `main.tex` | The five-page attachment. Cover variant 1, `\appbanner` |
| `appstyle.sty` | Shared style, self-contained copy |
| `references.bib` | Shared bibliography, self-contained copy |
| `sections/sec-01-independent-scientist.tex` | The $200 billion realignment and the individual-scientist case, with a five-row request-and-answer table |
| `sections/sec-02-mechanism-fit.tex` | Fit to the Pioneer Award mechanism. **Figure 1**, mermaid-type flowchart |
| `sections/sec-03-evidence.tex` | Simulation and readout chronology, with the evidence table. **Figure 2**, d2-type grid |
| `sections/sec-04-operation-governance.tex` | The operation, the advisory boundary, and the trial-parameter table |
| `sections/sec-05-budget-site.tex` | Milestone and go/no-go table, Moores partnership. **Figure 3**, graphviz-type DAG |
| `sections/sec-06-backmatter.tex` | Scope of claims, availability, conflicts, two-column references |
| `app-01-nih-pioneer-award-LaTeX.zip` | Overleaf bundle of the four items above |

## Figures, and why each type was chosen

| Fig | Type | Why this platform |
|:--|:--|:--|
| 1 | mermaid-type flowchart | The subject is a **sequence in time**: what was completed self-funded, then what the five-year term buys. Mermaid is the vocabulary for order and decision |
| 2 | d2-type container grid | The subject is a **tabulation**: four evidence tiers scored on three attributes. D2's true grid keeps the tiers from being read as one class |
| 3 | graphviz-type dependency DAG | The subject is a **dependency graph**: what the award unblocks, and which node has no redundant upstream path. That is what graphviz is for |

No PlantUML or diagrams-python figure appears here, because this application has
no formal actor-authority question and no deployment-topology question. Type
follows purpose, not quota.

## Files used from other directories (Rule 5)

| Source | Used in |
|:--|:--|
| [`../../../science-golden-age/chunk-01-front-matter-and-summary.md`](../../../science-golden-age/chunk-01-front-matter-and-summary.md) | §1, the transmittal-letter quotation and the Pioneer Award sentence |
| [`../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md`](../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md) | §1, the $200 billion portfolio and the incumbency-tax finding; §2, the mid-scale definition |
| [`../../../science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md`](../../../science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md) | §5, long-duration grants fully funded in year one |
| [`../../../RFA-RM-27-001-v2/LaTeX Source Files.zip`](../../../RFA-RM-27-001-v2) | §4 trial parameters and §5 budget figures, carried unchanged |
| [`../../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../../supplementary/source-files) | §3, the QSP and digital-twin numbers and the cost-per-simulation comparison |
| [`../../../daraxonrasib-llm-story.md`](../../../daraxonrasib-llm-story.md) | §3, the chronology and the three stated differences from RASolute 302 |
| [`../../../supplementary/Physical AI Oncology Trial Founding Documents.md`](../../../supplementary) | §1 prior-work count; the manual-attachment list in the `.txt` |
| [`../../../potential-partners/UC-San-Diego/README.md`](../../../potential-partners/UC-San-Diego) | §5, the partnership sequence and the positioning constraint |
| [`../../../pdfs/`](../../../pdfs) | The local-copy path given in the `.txt` attachment list |
| [`../appstyle.sty`](../appstyle.sty), [`../references.bib`](../references.bib) | Copied here so the directory compiles standalone |

## Positioning

UC San Diego is named as the intended partner of choice and nothing more. The
advisory system is described as bounded throughout, with licensed clinicians
retaining final authority. No patient has been treated and no trial exists.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
