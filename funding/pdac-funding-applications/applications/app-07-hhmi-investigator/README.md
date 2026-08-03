# app-07 - HHMI Investigator Program (medical oncology perspective)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Recipient](https://img.shields.io/badge/Recipient-HHMI%20Investigator-00417A.svg)](.)
[![Perspective](https://img.shields.io/badge/Perspective-Medical%20oncology-3C7DB2.svg)](.)
[![Model](https://img.shields.io/badge/Model-Person--based%2C%207%20years-6C757D.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](.)
[![Prior work DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20843290-blue.svg)](https://doi.org/10.5281/zenodo.20843290)

An eligibility inquiry to the **HHMI Investigator Program**, dated **August 3,
2026**.

## Why this recipient

Chapter II of *Science: A New Golden Age* cites HHMI-style person-based funding,
roughly $10 million over seven years, as producing high-impact publications at
nearly double the rate of comparable federally funded peers, and asks the
portfolio to "bet on people, not just projects." This application tests the one
place the argument might not reach: an investigator with no academic
appointment. The email asks the eligibility question directly and invites a
plain no.

## The ask

| Item | Value |
|:--|:--|
| Mechanism | Person-based investigator support, seven-year horizon |
| Clinical cost | $700,000 per year for the Phase 1 and its verification work |
| Programme | One pathway question, not a set of aims |
| First readout | Four years out |
| Partner site | UC San Diego Moores Cancer Center, feasibility stage only |

## Files

| File | What it is |
|:--|:--|
| `email-app-07-hhmi-investigator.txt` | Recipients, subject, body, four-line closing, attachment lists, pre-send checklist |
| `main.tex` | The inquiry. Cover variant 7, `\appperson`: the investigator block sits above the title |
| `appstyle.sty`, `references.bib` | Shared style and bibliography, self-contained copies |
| `sections/sec-01-independent-scientist.tex` | Bet on people, not just projects; the three-property table |
| `sections/sec-02-mechanism-fit.tex` | The programme as states rather than aims. **Figure 1**, plantuml-type state diagram with guards |
| `sections/sec-03-evidence.tex` | The record so far. **Figure 2**, mermaid-type chronology with author and external bands separated |
| `sections/sec-04-operation-governance.tex` | The clinical vehicle and the endpoint set by class |
| `sections/sec-05-budget-site.tex` | Seven years and the partner site. **Figure 3**, d2-type comparison grid |
| `sections/sec-06-backmatter.tex` | Scope of claims, availability, conflicts, two-column references |
| `app-07-hhmi-investigator-LaTeX.zip` | Overleaf bundle |

## Figures, and why each type was chosen

| Fig | Type | Why this platform |
|:--|:--|:--|
| 1 | plantuml-type state with guards | The subject is **what the programme does when an aim fails**. Only a guarded state machine shows that every failure transition re-enters the programme rather than leaving it |
| 2 | mermaid-type chronology | The subject is **time, with a provenance distinction**. The author's own record is set above the axis and the one independent readout below it, so a simulation can never be read as a result |
| 3 | d2-type comparison grid | The subject is a **two-column comparison on four properties**. A grid makes the decisive row visible without argument |

No diagrams-python-type or graphviz-type figure appears here.

## Files used from other directories (Rule 5)

| Source | Used in |
|:--|:--|
| [`../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md`](../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md) | §1, HHMI-style person funding and the near-double publication rate; §5, grant lead times and administrative burden |
| [`../../../science-golden-age/chunk-01-front-matter-and-summary.md`](../../../science-golden-age/chunk-01-front-matter-and-summary.md) | §1, "bet on people, not just projects" and the $200 billion framing |
| [`../../../daraxonrasib-llm-story.md`](../../../daraxonrasib-llm-story.md) | §3, the full chronology and the three stated differences from RASolute 302 |
| [`../../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../../supplementary/source-files) | §3, the QSP result, the 81.9 credibility score, the cost per simulation |
| [`../../../supplementary/Physical AI Oncology Trial Founding Documents.md`](../../../supplementary) | Cover block and §1, the fourteen deposited works |
| [`../../../RFA-RM-27-001-v2/LaTeX Source Files.zip`](../../../RFA-RM-27-001-v2) | §4, the endpoint set; §5, the budget frame |
| [`../../../potential-partners/UC-San-Diego/README.md`](../../../potential-partners/UC-San-Diego) | §5, positioning constraint |
| [`../appstyle.sty`](../appstyle.sty), [`../references.bib`](../references.bib) | Copied here so the directory compiles standalone |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
