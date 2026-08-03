# app-02 - ARPA-H mission office (surgical perspective)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Recipient](https://img.shields.io/badge/Recipient-ARPA--H-00417A.svg)](.)
[![Perspective](https://img.shields.io/badge/Perspective-Surgical-3C7DB2.svg)](.)
[![Pages](https://img.shields.io/badge/Compiled-5%20pages-6C757D.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-4-6C757D.svg)](.)
[![Term](https://img.shields.io/badge/Term-36%20months%2C%203%20gates-6C757D.svg)](.)
[![Prior work DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21097442-blue.svg)](https://doi.org/10.5281/zenodo.21097442)

A proposal abstract to **ARPA-H**, written in Kevin Kawchak's name as a single
independent performer and dated **August 3, 2026**.

## Why this recipient

*Science: A New Golden Age* names the ARPA program-manager model in Chapter II
as the instrument it trusts for high-risk work, with ARPA-H and ARPA-E as the
live examples, chosen because a program manager can start a bet, run it on
milestones, and end it without a committee. This application is therefore
written as a gated program rather than as a project narrative: three gates,
each with a numeric kill criterion checkable from data the program itself
produces.

## The ask

| Item | Value |
|:--|:--|
| Mechanism | ARPA-H program, milestone-bounded |
| Term | 36 months, three gates at months 12, 21, and 33 |
| Direct cost | $700,000 per year, $2,100,000 total |
| Cost share | None requested |
| Transition | RP2D, public safety set, open verification package |
| Partner site | UC San Diego Moores Cancer Center, feasibility stage only |

## Files

| File | What it is |
|:--|:--|
| `email-app-02-arpa-h.txt` | Recipients, subject, body, four-line closing, attachment lists, pre-send checklist |
| `main.tex` | The five-page abstract. Cover variant 2, `\appledger` |
| `appstyle.sty`, `references.bib` | Shared style and bibliography, self-contained copies |
| `sections/sec-01-independent-scientist.tex` | One performer, bounded milestones; ARPA-practice table |
| `sections/sec-02-mechanism-fit.tex` | The 36-month schedule. **Figure 1**, mermaid-type gantt with gates |
| `sections/sec-03-evidence.tex` | What is already de-risked, with the four-item table |
| `sections/sec-04-operation-governance.tex` | Authority boundaries and surgical measures. **Figure 2**, plantuml-type use case |
| `sections/sec-05-budget-site.tex` | Budget layers and kill criteria. **Figures 3 and 4**, d2-type layered stack and graphviz-type fault tree |
| `sections/sec-06-backmatter.tex` | Scope of claims, availability, conflicts, two-column references |
| `app-02-arpa-h-LaTeX.zip` | Overleaf bundle |

## Figures, and why each type was chosen

| Fig | Type | Why this platform |
|:--|:--|:--|
| 1 | mermaid-type gantt | The subject is a **schedule with decision points**. A gantt is the only one of the five vocabularies that puts duration and gate position on the same axis |
| 2 | plantuml-type use case | The subject is **who is authorized to do what**. Use-case associations state authority precisely, and a struck association states its absence |
| 3 | d2-type layered stack | The subject is **money by layer**, ordered by when it is committed. D2's layer construct carries that without implying a flow |
| 4 | graphviz-type fault tree | The subject is **how the program fails**, with AND and OR structure. A fault tree is the standard notation and the gates map onto its branches |

No diagrams-python figure appears here, because this application has no
deployment-topology question. Type follows purpose, not quota.

## Files used from other directories (Rule 5)

| Source | Used in |
|:--|:--|
| [`../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md`](../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md) | §1, the $200 billion portfolio, the incumbency finding, and the ARPA program-manager model |
| [`../../../science-golden-age/chunk-01-front-matter-and-summary.md`](../../../science-golden-age/chunk-01-front-matter-and-summary.md) | §1, the individual-scientist goal |
| [`../../../RFA-RM-27-001-v2/LaTeX Source Files.zip`](../../../RFA-RM-27-001-v2) | §5 budget frame at $700,000 per year; §4 endpoint wording |
| [`../../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../../supplementary/source-files) | §3, the QSP result, the 81.9 credibility score, the 55 verification tests, the cost benchmark |
| [`../../../daraxonrasib-llm-story.md`](../../../daraxonrasib-llm-story.md) | §3, the chronology and the three stated differences from RASolute 302 |
| [`../../../supplementary/Physical AI Oncology Trial Founding Documents.md`](../../../supplementary) | §1 prior-work claim; the `.txt` attachment list |
| [`../../../potential-partners/UC-San-Diego/README.md`](../../../potential-partners/UC-San-Diego) | §5, the partnership sequence and positioning constraint |
| [`../appstyle.sty`](../appstyle.sty), [`../references.bib`](../references.bib) | Copied here so the directory compiles standalone |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
