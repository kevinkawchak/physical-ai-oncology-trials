# app-05 - NIH SEED, SBIR/STTR (surgical perspective)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Recipient](https://img.shields.io/badge/Recipient-NIH%20SEED%20SBIR-00417A.svg)](.)
[![Perspective](https://img.shields.io/badge/Perspective-Surgical-3C7DB2.svg)](.)
[![Phase I](https://img.shields.io/badge/Phase%20I-%249%20mo%20%2F%20%24306K-6C757D.svg)](.)
[![Phase II](https://img.shields.io/badge/Phase%20II-24%20mo%20%2F%20%241.3M-6C757D.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](.)
[![Prior work DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21018646-blue.svg)](https://doi.org/10.5281/zenodo.21018646)

An SBIR inquiry to **NIH SEED**, dated **August 3, 2026**, submitted by
ChemicalQDevice as a small business.

## Why this recipient

Chapter IV of *Science: A New Golden Age* asks for pathways "grounded in
demonstrated skill rather than academic pedigree alone" and names SBIR as the
program that "opens doors for technician-founded ventures, directing resources
toward ideas that do not always start with a dissertation." This is the only one
of the ten applications where the commercial product, rather than the science,
carries the argument.

## The ask

| Item | Value |
|:--|:--|
| Mechanism | SBIR Phase I with a measured gate into Phase II |
| Phase I | 9 months, $306,000: interlock rig, logging schema, workflow dry run |
| Phase II | 24 months, $1,300,000: IND activation, clinical conduct, escalation, data release |
| Cost share | None requested |
| Product | A site-portable governance package, not a robot and not a drug |
| Partner site | UC San Diego Moores Cancer Center, feasibility stage only |

## Files

| File | What it is |
|:--|:--|
| `email-app-05-nih-sbir-seed.txt` | Recipients, subject, body, four-line closing, attachment lists, pre-send checklist including SAM and eRA Commons registration |
| `main.tex` | The inquiry. Cover variant 5, `\apptwopanel`: technical objective beside the commercial case |
| `appstyle.sty`, `references.bib` | Shared style and bibliography, self-contained copies |
| `sections/sec-01-independent-scientist.tex` | A venture that did not start with a dissertation; SBIR criteria table |
| `sections/sec-02-mechanism-fit.tex` | Phase I, Phase II, and after. **Figure 1**, mermaid-type phased flowchart |
| `sections/sec-03-evidence.tex` | Unit economics against industry benchmarks, plus the clinical chronology |
| `sections/sec-04-operation-governance.tex` | Technical risk. **Figure 2**, graphviz-type risk dependency DAG |
| `sections/sec-05-budget-site.tex` | Budget by phase. **Figure 3**, d2-type two-phase layered stack |
| `sections/sec-06-backmatter.tex` | Scope of claims, availability, conflicts, two-column references |
| `app-05-nih-sbir-seed-LaTeX.zip` | Overleaf bundle |

## Figures, and why each type was chosen

| Fig | Type | Why this platform |
|:--|:--|:--|
| 1 | mermaid-type phased flowchart | The subject is a **phased pathway with one gate and a return branch**. Mermaid's decision node is the only vocabulary here that shows failure returning to Phase I rather than ending the venture |
| 2 | graphviz-type dependency DAG | The subject is **what blocks what**, with two kinds of blocking: three technical dependencies and one regulatory one, distinguished by line style |
| 3 | d2-type two-phase stack | The subject is **money in two containers with a gate between them**. D2's container construct keeps Phase I and Phase II from being read as one budget |

No plantuml-type or diagrams-python-type figure appears here: this application
has no authority question and no deployment-topology question.

## Files used from other directories (Rule 5)

| Source | Used in |
|:--|:--|
| [`../../../science-golden-age/chunk-05-chapter-four-science-and-technology-better-lives-of-all-americans.md`](../../../science-golden-age/chunk-05-chapter-four-science-and-technology-better-lives-of-all-americans.md) | §1, the SBIR and technician-founded-venture passage |
| [`../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md`](../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md) | §1, the $200 billion portfolio and the incumbency finding |
| [`../../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../../supplementary/source-files) | §3, all three cost and schedule benchmarks and the company's own figures |
| [`../../../daraxonrasib-llm-story.md`](../../../daraxonrasib-llm-story.md) | §3, the chronology and the three differences from RASolute 302 |
| [`../../../supplementary/Physical AI Oncology Trial Founding Documents.md`](../../../supplementary) | §1, the fourteen deposited works |
| [`../../../RFA-RM-27-001-v2/LaTeX Source Files.zip`](../../../RFA-RM-27-001-v2) | §4, trial parameters; §5, the budget frame split across two phases |
| [`../../../potential-partners/UC-San-Diego/README.md`](../../../potential-partners/UC-San-Diego) | §5, partnership sequence and positioning constraint |
| [`../appstyle.sty`](../appstyle.sty), [`../references.bib`](../references.bib) | Copied here so the directory compiles standalone |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
