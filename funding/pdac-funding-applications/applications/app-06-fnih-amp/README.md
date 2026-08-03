# app-06 - Foundation for the NIH, AMP (medical oncology perspective)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Recipient](https://img.shields.io/badge/Recipient-FNIH%20%2F%20AMP-00417A.svg)](.)
[![Perspective](https://img.shields.io/badge/Perspective-Medical%20oncology-3C7DB2.svg)](.)
[![Model](https://img.shields.io/badge/Model-Pre--competitive%20consortium-6C757D.svg)](.)
[![Cost share](https://img.shields.io/badge/Non--federal%20share-contributed-6C757D.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](.)
[![Prior work DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20807027-blue.svg)](https://doi.org/10.5281/zenodo.20807027)

A partnership concept to the **Foundation for the NIH**, dated **August 3,
2026**. First application of Set B, which leads with the drug and the patient
selection rather than the operation.

## Why this recipient

Chapter III of *Science: A New Golden Age* holds up FNIH and the Accelerating
Medicines Partnership as the model for public-private structure, and the annexed
FY 2028 memorandum asks agencies to prioritise non-federal cost share. This
application proposes a narrow pre-competitive question that no single party owns:
what happens pharmacologically when a RAS(ON) multi-selective inhibitor is given
around a curative-intent pancreatic resection.

## The ask

| Item | Value |
|:--|:--|
| Mechanism | FNIH-convened pre-competitive consortium, AMP model |
| Cash ask | $700,000 per year for five years, $3,500,000 total |
| Contributed share | Drug supply, site infrastructure, pathology, bioanalytical support |
| Shared output | Specimen-level perioperative pharmacodynamic dataset, openly released, owned by none of the three parties |
| Partner site | UC San Diego Moores Cancer Center, feasibility stage only |

## Files

| File | What it is |
|:--|:--|
| `email-app-06-fnih-amp.txt` | Recipients, subject, body, four-line closing, attachment lists, and a pre-send instruction not to name any drug developer as a participating party |
| `main.tex` | The concept. Cover variant 6, `\appconsortium` |
| `appstyle.sty`, `references.bib` | Shared style and bibliography, self-contained copies |
| `sections/sec-01-independent-scientist.tex` | The AMP model and its four features |
| `sections/sec-02-mechanism-fit.tex` | Who contributes what, and in what order. **Figures 1 and 2**, d2-type nested containers and mermaid-type sequence |
| `sections/sec-03-evidence.tex` | Pharmacologic rationale, the simulation and readout table, and why neither speaks to the resectable setting |
| `sections/sec-04-operation-governance.tex` | The trial as the consortium's instrument; four measurements against resection timing |
| `sections/sec-05-budget-site.tex` | Cost share and leverage. **Figure 3**, graphviz-type contribution dependency graph |
| `sections/sec-06-backmatter.tex` | Scope of claims, availability, conflicts, two-column references |
| `app-06-fnih-amp-LaTeX.zip` | Overleaf bundle |

## Figures, and why each type was chosen

| Fig | Type | Why this platform |
|:--|:--|:--|
| 1 | d2-type nested containers | The subject is **membership**: three parties, each holding only what it can supply, feeding one shared output. Containers state ownership; arrows would imply process |
| 2 | mermaid-type sequence | The subject is **order between parties**. A sequence diagram is the only vocabulary here that puts the cross-reference letter visibly before the IRB submission |
| 3 | graphviz-type dependency graph | The subject is **which contribution unblocks which measurement**, and which measurement has no redundant contributor |

No plantuml-type or diagrams-python-type figure appears here.

## Files used from other directories (Rule 5)

| Source | Used in |
|:--|:--|
| [`../../../science-golden-age/chunk-04-chapter-three-securing-dominance-in-critical-and-emerging-technologies.md`](../../../science-golden-age/chunk-04-chapter-three-securing-dominance-in-critical-and-emerging-technologies.md) | §1, FNIH and AMP as the model, the twenty validated targets, trial-cost offshoring |
| [`../../../science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md`](../../../science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md) | §1 and §5, non-federal cost share |
| [`../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md`](../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md) | §1, the $200 billion framing |
| [`../../../daraxonrasib-llm-story.md`](../../../daraxonrasib-llm-story.md) | §3, the chronology, the RASolute 302 comparison, and the three stated differences |
| [`../../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../../supplementary/source-files) | §3, QSP and digital-twin results and the 81.9 credibility score |
| [`../../../RFA-RM-27-001-v2/LaTeX Source Files.zip`](../../../RFA-RM-27-001-v2) | §4, trial parameters; §5, budget frame |
| [`../../../potential-partners/UC-San-Diego/README.md`](../../../potential-partners/UC-San-Diego) | §5, positioning constraint |
| [`../appstyle.sty`](../appstyle.sty), [`../references.bib`](../references.bib) | Copied here so the directory compiles standalone |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
