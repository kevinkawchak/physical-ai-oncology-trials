# app-04 - DOE Office of Science, Genesis Mission (surgical perspective)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Recipient](https://img.shields.io/badge/Recipient-DOE%20Genesis%20Mission-00417A.svg)](.)
[![Mission](https://img.shields.io/badge/National%20mission-Robotics-3C7DB2.svg)](.)
[![Authority](https://img.shields.io/badge/Authority-EO%2014363-6C757D.svg)](https://www.federalregister.gov/documents/2025/11/28/2025-21665/launching-the-genesis-mission)
[![Pages](https://img.shields.io/badge/Compiled-%E2%89%A45%20pages-6C757D.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-4-6C757D.svg)](.)
[![Prior work DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21097442-blue.svg)](https://doi.org/10.5281/zenodo.21097442)

A white paper to the **DOE Office of Science, Genesis Mission**, dated
**August 3, 2026**.

## Why this recipient

The FY 2028 budget priorities memorandum annexed to *Science: A New Golden Age*
names six national missions. The Robotics mission is written as "general-purpose
autonomous systems capable of dexterous manipulation, mobility, and reliable
operation in real-world environments, to initiate the era of physical AI-driven
scientific discovery and American reindustrialization." This application argues
that a human operating room is the hardest available instance of that sentence,
and offers three Mission-facing contributions: a trust-boundary topology, a
closed loop with one measured human gate, and open data formats.

## The ask

| Item | Value |
|:--|:--|
| Mechanism | Genesis Mission, Robotics national mission testbed |
| Authority cited | Executive Order 14363; Executive Order 14303 (Gold Standard Science) |
| Term | 5 years |
| Direct cost | $700,000 per year, $3,500,000 total |
| Mission artifacts | Open advisory logging schema, open stop-latency method, escalation dataset with negative results |
| Partner site | UC San Diego Moores Cancer Center, feasibility stage only |

## Files

| File | What it is |
|:--|:--|
| `email-app-04-doe-genesis-mission.txt` | Recipients, subject, body, four-line closing, attachment lists, pre-send checklist |
| `main.tex` | The white paper. Cover variant 4, `\appmissiontile` |
| `appstyle.sty`, `references.bib` | Shared style and bibliography, self-contained copies |
| `sections/sec-01-independent-scientist.tex` | The Robotics mission and the annex directives table |
| `sections/sec-02-mechanism-fit.tex` | What runs where. **Figure 1**, diagrams-python-type clustered topology |
| `sections/sec-03-evidence.tex` | The closed loop and the quantitative record. **Figure 2**, mermaid-type state diagram |
| `sections/sec-04-operation-governance.tex` | Gold Standard Science, artifact by artifact. **Figure 3**, graphviz-type record nodes |
| `sections/sec-05-budget-site.tex` | Mission alignment, budget, partner site. **Figure 4**, d2-type alignment grid |
| `sections/sec-06-backmatter.tex` | Scope of claims, availability, conflicts, two-column references |
| `app-04-doe-genesis-mission-LaTeX.zip` | Overleaf bundle |

## Figures, and why each type was chosen

| Fig | Type | Why this platform |
|:--|:--|:--|
| 1 | diagrams-python-type topology | The subject is **what runs where, across a trust boundary**. Glyph tiles inside dashed clusters are the only vocabulary here that reads as deployed infrastructure |
| 2 | mermaid-type state diagram | The subject is a **loop with a guarded transition**. States and a decision node carry both branches of the human gate |
| 3 | graphviz-type record nodes | The subject is **data structure**: three record types, field by field, each field indexed to the principle it discharges |
| 4 | d2-type alignment grid | The subject is a **mapping**: four Mission challenges against what is supplied and when. A grid states it without implying a sequence |

## Files used from other directories (Rule 5)

| Source | Used in |
|:--|:--|
| [`../../../science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md`](../../../science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md) | §1, the Robotics mission wording and the three annex directives; §5, the four Mission challenges |
| [`../../../science-golden-age/chunk-06-chapter-five-a-new-golden-age.md`](../../../science-golden-age/chunk-06-chapter-five-a-new-golden-age.md) | §3, closed-loop autonomous experimentation; §4, Gold Standard Science and EO 14303 |
| [`../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md`](../../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md) | §1, the $200 billion realignment |
| [`../../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../../supplementary/source-files) | §3, the QSP result and the 81.9 credibility score over 55 tests |
| [`../../../daraxonrasib-llm-story.md`](../../../daraxonrasib-llm-story.md) | §3, the chronology and the three differences from RASolute 302 |
| [`../../../RFA-RM-27-001-v2/LaTeX Source Files.zip`](../../../RFA-RM-27-001-v2) | §5, trial parameters and budget frame |
| [`../../../potential-partners/UC-San-Diego/README.md`](../../../potential-partners/UC-San-Diego) | §5, positioning constraint |
| [`../appstyle.sty`](../appstyle.sty), [`../references.bib`](../references.bib) | Copied here so the directory compiles standalone |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
