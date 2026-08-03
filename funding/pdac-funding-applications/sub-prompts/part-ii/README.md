# sub-prompts/part-ii - the PART II schedule (summary paper, v4.4.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stages](https://img.shields.io/badge/Stages-8-00417A.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-20-3C7DB2.svg)](.)
[![Paper DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.xxxxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxxxx)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-blue.svg)](../../../../README.md)

The eight sub-prompts that build PART II: one summary paper, one DOI, titled
*10 Funding Applications: A Phase 1, First-in-Human, PDAC Clinical Trial
Protocol of a LLM-Directed Robotic Whipple with Daraxonrasib (RMC-6236)*, at
approximately one quarter of the `patient-robot-advocacy` character count.

## The eight stages

| # | File | Output | Figures | Commit floor |
|:--|:--|:--|:--|:--|
| 1 | [`prompt-1-mermaid.md`](prompt-1-mermaid.md) | [`../../mermaid/`](../../mermaid) | 6 | 7 |
| 2 | [`prompt-2-plantuml.md`](prompt-2-plantuml.md) | [`../../plantuml/`](../../plantuml) | 3 | 4 |
| 3 | [`prompt-3-d2.md`](prompt-3-d2.md) | [`../../d2/`](../../d2) | 4 | 5 |
| 4 | [`prompt-4-diagrams-python.md`](prompt-4-diagrams-python.md) | [`../../diagrams-python/`](../../diagrams-python) | 3 | 4 |
| 5 | [`prompt-5-graphviz.md`](prompt-5-graphviz.md) | [`../../graphviz/`](../../graphviz) | 4 | 5 |
| 6 | [`prompt-6-draft-apply.md`](prompt-6-draft-apply.md) | [`../../draft-apply/`](../../draft-apply) | 20 placed | 10+ |
| 7 | [`prompt-7-full-apply.md`](prompt-7-full-apply.md) | [`../../full-apply/`](../../full-apply) | 20 drawn | 10+ |
| 8 | [`prompt-8-final-apply.md`](prompt-8-final-apply.md) | [`../../final-apply/`](../../final-apply) | 20 polished | 10+ |

## Why the type split is uneven

The master prompt requires the diagram type to follow the purpose, not a quota.

| Type | Count | The question it answers |
|:--|:--|:--|
| Mermaid | 6 | What happens, in what order, and what decides |
| PlantUML | 3 | Who is permitted to do what, under which guard |
| D2 | 4 | What contains what, and what tabulates against what |
| Diagrams (Python) | 3 | What runs where, across which trust boundary |
| Graphviz | 4 | What depends on what, and how a failure propagates |

## The twenty figures, in paper order

| Fig | Type | Subject |
|:--|:--|:--|
| 1 | mermaid | Policy paragraph to ten addressed applications |
| 2 | mermaid | Independent-scientist proposal states and the incumbency tax |
| 3 | d2 | The ten applications as a scored grid |
| 4 | mermaid | Daraxonrasib chronology, June 2025 to August 2026 |
| 5 | plantuml | Actor authority: who may do what |
| 6 | graphviz | Award dependency DAG |
| 7 | d2 | Evidence tiers as nested containers |
| 8 | mermaid | Reviewer go / no-go gates |
| 9 | diagrams-python | On-premises topology and the trust boundary |
| 10 | graphviz | Stop-authority fault tree |
| 11 | d2 | Budget layers and the cost-share separation |
| 12 | mermaid | Perioperative sequence across the operative day |
| 13 | plantuml | Advisory state guards |
| 14 | diagrams-python | Three frontier-model roles and their artifacts |
| 15 | d2 | Trial data record schema |
| 16 | graphviz | Prior-work citation graph |
| 17 | mermaid | Ten-submission schedule |
| 18 | diagrams-python | One person's stack against a program office |
| 19 | plantuml | Award lifecycle activity, forks and joins |
| 20 | graphviz | Verification record nodes |

## Files used from other directories (Rule 5)

| Source | Stage that reads it |
|:--|:--|
| [`../../../supplementary/source-files/patient-robot-advocacy.zip`](../../../supplementary/source-files) | 1 to 5 (palette and the five TikZ vocabularies), 7 (table column method), 8 (float and spacing method) |
| [`../../../RFA-RM-27-001-v2/`](../../../RFA-RM-27-001-v2) | 6 (cover must vary from this theme), 7 (trial numbers) |
| [`../../../science-golden-age/`](../../../science-golden-age) | 6 and 7 (§1 and §2 content, bib keys) |
| [`../../applications/`](../../applications) | 6 and 7 (§2, §3, §4: the ten file sets the paper summarizes) |
| [`../../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../../supplementary/source-files) | 7 (§5 quantitative tables) |
| [`../../../../trial-ind/`](../../../../trial-ind) | 6 to 8 (stage layout, zip convention, commit discipline) |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
