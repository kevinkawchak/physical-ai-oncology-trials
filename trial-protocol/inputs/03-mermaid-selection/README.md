# 03-mermaid-selection - Stage 3: Mermaid and ASCII selection (v5.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-3%20of%207-blue.svg)](.)
[![Diagrams](https://img.shields.io/badge/Diagrams-6%20Mermaid%20%2B%206%20ASCII-lightgrey.svg)](.)

Stage 3 of Process B. Running
[`../sub-prompts/prompt-3-mermaid-selection.md`](../sub-prompts/prompt-3-mermaid-selection.md)
produced [`output-3-mermaid-selection.md`](output-3-mermaid-selection.md): the
curated visual set for H. R. 9510 v5.0, in two halves, because Bill v5.0
restores the ASCII medium beside gray-scale Mermaid and tables. **Half one** is
six gray-scale Mermaid diagrams, each a different family (flowchart LR, BPMN
flowchart, sequence, clustered component, C4 view, hybrid flowchart TB), every
one re-themed to the financial-data subject. **Half two** is six ASCII forms
(cover overview, side-by-side cost ledger, statutory layering with the
financial sections, comparative-print order, money flow, fiscal waterfall),
each set in the centered white-background `asciifig` frame. The slot map binds
every diagram to its bill figure slot and states why that medium is correct.

## Files

| File | Role |
|:--|:--|
| [`output-3-mermaid-selection.md`](output-3-mermaid-selection.md) | The Stage 3 narrative output (slot map + 6 Mermaid sources + 6 ASCII forms) |
| `README.md` | This index |

## Sources used from other directories (Rule 6)

| No. | Asset used | Upstream source |
|:--|:--|:--|
| 01 | The selection sub-prompt run here | `auto-bill-02/sub-prompts/prompt-3-mermaid-selection.md` |
| 02 | Mermaid diagram families (workflow, PlantUML component/sequence, D2, Graphviz/DOT clustered, Structurizr/C4, BPMN, draw.io swimlane, hybrid) | `Clinical-AI-Demos/tree/main/ai-outputs/output-01` |
| 03 | Gray-scale palette and TikZ reproduction convention | `single-prompt-bill/auto-bill-01/03-mermaid-selection/output-3-mermaid-selection.md` |
| 04 | ASCII visual catalog (lineage boxes, layering, timelines, ledgers, trees) | `cancer-automated/.../papers/VVUQ-05/update-bill/figures-bill` |
| 05 | Rendered ASCII conventions being re-themed | `cancer-automated/.../papers/VVUQ-05/final-bill/sections` (Cover, Fig. 2, 4, 5) |
| 06 | Financial subjects each diagram carries | `auto-bill-02/01-research/output-1-research.md` |

## Where this stage flows next

Stage 4 numbers these twelve visuals into the figure plan and adds the table
plan; the bill stages render the Mermaid half as gray-scale TikZ and the ASCII
half as `asciifig` frames.

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
