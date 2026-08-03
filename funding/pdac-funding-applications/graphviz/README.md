# graphviz - Stage 5 of the PART II schedule (4 figure specifications)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-5%20of%208-00417A.svg)](../sub-prompts/part-ii/prompt-5-graphviz.md)
[![Figures](https://img.shields.io/badge/Figures-4%20of%2020-3C7DB2.svg)](.)
[![Palette](https://img.shields.io/badge/Palette-patient--robot--advocacy-6C757D.svg)](../../supplementary/source-files)
[![Black fill](https://img.shields.io/badge/Black%20fill-none-9AA1A8.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-6C757D.svg)](../../../README.md)

Four **graphviz-type** figure specifications. Graphviz is used where the subject
is a graph proper: a dependency order, a fault tree, a citation graph, or a
record node with ruled fields. Thin black strokes and serif labels are the
notation's own idiom and are kept.

## The four figures

| Fig | File | Construct | Perspective |
|:--|:--|:--|:--|
| 6 | [`fig-06-funding-dependency-dag.gv.md`](fig-06-funding-dependency-dag.gv.md) | directed acyclic graph | Ten awards against eight activities; only two activities have one source |
| 10 | [`fig-10-stop-authority-fault-tree.gv.md`](fig-10-stop-authority-fault-tree.gv.md) | fault tree | One top event; three AND branches and the one OR branch that is procedural |
| 16 | [`fig-16-prior-work-citation-graph.gv.md`](fig-16-prior-work-citation-graph.gv.md) | citation graph | Fourteen works in two parallel lines that meet exactly twice |
| 20 | [`fig-20-verification-record-nodes.gv.md`](fig-20-verification-record-nodes.gv.md) | record nodes | Five artifacts, each field indexed to a reviewer question |

## Why graphviz for exactly these four

| Question | Why the notation is required |
|:--|:--|
| What depends on what, and what is redundant | A DAG makes a single point of failure visible as an in-degree of one |
| How does a failure reach the patient | A fault tree with real gate glyphs distinguishes AND from OR, which is the entire safety argument |
| What came from what | A citation graph shows two parallel lines and the two places they touch; a timeline would flatten that |
| What can a sceptic check | A record node with typed fields is the only vocabulary that puts a question next to the column that answers it |

## The three rules every file here follows

1. A fault tree has **exactly one top event**, gates drawn as gate glyphs
   (`\umlgateand`, `\umlgateor`, both filled from the grayscale ramp and never
   black), and **no edge crossing a gate**.
2. Record nodes are **ruled boxes with field separators**, not tables with
   borders.
3. Edge corridors are left empty on purpose. In figure 6 the corridor density is
   part of the reading; in figure 16 the empty gap between the two lines is the
   argument.

## Where a figure deliberately draws nothing

Figure 20 draws **no edges**. Figure 15 is the paper's figure about joins;
drawing edges here would import that question into a figure about what a single
artifact answers on its own.

## Files used from other directories (Rule 5)

| Source | Figures that read it |
|:--|:--|
| [`../applications/`](../applications), all ten `sec-05-budget-site.tex` | 6 |
| [`../../potential-partners/UC-San-Diego/priority-steps.md`](../../potential-partners/UC-San-Diego/priority-steps.md) | 6 |
| [`../applications/app-01-nih-pioneer-award/sections/sec-04-operation-governance.tex`](../applications/app-01-nih-pioneer-award/sections/sec-04-operation-governance.tex) | 10 |
| [`../applications/app-02-arpa-h/sections/sec-05-budget-site.tex`](../applications/app-02-arpa-h/sections/sec-05-budget-site.tex) | 6, 10 |
| [`../applications/app-04-doe-genesis-mission/sections/`](../applications/app-04-doe-genesis-mission/sections) | 10, 20 |
| [`../../supplementary/Physical AI Oncology Trial Founding Documents.md`](../../supplementary) | 16 |
| [`../../RFA-RM-27-001/`](../../RFA-RM-27-001), [`../../RFA-RM-27-001-v2/`](../../RFA-RM-27-001-v2) | 16 |
| [`../../daraxonrasib-llm-story.md`](../../daraxonrasib-llm-story.md) | 16 |
| [`../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../supplementary/source-files) | 20 |
| [`../applications/app-09-convergent-fro/sections/sec-03-evidence.tex`](../applications/app-09-convergent-fro/sections/sec-03-evidence.tex) | 20 |
| [`../../science-golden-age/chunk-06-chapter-five-a-new-golden-age.md`](../../science-golden-age/chunk-06-chapter-five-a-new-golden-age.md) | 20 |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
