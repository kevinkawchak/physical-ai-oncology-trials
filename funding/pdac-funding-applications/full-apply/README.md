# full-apply - Stage 7 of the PART II schedule (the paper, fully written)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-7%20of%208-00417A.svg)](../sub-prompts/part-ii/prompt-7-full-apply.md)
[![Figures](https://img.shields.io/badge/Figures-20%20drawn-3C7DB2.svg)](.)
[![Tables](https://img.shields.io/badge/Tables-18-6C757D.svg)](.)
[![Compiles](https://img.shields.io/badge/pdfLaTeX-0%20errors%2C%200%20overfull%2C%200%20underfull-6C757D.svg)](.)
[![Pages](https://img.shields.io/badge/Pages-33-6C757D.svg)](.)
[![Paper DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.xxxxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxxxx)

Every `[DRAFTING INSTRUCTION]` left by
[`../draft-apply`](../draft-apply) is resolved against the file it named, all
twenty figures are drawn in full, and eighteen full-width tables are populated
with author-source quantitative data. No `\draftinstr` and no `\figslot`
survives into this stage.

## The twenty figures, as drawn

| Fig | Type | Section | What was drawn |
|:--|:--|:--|:--|
| 1 | mermaid flowchart | §1 | Policy sentence to ten recipients, split by mechanism family; recipient 10 tinted because it is not a funder |
| 2 | plantuml state, guards | §1 | Three stalls, each placed directly below the state it interrupts |
| 3 | d2 grid | §2 | Ten applications on four attributes; the science column is dropped because it would be constant |
| 4 | mermaid gantt | §5 | Fourteen months with the two 2025 overlaps shaded and the external band ruled off |
| 5 | plantuml use case | §3 | Seven actors, eleven actions, four struck links |
| 6 | graphviz DAG | §2 | Ten awards against eight activities; the two single-source activities drawn heavier |
| 7 | d2 nested containers | §5 | Four evidence tiers as containment; the outer tier dashed because it holds no evidence |
| 8 | mermaid decisions | §3 | Five reviewer gates with the answering section directly above each |
| 9 | diagrams-python topology | §6 | Three stacked clusters, one full-height trust boundary, one struck path |
| 10 | graphviz fault tree | §6 | One top event, three AND branches, one OR branch with two procedural leaves |
| 11 | d2 layers | §8 | Cash beside contributed value; the contributed column deliberately unpriced |
| 12 | mermaid sequence | §6 | Ten messages across five lifelines; the absent model path struck |
| 13 | plantuml state, guards | §6 | Six advisory states; the two human-only transitions marked on the edge |
| 14 | diagrams-python by vendor | §6 | Three vendors, nine roles, two return edges making the chain a cycle |
| 15 | d2 sql tables | §6 | Six tables, five keys; `operative_step` beneath `participant` for the two-hop path |
| 16 | graphviz citation graph | §5 | Two parallel lines, two crossing edges, an empty corridor |
| 17 | mermaid gantt | §8 | Ten review clocks with the site-agreement date as a full-height rule |
| 18 | diagrams-python by layer | §9 | Five layers twice, aligned horizontally; one shared bottom layer |
| 19 | plantuml fork and join | §4 | Four lanes; bars drawn wider than the outermost lane |
| 20 | graphviz record nodes | §10 | Five artifacts, two tinted falsifying fields, no edges by design |

## Figure verification, run twice

The stage sub-prompt requires three checks on every figure, run once, fixed, and
run again from the top.

| Check | Method | Result |
|:--|:--|:--|
| a) No text box or arrow overlap | Every node placed on a stated pitch, recorded in a comment above each figure; neighbour widths compared against pitch | 20 of 20 pass |
| b) Curved-arrow looseness stated | Every `to[...]` carries an explicit `looseness` or `bend` value; none exceeds 1.1 | 20 of 20 pass |
| c) Box-to-box spacing | Minor axis at least 6mm, major axis at least 10mm | 20 of 20 pass |

Figures are equally complete throughout: the last figure in the paper carries
five record columns and twenty-five cells, and the first carries fourteen nodes
and thirteen edges. No figure is thinner because it appears late.

## Column-width method, carried from the parent work

Every table is set with `tabularx` at exactly `\textwidth`. The widest prose
column takes the residual `X`; every fixed column is
`>{\raggedright\arraybackslash}p{...}` at the width its longest realistic cell
needs plus one `\tabcolsep`. Header rows are Corporate Blue with white bold
text. Widths are tuned per table rather than divided evenly, which is why §5's
five-column results table uses 2.6, 1.5, 1.6, 1.5 and the residual, while §10's
two-column table uses 5.6 and the residual.

## Length

| Measure | Parent work | This paper | Ratio |
|:--|:--|:--|:--|
| Source characters, `sections/` plus `main.tex` | 301,310 | 111,359 | 1/2.7 |
| Prose characters, excluding TikZ bodies, tables and comments | 131,774 | 48,007 | 1/2.7 |

The target is approximately one quarter. This stage is deliberately long: the
senior-author pass in [`../final-apply`](../final-apply) tightens the prose
toward the target without removing a figure or a table, which is the correct
order of operations. Cutting content before the argument is settled would have
removed the wrong material.

## Verification at this stage

```
pdflatex main -> bibtex main -> pdflatex main -> pdflatex main
0 errors   0 overfull hboxes   0 underfull hboxes   0 undefined citations   33 pages
```

Two underfull boxes were found and fixed in the error-fix commit: a long
`\href` display string cannot break under `\RaggedRight`, so both addresses were
changed to `\url`, which is clickable under `hyperref` and honours `\UrlBreaks`.

## Files used from other directories (Rule 5)

| Source | Resolved into |
|:--|:--|
| [`../../science-golden-age/chunk-01`](../../science-golden-age/chunk-01-front-matter-and-summary.md), [`chunk-03`](../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md) | §1 in full: the four goals, the $200 billion finding, the seven-row diagnosis table |
| [`../../science-golden-age/chunk-08`](../../science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md) | §2 mechanism families, §4 Robotics mission, §8 cost-share directive |
| [`../applications/README.md`](../applications/README.md) and the ten `app-*/` | §2, §3, §4, §8: every recipient, ask, term and figure-inventory row |
| [`../mermaid/`](../mermaid), [`../plantuml/`](../plantuml), [`../d2/`](../d2), [`../diagrams-python/`](../diagrams-python), [`../graphviz/`](../graphviz) | All twenty figures, drawn from their construction notes |
| [`../../supplementary/source-files/`](../../supplementary/source-files) | §5 results and limitations table, §5 cost table, §10 falsifiers |
| [`../../daraxonrasib-llm-story.md`](../../daraxonrasib-llm-story.md) | §5 chronology and the three stated differences |
| [`../../supplementary/Physical AI Oncology Trial Founding Documents.md`](../../supplementary) | §5 provenance figure, all fourteen nodes |
| [`../../RFA-RM-27-001-v2/`](../../RFA-RM-27-001-v2) | §8 budget frame; the cover theme this paper varies from |
| [`../../tripartisan-llm-support.md`](../../tripartisan-llm-support.md) | §6 model-roles table, verbatim |
| [`../../potential-partners/`](../../potential-partners) | §7 in full, both routes and the three corrections |
| [`../prompts/prompt-apply.md`](../prompts/prompt-apply.md), [`../sub-prompts/`](../sub-prompts) | §9 sub-prompt table and the diagram-split argument |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
