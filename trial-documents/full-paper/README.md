# trial-documents/full-paper - Stage 3 full paper

[![Stage](https://img.shields.io/badge/Stage-3%20Full-2F5D7C.svg)](.)
[![Sections](https://img.shields.io/badge/Sections-8%20.tex-8B2E3F.svg)](sections)
[![Figures](https://img.shields.io/badge/TikZ%20figures-24-D08770.svg)](sections)
[![Tables](https://img.shields.io/badge/Tables-6-BFD7EA.svg)](sections)
[![Repository](https://img.shields.io/badge/Repository-v4.2.0-2F5D7C.svg)](../../README.md)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)

Stage 3 of the mermaid -> draft -> full -> final build for the paper *Phase 1
Pancreatic Cancer Trial Efficient LLM Document Generations* (paper v1.0, repository
v4.2.0). The full stage resolves every draft `[DRAFTING INSTRUCTION]` into full
prose, reproduces all 24 Mermaid figures as TikZ `mermaidfig` figures, and fills six
full-width tables.

## Files

| File | Role |
|:--|:--|
| [`main.tex`](main.tex) | Cover, front matter, ToC after Introduction, `\input` of all sections |
| [`paperstyle.sty`](paperstyle.sty) | Shared style; the `mermaidfig` environment uses `adjustbox` max-width so no figure overflows the right margin |
| [`references.bib`](references.bib) | Bibliography (clickable DOI and DOI URL per entry) |
| [`prompt-full-paper.md`](prompt-full-paper.md) | The Stage 3 sub-prompt executed here |
| [`output-full-paper.md`](output-full-paper.md) | The Stage 3 narrative output |
| [`sections/`](sections) | Eight full section `.tex` files (Rule 6) |
| `full-paper-LaTeX.zip` | The Overleaf bundle |

## Figures and tables

The 24 TikZ figures are distributed as Figures 1-3 (Introduction), Figures 4-15
(Methods), Figures 16-22 (Results), and Figures 23-24 (Discussion); each reproduces
the Mermaid source of the same number-family in [`../mermaid`](../mermaid). The six
full-width tables are the gate taxonomy (Table 1), the stage outputs (Table 2), the
six acceleration targets (Table 3), the five verification checks (Table 4), the
prior single-prompt repositories (Table 5), and the 2025 evidence chronology
(Table 6).

## Compile

```
pdflatex main  ->  bibtex main  ->  pdflatex main  ->  pdflatex main
```

## Sources used (Rule 5)

| Source | Supplies |
|:--|:--|
| [`../draft-paper`](../draft-paper) | The section scaffolds and `[DRAFTING INSTRUCTION]` pointers resolved here |
| [`../mermaid`](../mermaid) | The 24 figures reproduced as TikZ |
| [`../research/document-types`](../research/document-types) | The gate taxonomy, acceleration targets, time buckets |
| [`../research/industry-workflow`](../research/industry-workflow) | The before/during/after data and document workflow |
| [`../inputs/references.bib`](../inputs/references.bib) | The author-works citation keys |
| [`../../trial-protocol/final-protocol/publication`](../../trial-protocol/final-protocol/publication) | Image, white-space, and column-width formatting strategies |

## Next stage

[`../final-paper`](../final-paper) applies the senior-author polish (clearpage per
section, vspace/hspace, re-verified figures) for maximum quality.

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice.
