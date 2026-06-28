# trial-documents/final-paper - Stage 4 final paper

[![Stage](https://img.shields.io/badge/Stage-4%20Final-8B2E3F.svg)](.)
[![Sections](https://img.shields.io/badge/Sections-8%20.tex-2F5D7C.svg)](sections)
[![Figures](https://img.shields.io/badge/TikZ%20figures-24-D08770.svg)](sections)
[![Quality](https://img.shields.io/badge/Quality-maximum%20(polished)-8B2E3F.svg)](.)
[![Paper](https://img.shields.io/badge/Paper-v1.0-BFD7EA.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.2.0-2F5D7C.svg)](../../README.md)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)

Stage 4 of the mermaid -> draft -> full -> final build for the paper *Phase 1
Pancreatic Cancer Trial Efficient LLM Document Generations* (paper v1.0, repository
v4.2.0). This is the polished, maximum-quality source. There is no `publication`
subdirectory (per the rules).

## Senior-author polish applied here

- Each major section starts on a fresh page via `\clearpage` in `main.tex`, so every
  section is self-standing and no line is stranded across a page boundary.
- The `mermaidfig` environment wraps each figure in an `adjustbox` max-width box, so
  no figure runs off the right margin; the densest figure (Figure 16) is expanded to
  full fidelity with its schedule-value note and constraint-limit nodes.
- Tables use `>{\raggedright\arraybackslash}` columns at `\textwidth` with widths
  tuned to the text per column, following the
  `trial-protocol/final-protocol/publication` methods.
- `\RaggedRight` with even interword spacing, maximal widow/orphan penalties, and
  `raggedbottom` prevent stranded single lines and large vertical gaps; long URLs
  break on any character; single dashes only; the section symbol renders as §.

## Files

| File | Role |
|:--|:--|
| [`main.tex`](main.tex) | Polished assembly with `\clearpage` per section |
| [`paperstyle.sty`](paperstyle.sty) | Shared polished style (adjustbox, ragged-right tables, mermaidfig) |
| [`references.bib`](references.bib) | Bibliography (clickable DOI and DOI URL per entry) |
| [`prompt-final-paper.md`](prompt-final-paper.md) | The Stage 4 sub-prompt executed here |
| [`output-final-paper.md`](output-final-paper.md) | The Stage 4 narrative output |
| [`sections/`](sections) | Eight polished section `.tex` files (Rule 6) |
| `final-paper-LaTeX.zip` | The Overleaf bundle |

## Compile

```
pdflatex main  ->  bibtex main  ->  pdflatex main  ->  pdflatex main
```

## Sources used (Rule 5)

| Source | Supplies |
|:--|:--|
| [`../full-paper`](../full-paper) | The full sections refined here |
| [`../mermaid`](../mermaid) | The 24 figures reproduced as TikZ |
| [`../research`](../research) | The document-types and industry-workflow grounding |
| [`../inputs/references.bib`](../inputs/references.bib) | The author-works citation keys |
| [`../../trial-protocol/final-protocol/publication`](../../trial-protocol/final-protocol/publication) | The clearpage, vspace/hspace, and column-width formatting strategies |

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice.
