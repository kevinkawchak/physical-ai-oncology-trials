# trial-documents/draft-paper - Stage 2 draft (scaffold)

[![Stage](https://img.shields.io/badge/Stage-2%20Draft-2F5D7C.svg)](.)
[![Sections](https://img.shields.io/badge/Sections-8%20.tex-8B2E3F.svg)](sections)
[![Build](https://img.shields.io/badge/Build-pdfLaTeX%20%2B%20bibtex-D08770.svg)](.)
[![Paper](https://img.shields.io/badge/Paper-v1.0%20draft-BFD7EA.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.2.0-2F5D7C.svg)](../../README.md)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)

Stage 2 of the mermaid -> draft -> full -> final build for the paper *Phase 1
Pancreatic Cancer Trial Efficient LLM Document Generations* (paper v1.0, repository
v4.2.0). The draft is the paper's first LaTeX files: a compiling scaffold that lays
out every section and carries bracketed `[DRAFTING INSTRUCTION]` pointers
(`\draftinstr`) naming the exact repository files the full stage processes.

## Files

| File | Role |
|:--|:--|
| [`main.tex`](main.tex) | Cover, front matter, and `\input` of all eight sections; ToC after the Introduction (PAPER FORMAT order) |
| [`paperstyle.sty`](paperstyle.sty) | Style: five-color figure palette, black body text, no ORCID logo, full-width ragged-right tables, mermaidfig TikZ environment |
| [`references.bib`](references.bib) | Bibliography (clickable DOI and DOI URL per entry) |
| [`prompt-draft-paper.md`](prompt-draft-paper.md) | The Stage 2 sub-prompt executed here |
| [`output-draft-paper.md`](output-draft-paper.md) | The Stage 2 narrative output |
| [`sections/`](sections) | One `.tex` per paper section (Rule 6) |
| `draft-paper-LaTeX.zip` | The Overleaf bundle (main.tex, paperstyle.sty, references.bib, sections/) |

## Compile

```
pdflatex main  ->  bibtex main  ->  pdflatex main  ->  pdflatex main
```

## Sources used (Rule 5)

| Source | Supplies |
|:--|:--|
| [`../inputs/llm-adoption/main.tex`](../inputs/llm-adoption/main.tex) | The paper template (layout, repository-LLM and prompt guidance) |
| [`../mermaid`](../mermaid) | The 24 figures named in the `\draftinstr` pointers |
| [`../research/document-types`](../research/document-types) | The gate taxonomy and the six acceleration targets |
| [`../research/industry-workflow`](../research/industry-workflow) | The before/during/after Phase 1 document workflow |
| [`../inputs/references.bib`](../inputs/references.bib) | The author-works citation keys |
| [`../../trial-protocol/final-protocol/publication`](../../trial-protocol/final-protocol/publication) | Image, white-space, and formatting code strategies (not the template) |

## Next stage

[`../full-paper`](../full-paper) resolves every `[DRAFTING INSTRUCTION]` into full
prose, TikZ figures, and full-width tables.

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice.
