## output-draft-paper

Stage 2 of the mermaid -> draft -> full -> final build is complete. Process B
executed [`prompt-draft-paper.md`](prompt-draft-paper.md) and produced the draft
(scaffold) of *Phase 1 Pancreatic Cancer Trial Efficient LLM Document Generations*
(paper v1.0, repository v4.2.0), one commit per file pushed in real time.

The draft compiles on Overleaf with pdfLaTeX as a scaffold: `main.tex` builds the
cover and front matter and `\input`s all eight section files; `paperstyle.sty`
carries the five-color figure palette, black body text, no ORCID logo (Rule 12),
the full-width ragged-right table column types, and the `mermaidfig` TikZ
environment; and `references.bib` carries every citation with a clickable DOI and a
clickable DOI URL.

Each section file (`sections/sec-01-abstract.tex` through
`sections/sec-08-references-backmatter.tex`) is a scaffold whose bracketed
`[DRAFTING INSTRUCTION]` pointers (`\draftinstr`) name the exact repository files
the full stage will process: the 24 `../mermaid/fig-NN-*.md` figures; the four
research sources in `../research/document-types` and `../research/industry-workflow`;
the author works in `../inputs/references.bib`; the main documents (the 2030
60-second PDAC simulation, H. R. 9510 v5.0, the National Platform framework, and the
five DARAXONRASIB entries); the Phase 2 directory `../../trial-phase-2`; and the
cited single-prompt repositories.

The Table of Contents is generated in `main.tex` (placed after the Introduction per
the PAPER FORMAT). The back matter adds the Keywords and the Rights and Permissions
(CC) sections to the adoption-guide template. The next stage,
[`../full-paper`](../full-paper), resolves every drafting instruction into full
prose, TikZ figures, and full-width tables.
