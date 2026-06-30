# draft-ind - Stage 2 (scaffold) (IND v1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-2%20draft%20(scaffold)-3F3F3F.svg)](../sub-prompts/prompt-2-draft-ind.md)
[![Sections](https://img.shields.io/badge/IND%20sections-12-000000.svg)](sections)
[![Compiles](https://img.shields.io/badge/Overleaf-pdfLaTeX-6C757D.svg)](main.tex)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.xxxxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxxxx)
[![Repository](https://img.shields.io/badge/Repository-v4.3.0-blue.svg)](../../README.md)

Stage 2 of the `trial-ind/` build: the draft IND scaffold. Each ReGARDD IND Table
of Contents section is laid out as its own `sections/sec-*.tex` file and filled
with bracketed `[DRAFTING INSTRUCTION]` markers (the `\draftinstr` macro) that name
the exact `physical-ai-oncology-trials` files the full stage will process, and that
point to the grayscale figures (`../mermaid/fig-01 .. fig-22`) to render.

## Files

| File | Purpose |
|:--|:--|
| [`main.tex`](main.tex) | Cover page, ReGARDD ordering (Cover Letter and FDA 1571 before the numbered Table of Contents), `\input` of all sections, `\clearpage` per section. |
| [`indstyle.sty`](indstyle.sty) | The shared grayscale style: adapts the 21 CFR Part 312 template and the final-paper paperstyle primitives; black body / links / headings; eight-tone grayscale `mm*` figure styles; a back-matter section command. |
| [`references.bib`](references.bib) | Copied from `../inputs/references.bib` (50 author `@misc` entries), extended per stage; clickable URLs and DOI text plus clickable DOI URLs via `ieeetr`. |
| `sections/sec-00 .. sec-11` | One `.tex` per IND TOC section (Rule 6). |

## IND sections (one `sections/*.tex` per section, Rule 6)

`sec-00-cover-letter`, `sec-01-fda-forms`, `sec-02-introduction`,
`sec-03-general-investigational-plan`, `sec-04-investigator-brochure`,
`sec-05-proposed-clinical-research`, `sec-06-cmc`,
`sec-07-pharmacology-toxicology`, `sec-08-previous-human-experience`,
`sec-09-additional-information`, `sec-10-relevant-information`,
`sec-11-references-backmatter`.

## Files from other directories used here (Rule 5)

| Source | Used for |
|:--|:--|
| [`../mermaid/fig-01 .. fig-22`](../mermaid) | bracketed figure pointers in every section |
| [`../../regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.sty`](../../regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.sty) | the paper template `indstyle.sty` adapts (plus a back-matter section) |
| [`../../trial-documents/final-paper/publication/sections`](../../trial-documents/final-paper/publication/sections) | the acceleration argument and the back matter to adapt |
| [`../../trial-protocol/final-protocol/publication/sections`](../../trial-protocol/final-protocol/publication/sections) | clinical content, tables, and quantitative data named in the instructions |
| [`../inputs/ReGARDD_IND_Template.docx`](../inputs/ReGARDD_IND_Template.docx) | section order and required content |
| [`../inputs/FDA-1571_Instructions_R14_03-21-2023.md`](../inputs/FDA-1571_Instructions_R14_03-21-2023.md) | FDA 1571 / 3674 fields |
| [`../inputs/references.bib`](../inputs/references.bib) | citations |

## Compile (Overleaf, pdfLaTeX)

`pdflatex main` then `bibtex main` then `pdflatex main` twice. The packaged
`draft-ind-LaTeX.zip` contains `main.tex`, `indstyle.sty`, `references.bib`, and
`sections/`.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
