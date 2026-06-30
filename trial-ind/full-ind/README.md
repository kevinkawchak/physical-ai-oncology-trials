# full-ind - Stage 3 (full rendering) (IND v1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-3%20full-3F3F3F.svg)](../sub-prompts/prompt-3-full-ind.md)
[![Sections](https://img.shields.io/badge/IND%20sections-12-000000.svg)](sections)
[![Figures](https://img.shields.io/badge/TikZ%20figures-22-6C757D.svg)](.)
[![Compiles](https://img.shields.io/badge/Overleaf-pdfLaTeX-6C757D.svg)](main.tex)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.xxxxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxxxx)
[![Repository](https://img.shields.io/badge/Repository-v4.3.0-blue.svg)](../../README.md)

Stage 3 of the `trial-ind/` build: the full IND. Every bracketed
`[DRAFTING INSTRUCTION]` from `../draft-ind/` is executed here, replaced with
finished IND prose, full-width ragged-right tables, and the 22 grayscale Mermaid
figures reproduced one-to-one as TikZ `mermaidfig` figures. The quantitative data
needed for Phase 1 review is carried from the author sources: the DL1 160 / DL2 220
/ DL3 300 mg dose levels, the 3+3 / 28-day-DLT rule, the eight-arm device
specification, the five-vessel no-fly gate, the three anastomosis ring-tension
bands, the perioperative advisory sweep (29 / 3 / 0 of 32), the Dutch 2025
benchmark comparators, the safety-reporting clocks, and the analysis populations.

## Files

| File | Purpose |
|:--|:--|
| [`main.tex`](main.tex) | Cover page, ReGARDD ordering, numbered Table of Contents, `\input` of all sections, `\clearpage` per section. |
| [`indstyle.sty`](indstyle.sty) | The shared grayscale style (identical to the draft stage). |
| [`references.bib`](references.bib) | Author `@misc` entries; `ieeetr`, clickable URLs and DOI text plus clickable DOI URLs. |
| `sections/sec-00 .. sec-11` | The 12 full IND sections (Rule 6), each with its assigned TikZ figures and tables. |

## TikZ figures rendered (from the Stage 1 Mermaid catalog)

All 22 figures from [`../mermaid/`](../mermaid) appear once across the sections:
sec-00 (figs 1, 2), sec-01 (3), sec-02 (6, 14), sec-03 (4, 8, 10, 15, 21),
sec-04 (17), sec-05 (18, 19), sec-06 (16), sec-08 (13, 22), sec-09 (20),
sec-10 (9, 11). Each is reproduced with the same nodes, edges, grayscale tones,
and quantitative values as its Mermaid source.

## Quality gates applied (Stage 3, verify twice)

- No text-box or arrow overlaps in any figure.
- Curved arrows carry the specified `looseness` (0.4 to 0.7).
- Proper spacing between boxes (side-by-side centers at least 4.0 cm apart).
- Tables sit at the body measure (`\textwidth`) with `>{\raggedright\arraybackslash}`
  columns sized to the text per column.

## Files from other directories used here (Rule 5)

| Source | Used for |
|:--|:--|
| [`../draft-ind/`](../draft-ind) | the scaffold and its bracketed instructions, executed here |
| [`../mermaid/fig-01 .. fig-22`](../mermaid) | the figures rendered as TikZ |
| [`../../trial-protocol/final-protocol/publication/sections`](../../trial-protocol/final-protocol/publication/sections) | clinical content, quantitative tables, and data |
| [`../../trial-documents/final-paper/publication/sections`](../../trial-documents/final-paper/publication/sections) | the acceleration argument and the back matter |
| [`../inputs`](../inputs) | the ReGARDD IND template, FDA 1571 instructions, ReGARDD guidance, references |

## Compile (Overleaf, pdfLaTeX)

`pdflatex main` then `bibtex main` then `pdflatex main` twice. The packaged
`full-ind-LaTeX.zip` is a self-contained project.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
