# final-protocol - Stage 4 (polished) Phase 2 protocol (v1.1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-4%20final%20polished-800020.svg)](../sub-prompts/prompt-4-final-protocol.md)
[![Design](https://img.shields.io/badge/Design-Phase%202%20Randomized%20Multicenter-800020.svg)](.)
[![Sections](https://img.shields.io/badge/NIH%20sections-13-800020.svg)](sections)
[![Figures](https://img.shields.io/badge/TikZ%20figures-22-6B6B6B.svg)](.)
[![Tables](https://img.shields.io/badge/Full--width%20tables-11-6B6B6B.svg)](.)
[![Overleaf](https://img.shields.io/badge/Overleaf-pdfLaTeX-6B6B6B.svg)](main.tex)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20807027-6B6B6B.svg)](https://doi.org/10.5281/zenodo.20807027)

The author-edited, publication-ready build is in
[`publication/`](publication), which is the paper URL directory. This directory is the output of **Stage 4** (sub-prompt
[`../sub-prompts/prompt-4-final-protocol.md`](../sub-prompts/prompt-4-final-protocol.md)):
the **polished, final** Phase 2 protocol, at maximum context and formatting
quality. 

## What changed from the full protocol (the final polish)

| Correction | Where | Effect |
|:--|:--|:--|
| `\raggedbottom` re-enabled | `protostyle.sty` | removes large inter-paragraph white gaps the full stage left |
| `\clearpage` after every NIH section | `main.tex` | each of the 13 sections is self-standing |
| Figure overlap, curved-arrow looseness, box spacing re-verified | all `sections/*` figures | every TikZ `mermaidfig` reads cleanly |
| Bibliography ragged-right; ORCID iD + URL; section symbol; single hyphens | throughout | senior-author proof-reading pass |

## Files

```
final-protocol/
  main.tex                  cover, clickable TOC, \clearpage per section
  protostyle.sty            recolored #800020; raggedbottom; mermaidfig, tables, TikZ ORCID
  references.bib            daraxonrasib (5), main documents (3), Phase 1 predicate, author works, clinical, methods, standards
  sections/                 sec-00 .. sec-12 (13 NIH sections, 22 figures, 11 tables)
  prompt-final-protocol.md  this stage's sub-prompt, verbatim
  output-final-protocol.md  narrative output
  final-protocol-LaTeX.zip  Overleaf bundle
  publication/              author-edited paper URL directory
```

## Files from other directories used here

| Source | Used for |
|:--|:--|
| `../full-protocol/` | the full protocol, polished here (pagination and white-space pass) |
| `../mermaid/` | the 24 Phase 2 Mermaid figures reproduced as TikZ `mermaidfig` |
| `../../trial-protocol/final-protocol/publication` | the Phase 1 `\clearpage` / `\vspace` / table-width proof-reading techniques learned here |
| `../../trial-protocol/nih-protocol/` | section order and required content |

## Quality verification (static)

Balanced environments across 13 sections; every cite key resolves; 22 TikZ
figures and 11 full-width tables; the locked constants (n = 220, HR 0.60, 85
percent power, about 140 events, RP2D 300 mg, eight sites, USL >= 8.0) consistent
throughout; single hyphens only; section symbol for codified references; no raster
images; white background; Burgundy `#800020` document color.

## Compile (Overleaf, pdfLaTeX)

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
