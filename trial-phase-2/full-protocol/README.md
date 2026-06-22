# full-protocol - Stage 3 (full render) Phase 2 protocol (v1.1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-3%20full%20render-800020.svg)](../sub-prompts/prompt-3-full-protocol.md)
[![Design](https://img.shields.io/badge/Design-Phase%202%20Randomized%20Multicenter-800020.svg)](.)
[![Sections](https://img.shields.io/badge/NIH%20sections-13-800020.svg)](sections)
[![Figures](https://img.shields.io/badge/TikZ%20figures-22-6B6B6B.svg)](.)
[![Tables](https://img.shields.io/badge/Full--width%20tables-11-6B6B6B.svg)](.)
[![Overleaf](https://img.shields.io/badge/Overleaf-pdfLaTeX-6B6B6B.svg)](main.tex)

This directory is the output of **Stage 3** (sub-prompt
[`../sub-prompts/prompt-3-full-protocol.md`](../sub-prompts/prompt-3-full-protocol.md)):
the **full render** of the Phase 2 protocol. Every `[DRAFTING INSTRUCTION]` from
[`../draft-protocol/`](../draft-protocol) is rendered into complete prose, with
each Mermaid figure drawn as a TikZ `mermaidfig` and every table filled at the
body measure.

## Relationship to the final stage

The full stage carries the complete rendered content; the senior-author pagination
polish is deferred to [`../final-protocol/`](../final-protocol). The visible
differences here are deliberate: `protostyle.sty` uses `\flushbottom` (the final
stage re-enables `\raggedbottom`), and `main.tex` runs the sections continuously
(the final stage adds a `\clearpage` after each NIH section). The section content,
the 22 TikZ figures, and the 11 full-width tables are the same as the final and
publication builds.

## Files

```
full-protocol/
  main.tex                  cover, clickable TOC, continuous sections
  protostyle.sty            recolored #800020; flushbottom; mermaidfig, tables, TikZ ORCID
  references.bib            daraxonrasib (5), main documents (3), Phase 1 predicate, author works, clinical, methods, standards
  sections/                 sec-00 .. sec-12 (13 NIH sections, 22 figures, 11 tables)
  prompt-full-protocol.md   this stage's sub-prompt, verbatim
  output-full-protocol.md   narrative output
  full-protocol-LaTeX.zip   Overleaf bundle
```

## Files from other directories used here

| Source | Used for |
|:--|:--|
| `../draft-protocol/` | the bracketed scaffold rendered here into full prose |
| `../mermaid/` | the 24 Phase 2 Mermaid figures reproduced as TikZ `mermaidfig` |
| `../../trial-protocol/inputs/2030-pdac-1min-final-paper` | the quantitative clinical and telemetry data |
| `../../trial-protocol/inputs/21cfr312_adapt` | the Physical AI Subpart J overlay |
| `../../trial-protocol/inputs/auto-bill-02` | the VVUQ and co-investment financial framing |
| `../../trial-protocol/nih-protocol/` | section order and required content |

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
