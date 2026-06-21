# Zenodo Publication with Author Edits - Stage 4 (polished) (v1.0.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-4%20final-00417A.svg)](../sub-prompts/prompt-4-final-protocol.md)
[![Sections](https://img.shields.io/badge/NIH%20sections-13-00417A.svg)](sections)
[![Figures](https://img.shields.io/badge/TikZ%20figures-20-6C757D.svg)](.)
[![Tables](https://img.shields.io/badge/Full--width%20tables-11-6C757D.svg)](.)
[![Compiles](https://img.shields.io/badge/Overleaf-pdfLaTeX-6C757D.svg)](main.tex)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--5457--8667-A6CE39.svg)](https://orcid.org/0009-0007-5457-8667)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20780121-blue.svg)](https://doi.org/10.5281/zenodo.20780121)

[Zenodo Protocol Files](https://doi.org/10.5281/zenodo.20780121). This directory is the output of **Stage 4** (sub-prompt
[`../sub-prompts/prompt-4-final-protocol.md`](../sub-prompts/prompt-4-final-protocol.md)):
the **polished, final** Phase 1 protocol, at maximum context and formatting
quality. It starts from the full protocol and implements the corrections
identified there. There is no `publication` subdirectory.

## What changed from the full protocol (the final corrections)

| Correction | Where | Effect |
|:--|:--|:--|
| Counterfactual figure expanded to all 3 scenarios | `sections/sec-02` Figure 3 | full Mermaid fidelity; clean fork layout, no overlap |
| Concerns figure expanded to all 8 concern-mitigation pairs | `sections/sec-02` Figure 4 | matches Table 1; black-box answer emphasized in `#00417A` |
| `\clearpage` between every NIH section | `main.tex` | each of the 13 sections is self-standing |
| `\raggedbottom` | `protostyle.sty` | removes large inter-paragraph white gaps |
| Bibliography ragged-right; ORCID iD + URL; section symbol; single hyphens | throughout | senior-author proof-reading pass |

## Files

```
final-protocol/
  main.tex                  cover, clickable TOC, \clearpage per section
  protostyle.sty            recolored #00417A; raggedbottom; mermaidfig, tables, TikZ ORCID
  references.bib            daraxonrasib (5), main documents (3), author works, clinical, standards
  sections/                 sec-00 .. sec-12 (13 NIH sections, final)
  prompt-final-protocol.md  this stage's sub-prompt, verbatim
  output-final-protocol.md  narrative output
  final-protocol-LaTeX.zip  Overleaf bundle
```

## Files from other directories used here (Rule 5)

| Source | Used for |
|:--|:--|
| `../full-protocol/` | the full protocol, refined here (not overwritten) |
| `../mermaid/fig-19`, `../mermaid/fig-20` | the two figures restored to full fidelity |
| `../inputs/2030-pdac-1min-final-paper` | the quantitative tables and clinical data |
| `../inputs/21cfr312_adapt` | the Physical AI IND overlay |
| `../inputs/auto-bill-02/final-bill` | the `\clearpage` / `\vspace` / table-width proof-reading techniques learned here |
| `../nih-protocol/` | section order and required content |

## Quality verification (static)

Balanced braces and environments (47 begin / 47 end); every cite key resolves;
20 TikZ figures and 11 full-width tables across 13 sections; 14 `\clearpage`
boundaries; dose ladder (160 / 220 / 300 mg) and sample size (n up to 18)
consistent throughout; single hyphens only; section symbol for codified
references; no raster images; white background.

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
