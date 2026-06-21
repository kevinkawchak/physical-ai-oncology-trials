# draft-protocol - Stage 2 (scaffold) (v1.0.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-2%20draft%20(scaffold)-00417A.svg)](../sub-prompts/prompt-2-draft-protocol.md)
[![Sections](https://img.shields.io/badge/NIH%20sections-13-00417A.svg)](sections)
[![Compiles](https://img.shields.io/badge/Overleaf-pdfLaTeX-6C757D.svg)](main.tex)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.xxxxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxxxx)

This directory is the output of **Stage 2** (sub-prompt
[`../sub-prompts/prompt-2-draft-protocol.md`](../sub-prompts/prompt-2-draft-protocol.md)):
the **draft (scaffold)** of the Phase 1 protocol. Every NIH section is present
and ordered; every content slot carries a bracketed `[DRAFTING INSTRUCTION]`
that names the exact repository file and the figure or table medium the full
stage will process. The project compiles in Overleaf as committed.

## Files

```
draft-protocol/
  main.tex                 cover, clickable TOC, one \input per section
  protostyle.sty           recolored #00417A; mermaidfig, asciifig, L/Y/R/C
                           columns, TikZ ORCID, senior-author formatting
  references.bib           daraxonrasib (5), main documents (3), author works,
                           clinical refs, FDA/CFR/standards
  sections/                sec-00 .. sec-12 (13 NIH sections)
  prompt-draft-protocol.md this stage's sub-prompt, verbatim
  output-draft-protocol.md narrative output
  draft-protocol-LaTeX.zip Overleaf bundle
```

## NIH sections (one `sections/*.tex` per section, Rule 6)

| File | NIH section |
|:--|:--|
| `sec-00-compliance.tex` | Statement of Compliance |
| `sec-01-summary.tex` | Protocol Summary (Synopsis, Schema, SoA) |
| `sec-02-introduction.tex` | Introduction (Rationale, Background, Risk/Benefit) |
| `sec-03-objectives.tex` | Objectives and Endpoints |
| `sec-04-design.tex` | Study Design |
| `sec-05-population.tex` | Study Population |
| `sec-06-intervention.tex` | Study Intervention |
| `sec-07-discontinuation.tex` | Intervention and Participant Discontinuation/Withdrawal |
| `sec-08-assessments.tex` | Study Assessments and Procedures |
| `sec-09-statistics.tex` | Statistical Considerations |
| `sec-10-oversight.tex` | Regulatory, Ethical, and Oversight Considerations |
| `sec-11-additional.tex` | Additional Considerations, Abbreviations, Amendment History |
| `sec-12-references-backmatter.tex` | References and Back Matter |

## Files from other directories used here (Rule 5)

| Source | Used for |
|:--|:--|
| `../template/tmpl01style.sty` | base style, recolored to `#00417A` in `protostyle.sty` |
| `../inputs/auto-bill-02/final-bill/usctitle.sty` | `mermaidfig`, `asciifig`, `L/Y/R` column primitives |
| `../mermaid/fig-01 .. fig-25` | bracketed TikZ-figure pointers in every section |
| `../inputs/2030-pdac-1min-final-paper` | clinical/device data and tables |
| `../inputs/21cfr312_adapt` | Subpart J overlay, AE reporting, holds, oversight |
| `../nih-protocol/01 .. 10` | section order and required content |
| `../research/*` | regulatory framing and the eight Physical AI concerns |
| `../inputs/author_works.bib` | directly relevant author works |

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
