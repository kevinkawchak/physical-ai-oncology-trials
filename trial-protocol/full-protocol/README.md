# full-protocol - Stage 3 (full rendering) (v1.0.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-3%20full-00417A.svg)](../sub-prompts/prompt-3-full-protocol.md)
[![Sections](https://img.shields.io/badge/NIH%20sections-13-00417A.svg)](sections)
[![Figures](https://img.shields.io/badge/TikZ%20figures-13%2B-6C757D.svg)](.)
[![Compiles](https://img.shields.io/badge/Overleaf-pdfLaTeX-6C757D.svg)](main.tex)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.xxxxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxxxx)

This directory is the output of **Stage 3** (sub-prompt
[`../sub-prompts/prompt-3-full-protocol.md`](../sub-prompts/prompt-3-full-protocol.md)):
the **full** Phase 1 protocol. It executes every draft `[DRAFTING INSTRUCTION]`
into finished prose, renders the referenced Mermaid figures as TikZ `mermaidfig`
environments with the same complexity and the protocol palette, and fills every
table with the quantitative data carried from the author sources. It does not
overwrite the draft.

## Files

```
full-protocol/
  main.tex                 cover, clickable TOC, one \input per section
  protostyle.sty           recolored #00417A; table aesthetics tuned to body width
  references.bib           daraxonrasib (5), main documents (3), author works,
                           clinical refs, FDA/CFR/standards
  sections/                sec-00 .. sec-12 (13 NIH sections, full)
  prompt-full-protocol.md  this stage's sub-prompt, verbatim
  output-full-protocol.md  narrative output
  full-protocol-LaTeX.zip  Overleaf bundle
```

## TikZ figures rendered (from the Stage 1 Mermaid catalog)

| Section | Figure(s) rendered as TikZ |
|:--|:--|
| sec-00 Compliance | fig-03 combined IND/IDE pathway |
| sec-01 Summary | fig-01 trial schema (+ Schedule of Activities table) |
| sec-02 Introduction | fig-19 counterfactual scenarios; fig-20 Physical AI concerns; fig-24 risk-benefit |
| sec-03 Objectives | fig-13 objectives-endpoints hierarchy (+ 3-column table) |
| sec-04 Design | fig-18 staged autonomy; fig-10 dose escalation |
| sec-05 Population | fig-02 CONSORT flow |
| sec-06 Intervention | fig-05 platform; fig-04 LLM advisory loop; fig-09 daraxonrasib advisory; fig-11 anastomoses (+ per-arm and sensor tables) |
| sec-08 Assessments | fig-07 vascular safety gate; fig-08 E-stop; fig-15 AE reporting |
| sec-09 Statistics | fig-22 analysis populations |
| sec-10 Oversight | fig-16 governance; fig-17 informed consent |
| sec-11 Additional | fig-12 VVUQ ten-gate (+ abbreviations table) |

## Files from other directories used here (Rule 5)

| Source | Used for |
|:--|:--|
| `../draft-protocol/` | the scaffold and its bracketed instructions, executed here |
| `../mermaid/fig-01 .. fig-25` | the figures rendered as TikZ |
| `../inputs/2030-pdac-1min-final-paper/sections/methods.tex` | per-arm tool, sensor, vessel-zone, ring-tension, force-clamp tables |
| `../inputs/2030-pdac-1min-final-paper/sections/results.tex` | daraxonrasib advisory distribution, composite metrics |
| `../inputs/21cfr312_adapt/*` | Subpart J overlay, AE reporting, holds, oversight, retention |
| `../nih-protocol/01 .. 10` | section order and required content |
| `../research/*` | regulatory framing and the eight Physical AI concerns |
| `../inputs/author_works.bib` | directly relevant author works |

## Quality gates applied (Stage 3)

Each TikZ figure was verified twice for (a) no text-box / arrow overlap, (b)
correct curved-arrow looseness via deliberate multi-segment routing, and (c)
proper spacing between boxes. Table column widths were optimized for the body
measure. Single hyphens only; the section symbol for codified references; no
raster images; white background throughout.

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
