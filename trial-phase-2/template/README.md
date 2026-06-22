# template - recolored Phase 2 paper template (v1.1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Document color](https://img.shields.io/badge/Document%20color-%23800020-800020.svg)](protostyle.sty)
[![Palette](https://img.shields.io/badge/Palette-800020%20%2F%202E2E2E%20%2F%206B6B6B%20%2F%20C9C9C9%20%2F%20F5F5F5-6B6B6B.svg)](protostyle.sty)
[![Overleaf](https://img.shields.io/badge/Overleaf-pdfLaTeX-6B6B6B.svg)](main.tex)

This directory holds the reusable, single-column LaTeX template for the Phase II
protocol, recolored from the Phase I template to the five-step Phase II palette
with **Burgundy `#800020` as the document color**. Every build stage
(`draft-protocol`, `full-protocol`, `final-protocol`, and `final-protocol/publication`)
derives its `main.tex` and uses this `protostyle.sty`.

## Files

```
template/
  protostyle.sty   the recolored style (palette, mermaidfig, full-width tables, PNG-free TikZ ORCID iD)
  main.tex         a cover + clickable TOC + per-section \input skeleton
  README.md        this file
```

## Color scheme (the document color and the four supporting tones)

| Role | Color | Hex | TikZ node style |
|:--|:--|:--|:--|
| End goals, investigational system, decisions (document color) | Burgundy | `#800020` | `mmgoal` |
| Harm, raw data, blocked paths | Charcoal | `#2E2E2E` | `mmdark` |
| Process and oversight | Slate Gray | `#6B6B6B` | `mmstep` |
| Decision and warning | Mist Gray | `#C9C9C9` | `mmdec` |
| Inputs and context | Cloud | `#F5F5F5` | `mmin` |

## What changed from the Phase 1 template

The Phase 1 template used Corporate Blue `#00417A` as the document color with a
Professional Gray and Classic White supporting set. The Phase 2 template replaces
the accent with Burgundy `#800020` and defines the full five-step palette above;
the typographic machinery (RaggedRight body, widow and orphan control, ragged
bottom, full-width ragged-right table columns, the `mermaidfig` and `asciifig`
environments, and the PNG-free TikZ ORCID iD mark) is carried forward unchanged.

## Files from other directories used here

| Source | Used for |
|:--|:--|
| [`../../trial-protocol/template`](../../trial-protocol/template) | the Phase 1 single-column paper template recolored here |
| [`../final-protocol/publication/protostyle.sty`](../final-protocol/publication/protostyle.sty) | the canonical Phase 2 style this template mirrors |

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
