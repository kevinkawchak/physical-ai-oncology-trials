# Diagrams (python)-type figures - Capitalization Plan (v4.5.0)

[![Platform](https://img.shields.io/badge/Platform-diagrams%20(python)-3C7DB2.svg)](https://diagrams.mingrammer.com)
[![Figures](https://img.shields.io/badge/Figures-3-00417A.svg)](.)
[![Emits](https://img.shields.io/badge/Emits-specification%2C%20not%20.py-6C757D.svg)](.)
[![Stage](https://img.shields.io/badge/Stage-4%20of%208-6C757D.svg)](../sub-prompts/stage-4-diagrams-python)
[![Glyphs](https://img.shields.io/badge/Glyphs-pure%20TikZ%20vector-9AA1A8.svg)](.)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

Three figure specifications produced by
[`../sub-prompts/stage-4-diagrams-python/`](../sub-prompts/stage-4-diagrams-python).
Each is reproduced in LaTeX by the `dg*` TikZ vocabulary in `capstyle.sty`. This
vocabulary is used wherever the claim is about **where something sits and which
boundary it crosses**.

## Contents

| File | Figure | § | Clusters | The question it answers |
|:--|:--|:--|:--|:--|
| [`fig-04-asset-zones.md`](fig-04-asset-zones.md) | 4 | 2 | 4, one empty | What is the shape of what the company holds? |
| [`fig-18-operating-topology.md`](fig-18-operating-topology.md) | 18 | 7 | 3, two boundaries | Who is employed, who is contracted, who is contributed? |
| [`fig-20-artifact-custody.md`](fig-20-artifact-custody.md) | 20 | 10 | 4 custodians | What survives if the programme stops, and who holds it? |

## No `.py` file is written

The `diagrams` library renders through Graphviz to a raster, and this paper
generates no raster. The stage therefore emits a machine-readable specification
in Markdown: node graph, cluster membership, glyph assignment, and TikZ
placement. Every tile is drawn in LaTeX by a `\glyph*` macro, which is pure
vector TikZ.

There is a second reason. The repository runs three `lint-and-format` jobs on
Python 3.10, 3.11 and 3.12, each running `ruff check` and `ruff format --check`
across the whole tree. A `.py` file added here would have to satisfy both on
every version. A specification carries the same information and cannot break the
build. The rule is inherited from
`funding/pdac-funding-applications/diagrams-python/README.md`.

## Glyph inventory used across the three figures

| Glyph | Used by | Meaning in this paper |
|:--|:--|:--|
| `\glyphdoc` | 4, 18, 20 | A document that exists as a file |
| `\glyphflask` | 4, 18, 20 | An investigational or laboratory object |
| `\glyphcpu` | 4, 18, 20 | Computation |
| `\glyphdb` | 4, 18, 20 | A store with a retention term |
| `\glyphai` | 4, 18 | The advisory model |
| `\glyphrobot` | 4, 18 | The surgical platform |
| `\glyphpill` | 4 twice, 18 | The investigational agent |
| `\glyphhand` | 4, 20 | A signature or a consent |
| `\glyphshield` | 4, 18 | An oversight body |
| `\glyphlink` | 4, 20 | A cross-reference between filings |
| `\glyphbank` | 4, 20 | A funder or an agency file |
| `\glyphuser`, `\glyphteam`, `\glyphgear` | 18, 20 | People and roles |
| `\glyphchart`, `\glyphmon`, `\glyphscalpel`, `\glyphlock`, `\glyphsignal` | 16, 18, 20 | Measures, monitoring, surgery, custody, reporting |

`\glyphpill` appears twice inside Figure 4, once in the Licensed zone and once
in the Absent zone. The repetition is deliberate: the same object is licensed to
somebody else and absent from this company.

## Anti-defect record

| Defect class | How these three avoid it |
|:--|:--|
| Label inside the tile | Every tile uses `\dgnode`, `\dgnodew` or `\dgnodeg`, each of which sets the label 5.4 mm beneath the 9 mm tile |
| Cluster border cutting a label | Every `fit` names both the tile node and its label node, `(n1)(n1l)` |
| Tiles too close | Horizontal pitch 27 mm, vertical pitch 22 mm, in all three figures |
| Dropped empty group | Figure 4's contracted zone is drawn as a fixed 29 by 19 mm rectangle, the size a three-tile zone would take, because a `fit` over zero nodes has no extent |
| Boundary collision | Figure 18's two rules are vertical and full height with 11 mm and 12 mm to the nearest cluster edge; only four labelled edges cross them |
| Unfinished edge | Figure 20's blocked edge is drawn to one third length and terminated with `\pxmark`, so it reads as a path that does not complete |

## Rule 5 source map

| These figures use | From | For |
|:--|:--|:--|
| `Physical AI Oncology Trial Founding Documents.md` | `../../supplementary` | Figure 4's owned zone and Figure 20's public custodian |
| `Physical-AI-Oncology-Trial-Competition-Proposal.zip` | `../../supplementary/source-files` | Figure 4's January 13, 2026 baseline |
| `final-apply/sections/sec-06-physical-ai-governance.tex` | `../../pdac-funding-applications` | Figure 18's trust boundaries and Figure 20's replay method |
| `final-apply/sections/sec-09-build-method.tex` | `../../pdac-funding-applications` | Figure 18's Phase I staffing |
| `final-apply/sections/sec-10-risks-and-limits.tex` | `../../pdac-funding-applications` | Figure 20's reproduce-the-81.9 record |
| `UC-San-Diego/` | `../../potential-partners` | Figures 4, 18, 20, the site's functions and file |
| `trial-protocol/`, `trial-ind/`, `trial-phase-2/` | repository root | All three figures |
| 21 CFR §312.57 and §312.62 | codified | Figure 20's two retention terms |
| `final-apply/applystyle.sty` | `../../pdac-funding-applications` | The `dg*` vocabulary and all 24 `\glyph*` macros |
