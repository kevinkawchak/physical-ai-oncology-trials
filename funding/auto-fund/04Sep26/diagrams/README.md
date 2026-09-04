# 04Sep26 / diagrams - three figure specifications (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-3%20of%205-2F5D3A.svg)](..)
[![Figures](https://img.shields.io/badge/Figures-3-2F5D3A.svg)](.)
[![Platforms](https://img.shields.io/badge/Platforms-Graphviz%2C%20Diagrams%2C%20Mermaid-6C757D.svg)](#the-three-figures)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)
[![Glyphs](https://img.shields.io/badge/Glyphs-vector%2C%20no%20stick%20figures-9AA1A8.svg)](#the-glyph-rule)

One specification per figure: the native source, the TikZ that renders it, the
coordinates and pitch, the provenance of every value, and the caption exactly as
printed.

## The three figures

| # | Platform | Native construct | Perspective no other figure in this day gives |
|:--|:--|:--|:--|
| 7 | Graphviz | Three dashed clusters with edges between them | Which obligations belong to the site, which to the sponsor, and which to the developer |
| 8 | Diagrams | Clustered infrastructure with vector glyphs | Where each trial function physically sits, across two campuses and one company |
| 9 | Mermaid | Flowchart with labeled gates | The foundation funnel from a mechanism question to an award, with attrition at each gate |

D2 and PlantUML are unused on this day and appear on days 1, 2, 4 and 5. Over the
five days each platform is used exactly three times.

## The glyph rule

Figure 8 is the one figure in this build whose native form, `mingrammer/diagrams`,
renders an icon per node. `fundstyle.sty` supplies twenty-four vector pictograms
for exactly this, drawn in TikZ paths rather than loaded as images, and the
stick-figure actor macro from the parent style is deleted rather than left
unused. Where a person or a role appears in Figure 8, it appears as a labeled
tile carrying a pictogram of the function, not of a body.

## The one content rule specific to this day

**No figure asserts an agreement.** Figure 7 draws obligations that would exist
under a site agreement, and it labels that column as conditional. Figure 8 draws
where functions would sit, and its cluster titles name candidate institutions at
the feasibility stage. A reader who screenshots either figure must not be able to
come away believing an institution has agreed to anything.

## The invariants every figure obeys

| # | Invariant | Value |
|:--|:--|:--|
| 1 | Frame | `\begin{appfig}[platform / construct]`, centered, ruled, white ground |
| 2 | Caption spacing | `\vspace{-0.60cm}`; 7.44 pt from rule to first caption line |
| 3 | Caption lines | Exactly two, balanced within a small character spread |
| 4 | Caption measure | `0.94\linewidth`, centered on the body measure |
| 5 | Palette | Cypress Green and its two lighter shades, three grays, white. No near-black fill |
| 6 | Output | TikZ only |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `graphviz/`, `diagrams-python/`, `mermaid/` specification format | [`../../../capitalization-plan`](../../../capitalization-plan) | The structure of all three files |
| `final-capital/capstyle.sty` | [`../../../capitalization-plan`](../../../capitalization-plan) | The `gv*`, `dg*` and `mm*` styles and the vector pictograms |
| `UC-San-Diego/priority-steps.md` §12 | [`../../../potential-partners`](../../../potential-partners) | Figure 7's obligation rows in full |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../move-in`](../../../move-in) | Figure 8's function list |
| `../emails/email-03`, `email-04` | This day | Figure 9's gates |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
