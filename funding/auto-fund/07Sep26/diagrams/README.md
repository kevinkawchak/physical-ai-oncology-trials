# 07Sep26 / diagrams - three figure specifications (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-5B3A5E.svg)](..)
[![Figures](https://img.shields.io/badge/Figures-3-5B3A5E.svg)](.)
[![Platforms](https://img.shields.io/badge/Platforms-PlantUML%2C%20D2%2C%20Graphviz-6C757D.svg)](#the-three-figures)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)
[![Gates](https://img.shields.io/badge/Fault%20tree%20gates-AND%20and%20OR-9AA1A8.svg)](fig-12-week-failure-tree.md)

One specification per figure: the native source, the TikZ that renders it, the
coordinates and pitch, the provenance of every value, and the caption exactly as
printed.

## The three figures

| # | Platform | Native construct | Perspective no other figure in this day gives |
|:--|:--|:--|:--|
| 10 | PlantUML | Activity with a fork and a join | The two concurrent lanes a closed market forces, and where they rejoin |
| 11 | D2 | SQL table records with an access class column | The data room as typed records rather than as a folder list |
| 12 | Graphviz | Fault tree with AND and OR gates | What combination of failures makes the week produce nothing |

Mermaid and Diagrams are unused on this day and appear on days 1, 2, 3 and 5.
Over the five days each platform is used exactly three times.

## Why a fault tree, and not a risk list

Figure 12 is the only figure in the five-day block that answers a question by
**combination** rather than by enumeration. A risk list says four things could go
wrong. A fault tree says which pairs of them are survivable and which single one
is not, and that is the difference between a list and an analysis.

The AND and OR gate glyphs come from the parent style and are drawn as filled
shapes in the palette's mid and light grays, never in a near-black fill.

## The one content rule specific to this day

**No figure shows anything as done.** Figure 10's right-hand lane is drawn in the
pale shade and labeled as held; Figure 11's access class column carries the words
"under a confidentiality agreement" for the two folders behind one; Figure 12's
root node is a condition, not an event. A reader who screenshots any of the three
must come away understanding that the day produced preparation and not action.

## The invariants every figure obeys

| # | Invariant | Value |
|:--|:--|:--|
| 1 | Frame | `\begin{appfig}[platform / construct]`, centered, ruled, white ground |
| 2 | Caption spacing | `\vspace{-0.60cm}`; 7.44 pt from rule to first caption line |
| 3 | Caption lines | Exactly two, balanced within a small character spread |
| 4 | Caption measure | `0.94\linewidth`, centered on the body measure |
| 5 | Palette | Slate Plum and its two lighter shades, three grays, white. No near-black fill |
| 6 | Node names | Plain letters and digits only. A generated name carrying a minus or a dot is read as a subtraction or as a node-and-anchor pair |

Invariant 6 is recorded because it caused two real compile failures in this
build, both fatal, and both diagnosed only from the log.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `plantuml/`, `d2/`, `graphviz/` specification format | [`../../../capitalization-plan`](../../../capitalization-plan) | The structure of all three files |
| `final-capital/capstyle.sty` | [`../../../capitalization-plan`](../../../capitalization-plan) | The `uml*`, `d2*` and `gv*` styles, and the AND and OR gate glyphs |
| `final-capital/sections/sec-09-risks-and-limits.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Figure 12's stop conditions |
| `../briefs/brief-01-data-room-index.md` | This day | Figure 11's nine records in full |
| `../README.md` | This day | Figure 10's two lanes |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
