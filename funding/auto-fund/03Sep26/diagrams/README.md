# 03Sep26 / diagrams - three figure specifications (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-2%20of%205-1B3A5C.svg)](..)
[![Figures](https://img.shields.io/badge/Figures-3-1B3A5C.svg)](.)
[![Platforms](https://img.shields.io/badge/Platforms-D2%2C%20PlantUML%2C%20Mermaid-6C757D.svg)](#the-three-figures)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)
[![Stick figures](https://img.shields.io/badge/Stick%20figures-none-9AA1A8.svg)](#the-actor-problem-and-how-it-is-solved)

One specification per figure: the native source, the TikZ that renders it, the
coordinates and pitch, the provenance of every value, and the caption exactly as
printed.

## The three figures

| # | Platform | Native construct | Perspective no other figure in this day gives |
|:--|:--|:--|:--|
| 4 | D2 | Three-column container comparison | Three instruments on the same eight attributes, side by side, with no arrow between them |
| 5 | PlantUML | State machine with guards | The financing as states and the four conditions that gate the transitions |
| 6 | Mermaid | Sequence with lifelines | Who signs what, in what order, and who waits while someone else acts |

Graphviz and Diagrams are unused on this day and appear on days 1, 3, 4 and 5.
Over the five days each platform is used exactly three times.

## The actor problem, and how it is solved

A sequence diagram and a use-case diagram both want an actor glyph, and the
parent style drew one as a circle with limbs. `fundstyle.sty` deletes that macro,
so Figure 6's four participants are drawn as `mmactor` boxes carrying a role name
rather than a person. That is closer to how a signing sequence actually works:
the participant is an office, not a body.

## The invariants every figure obeys

| # | Invariant | Value |
|:--|:--|:--|
| 1 | Frame | `\begin{appfig}[platform / construct]`, centered, ruled, white ground |
| 2 | Caption spacing | `\vspace{-0.60cm}`; 7.44 pt from rule to first caption line |
| 3 | Caption lines | Exactly two, balanced within a small character spread |
| 4 | Caption measure | `0.94\linewidth`, centered on the body measure |
| 5 | Palette | Harbor Navy and its two lighter shades, three grays, white. No near-black fill |
| 6 | Output | TikZ only |

## The one content rule specific to this day

No figure states a term as offered. Figure 4 compares attributes and Figure 5
draws states; neither carries a valuation, a cap, a discount, or a closing date.
A figure is the easiest thing in a document to screenshot out of context, so the
figures on this day are written to be safe out of context.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `d2/`, `plantuml/`, `mermaid/` specification format | [`../../../capitalization-plan`](../../../capitalization-plan) | The structure of all three files |
| `final-capital/capstyle.sty` | [`../../../capitalization-plan`](../../../capitalization-plan) | The `d2*`, `uml*` and `mm*` style names |
| `final-capital/sections/sec-04-capital-bridge.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Figure 5's state names and guard conditions |
| `../briefs/brief-01-instrument-comparison.md` | This day | Figure 4's eight attributes in full |
| `../briefs/brief-02-firewall-and-part-54.md` | This day | Figure 6's participant list and signing order |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
