# 08Sep26 / diagrams - three figure specifications (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-5%20of%205-8A4B2A.svg)](..)
[![Figures](https://img.shields.io/badge/Figures-3-8A4B2A.svg)](.)
[![Platforms](https://img.shields.io/badge/Platforms-Mermaid%2C%20Diagrams%2C%20D2-6C757D.svg)](#the-three-figures)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)
[![Platform balance](https://img.shields.io/badge/Each%20platform-3%20of%2015-9AA1A8.svg)](#the-platform-balance-closes-here)

One specification per figure: the native source, the TikZ that renders it, the
coordinates and pitch, the provenance of every value, and the caption exactly as
printed.

## The three figures

| # | Platform | Native construct | Perspective no other figure in this day gives |
|:--|:--|:--|:--|
| 13 | Mermaid | Gantt across one market session | Time-of-day sequencing and overlap inside a single session |
| 14 | Diagrams | Clustered topology with vector glyphs | The weekly cadence as five standing functions with an owner and a budget |
| 15 | D2 | Layered stack with a count per layer | The thirty-day pipeline as layers rather than as a list |

Graphviz and PlantUML are unused on this day and appeared on days 1 to 4.

## The platform balance closes here

Across the five days, fifteen figures were drawn and each of the five platforms
carries exactly three:

| Platform | Figures |
|:--|:--|
| Mermaid | 1, 6, 13 |
| Graphviz | 2, 7, 12 |
| D2 | 3, 4, 15 |
| PlantUML | 5, 10 and the day 2 state machine's guards |
| Diagrams | 8, 14, and the day 3 campus topology |

The split follows purpose rather than quota: a flowchart where a state changes, a
record table where fields repeat, a grid where alternatives are compared, a state
machine where transitions carry conditions, and a glyph topology where functions
have locations. Where two platforms could have served, the one whose native
construct needed no invention was chosen.

## The one content rule specific to this day

**Figure 13 shows a session, not a promise.** Its bars are durations of work, not
commitments about when a counterparty replies. The one bar that depends on an
external party, the auction cutoff, is drawn as a vertical rule rather than as a
bar, because a cutoff is an instant and drawing it as a duration would misstate
it.

## The invariants every figure obeys

| # | Invariant | Value |
|:--|:--|:--|
| 1 | Frame | `\begin{appfig}[platform / construct]`, centered, ruled, white ground |
| 2 | Caption spacing | `\vspace{-0.60cm}`; 7.44 pt from rule to first caption line |
| 3 | Caption lines | Exactly two, balanced within a small character spread |
| 4 | Caption measure | `0.94\linewidth`, centered on the body measure |
| 5 | Palette | Ember Rust and its two lighter shades, three grays, white. No near-black fill |
| 6 | Node names | Plain letters and digits only |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `mermaid/`, `diagrams-python/`, `d2/` specification format | [`../../../capitalization-plan`](../../../capitalization-plan) | The structure of all three files |
| `final-capital/capstyle.sty` | [`../../../capitalization-plan`](../../../capitalization-plan) | The `mm*`, `dg*` and `d2*` styles, the vector glyphs, and `\ganttrow` |
| `final-capital/sections/sec-07-operating-plan.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Figure 14's standing functions |
| `../briefs/brief-01-thirty-day-pipeline.md` | This day | Figure 15's layers and counts |
| `../briefs/brief-02-weekly-cadence.md` | This day | Figure 14's five weekdays and time budgets |
| `../investing/capital-05-execution-and-settlement.md` | This day | Figure 13's entry sequence |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
