# 29Sep26 / diagrams - three figure specifications (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-2%20of%205-34495E.svg)](..)
[![Figures](https://img.shields.io/badge/Figures-4%20to%206-34495E.svg)](.)
[![Platforms](https://img.shields.io/badge/Platforms-PlantUML%2C%20Mermaid%2C%20D2-6C757D.svg)](#the-three-figures)
[![Spacing](https://img.shields.io/badge/Caption%20spacing--0.60cm-6C757D.svg)](#the-caption-rule)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

One specification per figure: the native source on its own platform, the TikZ
construction that reproduces it in the packet, the exact caption, and the
section that uses it.

## The three figures

| # | File | Platform | Native construct | Used in |
|:--|:--|:--|:--|:--|
| 4 | [`fig-04-request-state-machine.md`](fig-04-request-state-machine.md) | PlantUML | State diagram with guards | `sec-02-action-register.tex` |
| 5 | [`fig-05-trials-and-papers-gantt.md`](fig-05-trials-and-papers-gantt.md) | Mermaid | Gantt chart, two sections | `sec-03-independence-and-simultaneity.tex` |
| 6 | [`fig-06-criteria-grid.md`](fig-06-criteria-grid.md) | D2 | Grid with shaded status cells | `sec-04-what-a-standard-requires.tex` |

Diagrams and Graphviz are not used on day 2. Across the block each platform is
used exactly three times, and never twice in one day.

## The caption rule

| Figure | Line 1 | Line 2 | Spread |
|:--|:--|:--|:--|
| 4 | 86 characters | 84 characters | 2 |
| 5 | 77 characters | 79 characters | 2 |
| 6 | 85 characters | 82 characters | 3 |

Every caption sits `-0.60cm` below its frame, which leaves exactly 7.44 pt from
the frame's last rule to the first caption line.

## Rules every figure here obeys

| Rule | How it is met |
|:--|:--|
| No stick figures | No person is drawn; parties appear as labeled states, lanes, or columns |
| No rasters | Every figure is TikZ in the section file |
| No broken words | Hyphenation is switched off inside every node by the style |
| Quantities from sources | Every date in Figure 5 is a registry date or a deposit date |
| Readable source | One node per line; the grid's fills are one row per criterion |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| Root `README.md` release list | [`../../../../README.md`](../../../../README.md) | The company's dates in Figure 5 |
| ClinicalTrials.gov NCT05379985, NCT06625320, NCT07252232 | Public registry | The developer's bars in Figure 5 |
| `../briefs/brief-02-what-an-external-standard-requires.md` | [`../briefs`](../briefs) | The eight criteria in Figure 6 |
| `../packet/fundstyle.sty` | [`../packet`](../packet) | The `uml*`, `mm*` and `d2*` vocabularies |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
