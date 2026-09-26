# 30Sep26 / diagrams - three figure specifications (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-3%20of%205-7A1F2B.svg)](..)
[![Figures](https://img.shields.io/badge/Figures-7%20to%209-7A1F2B.svg)](.)
[![Platforms](https://img.shields.io/badge/Platforms-Graphviz%2C%20D2%2C%20PlantUML-6C757D.svg)](#the-three-figures)
[![Spacing](https://img.shields.io/badge/Caption%20spacing--0.60cm-6C757D.svg)](#the-caption-rule)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

One specification per figure: the native source on its own platform, the TikZ
construction that reproduces it in the packet, the exact caption, and the
section that uses it.

## The three figures

| # | File | Platform | Native construct | Used in |
|:--|:--|:--|:--|:--|
| 7 | [`fig-07-authority-tree.md`](fig-07-authority-tree.md) | Graphviz | Directed tree, ranked top to bottom | `sec-03-the-structure-map.tex` |
| 8 | [`fig-08-weekday-timetable.md`](fig-08-weekday-timetable.md) | D2 | Grid of blocks against weekdays | `sec-04-the-early-rise-weekday.tex` |
| 9 | [`fig-09-letter-activity.md`](fig-09-letter-activity.md) | PlantUML | Activity diagram with three swimlanes | `sec-02-action-register.tex` |

Mermaid and Diagrams are not used on day 3.

## The caption rule

| Figure | Line 1 | Line 2 | Spread |
|:--|:--|:--|:--|
| 7 | 78 characters | 77 characters | 1 |
| 8 | 79 characters | 80 characters | 1 |
| 9 | 82 characters | 80 characters | 2 |

Every caption sits `-0.60cm` below its frame, which leaves exactly 7.44 pt from
the frame's last rule to the first caption line.

## Rules every figure here obeys

| Rule | How it is met |
|:--|:--|
| No stick figures | People appear as labeled roles in lanes and nodes, never as drawn figures |
| No rasters | Every figure is TikZ in the section file |
| No broken words | Hyphenation is switched off inside every node by the style |
| No edge label collisions | Labels on converging edges sit near their sources |
| Readable source | One node per line; the timetable's identical rows are one loop |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `../briefs/brief-02-the-structure-map.md` | [`../briefs`](../briefs) | The authorities in Figure 7 |
| `../briefs/brief-03-the-early-rise-weekday.md` | [`../briefs`](../briefs) | The blocks and themes in Figure 8 |
| `08Sep26/briefs/brief-02-weekly-cadence.md` | [`../../../auto-fund`](../../../auto-fund) | The weekday themes in Figure 8 |
| `../packet/fundstyle.sty` | [`../packet`](../packet) | The `gv*`, `d2*` and `uml*` vocabularies |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
