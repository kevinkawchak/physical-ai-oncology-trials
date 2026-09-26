# 02Oct26 / diagrams - three figure specifications (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-5%20of%205-8C5A12.svg)](..)
[![Figures](https://img.shields.io/badge/Figures-13%20to%2015-8C5A12.svg)](.)
[![Platforms](https://img.shields.io/badge/Platforms-Graphviz%2C%20Diagrams%2C%20Mermaid-6C757D.svg)](#the-three-figures)
[![Spacing](https://img.shields.io/badge/Caption%20spacing--0.60cm-6C757D.svg)](#the-caption-rule)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

One specification per figure: the native source on its own platform, the TikZ
construction that reproduces it in the packet, the exact caption, and the
section that uses it. These are the last three of the block's fifteen figures.

## The three figures

| # | File | Platform | Native construct | Used in |
|:--|:--|:--|:--|:--|
| 13 | [`fig-13-week-record-chain.md`](fig-13-week-record-chain.md) | Graphviz | Record nodes in ranks, one per weekday | `sec-01-the-week-in-one-record.tex` |
| 14 | [`fig-14-cadence-topology.md`](fig-14-cadence-topology.md) | Diagrams | Clustered topology with one loop | `sec-02-action-register.tex` |
| 15 | [`fig-15-follow-up-rule.md`](fig-15-follow-up-rule.md) | Mermaid | Top-down flowchart with five decisions | `sec-03-the-follow-up-rule.tex` |

D2 and PlantUML are not used on day 5. Across the block, each of the five
platforms draws exactly three figures.

## The caption rule

| Figure | Line 1 | Line 2 | Spread |
|:--|:--|:--|:--|
| 13 | 85 characters | 88 characters | 3 |
| 14 | 82 characters | 81 characters | 1 |
| 15 | 81 characters | 80 characters | 1 |

Every caption sits `-0.60cm` below its frame, which leaves exactly 7.44 pt from
the frame's last rule to the first caption line.

## Rules every figure here obeys

| Rule | How it is met |
|:--|:--|
| No stick figures | People appear as labeled glyph tiles and boxes, never as drawn figures |
| No rasters | Every figure is TikZ in the section file |
| No broken words | Hyphenation is switched off inside every node by the style |
| No crossings through labels | Figures 13 and 14 route their edges on orthogonal channels |
| Readable source | One node per line; the repeated ranks of Figure 13 are two loops |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| Every letter of days 1 to 4 | [`../..`](../..) | The ranks of Figure 13 and the counts of Figure 15 |
| `../briefs/brief-03-the-weekly-cadence.md` | [`../briefs`](../briefs) | The places and times of Figure 14, and the rule of Figure 15 |
| `30Sep26/briefs/brief-03-the-early-rise-weekday.md` | [`../../30Sep26/briefs`](../../30Sep26/briefs) | The weekday blocks in Figure 14 |
| `../packet/fundstyle.sty` | [`../packet`](../packet) | The `gv*`, `dg*` and `mm*` vocabularies |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
