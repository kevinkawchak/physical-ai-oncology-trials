# 01Oct26 / diagrams - three figure specifications (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-4B2A7B.svg)](..)
[![Figures](https://img.shields.io/badge/Figures-10%20to%2012-4B2A7B.svg)](.)
[![Platforms](https://img.shields.io/badge/Platforms-D2%2C%20PlantUML%2C%20Diagrams-6C757D.svg)](#the-three-figures)
[![Spacing](https://img.shields.io/badge/Caption%20spacing--0.60cm-6C757D.svg)](#the-caption-rule)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

One specification per figure: the native source on its own platform, the TikZ
construction that reproduces it in the packet, the exact caption, and the
section that uses it.

## The three figures

| # | File | Platform | Native construct | Used in |
|:--|:--|:--|:--|:--|
| 10 | [`fig-10-nsf-funding-ladder.md`](fig-10-nsf-funding-ladder.md) | D2 | Layered stack, each layer gated by the one below | `sec-01-the-inquiry-and-the-gate.tex` |
| 11 | [`fig-11-pitch-activity.md`](fig-11-pitch-activity.md) | PlantUML | Activity diagram with a fork, a join and a connector | `sec-02-action-register.tex` |
| 12 | [`fig-12-verification-infrastructure.md`](fig-12-verification-infrastructure.md) | Diagrams | Clustered infrastructure, one cluster nested | `sec-05-merit-review.tex` |

Mermaid and Graphviz are not used on day 4.

## The caption rule

| Figure | Line 1 | Line 2 | Spread |
|:--|:--|:--|:--|
| 10 | 84 characters | 81 characters | 3 |
| 11 | 82 characters | 84 characters | 2 |
| 12 | 85 characters | 84 characters | 1 |

Every caption sits `-0.60cm` below its frame, which leaves exactly 7.44 pt from
the frame's last rule to the first caption line.

## Rules every figure here obeys

| Rule | How it is met |
|:--|:--|
| No stick figures | People appear as labeled boxes and glyph tiles, never as drawn figures |
| No rasters | Every figure is TikZ in the section file |
| No broken words | Hyphenation is switched off inside every node by the style |
| No crossings through labels | The sealed link leaves from beneath its label; the nested cluster's title sits at its right |
| Readable source | One node per line; the gate ticks of Figure 10 are one loop |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `../briefs/brief-01-the-eligibility-gate.md` | [`../briefs`](../briefs) | The layers and gates in Figure 10 |
| `../forms/form-01-nsf-project-pitch.md`, `../forms/form-02-research-gov-organization.md` | [`../forms`](../forms) | The two branches in Figure 11 |
| `../emails/email-03-nairr-pilot-resources.txt` | [`../emails`](../emails) | The two constraints drawn in Figure 12 |
| `trial-protocol/`, `trial-ind/` | repository root | The stops, force limits and boundary in Figure 12 |
| `../packet/fundstyle.sty` | [`../packet`](../packet) | The `d2*`, `uml*` and `dg*` vocabularies |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
