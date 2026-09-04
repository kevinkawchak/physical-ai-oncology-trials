# 02Sep26 / diagrams - three figure specifications (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-1%20of%205-0E5C63.svg)](..)
[![Figures](https://img.shields.io/badge/Figures-3-0E5C63.svg)](.)
[![Platforms](https://img.shields.io/badge/Platforms-Mermaid%2C%20Graphviz%2C%20D2-6C757D.svg)](#the-three-figures)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)
[![Stick figures](https://img.shields.io/badge/Stick%20figures-none-9AA1A8.svg)](#no-stick-figures)

One specification per figure. Each carries the native source in the platform's
own language, the TikZ that renders it in the packet, the coordinates and the
pitch, the provenance of every number in it, and the caption exactly as printed.

## Why a specification and not just the TikZ

A figure that needs correcting is corrected in the `.tex`, but the person
correcting it needs to know what the figure was for and where its numbers came
from. The specification answers both without requiring the packet to be read.
The native source is included for the same reason: it is shorter than the TikZ,
so the shape of the figure can be checked in ten seconds before a coordinate is
touched.

## The three figures

| # | Platform | Native construct | Perspective no other figure in this day gives |
|:--|:--|:--|:--|
| 1 | Mermaid | Flowchart with a dated transition | What the approval changed in the ask, as one state change |
| 2 | Graphviz | Record nodes inside a cluster | The five federal mechanisms as records, each with its own state |
| 3 | D2 | Grid with an interval strip | The reserve against three claims on it, at the same scale |

## No stick figures

`fundstyle.sty` deletes the parent style's `\umlactor` macro rather than leaving
it unused, so a basic human stick figure cannot be drawn in these packets even by
accident. Where a person or a role has to appear, the figure uses a labeled box
or a tile carrying a vector pictogram.

## The invariants every figure obeys

| # | Invariant | Value |
|:--|:--|:--|
| 1 | Frame | `\begin{appfig}[platform / construct]`, centered, ruled, white ground |
| 2 | Caption spacing | `\vspace{-0.60cm}` between frame and caption; 7.44 pt from rule to first caption line |
| 3 | Caption lines | Exactly two, balanced within a small character spread |
| 4 | Caption measure | `0.94\linewidth`, centered on the body measure |
| 5 | Palette | The day accent, its two lighter shades, three grays, white. No near-black fill |
| 6 | Output | TikZ only. No raster, no external image file |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `mermaid/`, `graphviz/`, `d2/` specification format | [`../../../capitalization-plan`](../../../capitalization-plan) | The structure of all three files: perspective, native source, TikZ, value table, sources |
| `final-capital/capstyle.sty` | [`../../../capitalization-plan`](../../../capitalization-plan) | The `mm*`, `gv*` and `d2*` style names used in the TikZ blocks |
| `applications/emailed-source/README.md` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | Figure 2's five mechanism records and their contact dates |
| `final-capital/sections/sec-03-gate-and-programme.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Figure 3's money values |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
