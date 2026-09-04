# 08Sep26 / packet - The Execution Record (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-5%20of%205-8A4B2A.svg)](..)
[![Accent](https://img.shields.io/badge/Accent-Ember%20Rust%20%238A4B2A-8A4B2A.svg)](fundstyle.sty)
[![Sections](https://img.shields.io/badge/Sections-7-6C757D.svg)](sections)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-6C757D.svg)](sections)
[![Bibliography](https://img.shields.io/badge/Bibliography-56%20entries-6C757D.svg)](references.bib)
[![Compiler](https://img.shields.io/badge/pdfLaTeX%20%2B%20BibTeX-0%20errors-9AA1A8.svg)](#compile-record)
[![Closing table](https://img.shields.io/badge/Table%2025-the%20five--day%20record-8A4B2A.svg)](#the-closing-table)
[![DOI](https://img.shields.io/badge/DOI-none%20asserted-9AA1A8.svg)](.)

The closing document of the five-day block. Its last table is the one page a
person who missed the week can read to know what the week did.

## The files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | The shell: cover, badges, decision strip, notices, contents, seven `\input` lines |
| [`fundstyle.sty`](fundstyle.sty) | The style, in the Ember Rust palette. Differs from days 1 to 4 only in the six hex values of its palette block |
| [`references.bib`](references.bib) | 56 entries, every one carrying a resolvable target |
| [`sections/`](sections) | `sec-00` through `sec-06`, one file per section, per Rule 6 |
| `main.pdf` | The compiled packet |
| `08Sep26-packet-LaTeX.zip` | Everything above except the PDF, ready for Overleaf |

## The seven sections

| § | File | What it carries | Floats |
|:--|:--|:--|:--|
| 0 | `sec-00-front.tex` | Abstract, what the block produced, and the reading path | none |
| 1 | `sec-01-the-release.tex` | The six checks, the session sequence, what was and was not released | Figure 13, Table 21 |
| 2 | `sec-02-action-register.tex` | Execution and settlement, the fill record, the approval step | Table 24 |
| 3 | `sec-03-pipeline.tex` | Seventeen open items, four that unblock, four that cannot be chased | Figure 15, Table 22 |
| 4 | `sec-04-cadence.tex` | Five standing functions, the carry rule, and what is not in the cadence | Figure 14, Table 23 |
| 5 | `sec-05-the-record.tex` | The five-day record, one row per action | Table 25 |
| 6 | `sec-06-references.tex` | Positioning, method, and the reference list | none |

## The closing table

Table 25 is written so that **no row depends on a file being read.** Each names
what was sent, to whom, and what it asked, in one line. It is the table to hand
somebody who joins the program in a month and needs to know what was already
asked of whom, so that they do not ask it again.

## How to compile

Upload [`08Sep26-packet-LaTeX.zip`](.) to Overleaf, set the compiler to pdfLaTeX,
and run:

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Compile record

| Measure | Value |
|:--|:--|
| Errors | 0 |
| Overfull boxes | 0 |
| Underfull boxes | 0 |
| Undefined citations | 0 |
| Undefined references | 0 |
| Pages | 14 |
| Figures | 3, all TikZ |
| Tables | 5, all at the exact body measure |
| Raster images | 0 |

## The five-day block, measured

| Measure | Across all five days |
|:--|:--|
| Letters in `.txt` | 24 |
| Technical briefs in `.md` | 13 |
| Form and filing packs in `.md` | 9 |
| Capital instruction sets in `.md` | 5 |
| Figure specifications in `.md` | 15 |
| Compiled packets | 5, each 7 sections |
| Figures | 15, three per packet, three per platform |
| Tables | 25, five per packet |
| Raster images | 0 |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `final-capital/sections/sec-05-twelve-milestones.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | §3, Table 22 and Figure 15's layers |
| `final-capital/sections/sec-07-operating-plan.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | §4, Table 23 and Figure 14's standing functions |
| `UC-San-Diego/priority-steps.md` §4 | [`../../../potential-partners`](../../../potential-partners) | §1, the three-business-day interval |
| `../../07Sep26/investing/capital-04-queued-orders.md` | [`../../07Sep26`](../../07Sep26) | §1 and §2, the six orders |
| `../../07Sep26/README.md` | [`../../07Sep26`](../../07Sep26) | §1, Table 21 |
| `../briefs/brief-01`, `brief-02` | This day | §3 and §4 |
| `../diagrams/fig-13`, `fig-14`, `fig-15` | This day | The three figures |
| `../investing/capital-05-execution-and-settlement.md` | This day | §2 and Table 24 |
| Days 1 to 4 of this block | [`../..`](../..) | §5, Table 25 in full |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
