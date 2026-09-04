# 03Sep26 / packet - The Private Capital Bridge (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-2%20of%205-1B3A5C.svg)](..)
[![Accent](https://img.shields.io/badge/Accent-Harbor%20Navy%20%231B3A5C-1B3A5C.svg)](fundstyle.sty)
[![Sections](https://img.shields.io/badge/Sections-7-6C757D.svg)](sections)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-6C757D.svg)](sections)
[![Bibliography](https://img.shields.io/badge/Bibliography-56%20entries-6C757D.svg)](references.bib)
[![Compiler](https://img.shields.io/badge/pdfLaTeX%20%2B%20BibTeX-0%20errors-9AA1A8.svg)](#compile-record)
[![Offer](https://img.shields.io/badge/Offer%20or%20solicitation-none-9AA1A8.svg)](.)
[![DOI](https://img.shields.io/badge/DOI-none%20asserted-9AA1A8.svg)](.)

The document that accompanies the five letters in [`../emails`](../emails). It
describes three instruments under consideration and describes no offering.

## The files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | The shell: cover, badges, decision strip, notices, contents, seven `\input` lines |
| [`fundstyle.sty`](fundstyle.sty) | The style, in the Harbor Navy palette. Differs from day 1 only in the six hex values of its palette block |
| [`references.bib`](references.bib) | 56 entries, every one carrying a resolvable target |
| [`sections/`](sections) | `sec-00` through `sec-06`, one file per section, per Rule 6 |
| `main.pdf` | The compiled packet |
| `03Sep26-packet-LaTeX.zip` | Everything above except the PDF, ready for Overleaf |

## The seven sections

| § | File | What it carries | Floats |
|:--|:--|:--|:--|
| 0 | `sec-00-front.tex` | Abstract, the securities notice, and the reading path | none |
| 1 | `sec-01-the-gap.tex` | Why $2,104,000 and not another number, and the seven proceeds lines | Table 6 |
| 2 | `sec-02-action-register.tex` | The five letters, the signing order, and the single approval step | Figure 6, Table 7 |
| 3 | `sec-03-instruments.tex` | Three instruments on eight attributes, and the row that decides it | Figure 4, Table 8 |
| 4 | `sec-04-firewall-and-position.tex` | The 21 CFR part 54 triggers, the tranche structure, and the four guards | Figure 5, Table 9 |
| 5 | `sec-05-reserve.tex` | The reserve on two branches, and what stays excluded on both | Table 10 |
| 6 | `sec-06-references.tex` | Positioning, method, and the reference list | none |

## How to compile

Upload [`03Sep26-packet-LaTeX.zip`](.) to Overleaf, set the compiler to pdfLaTeX,
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
| Pages | 13 |
| Figures | 3, all TikZ |
| Tables | 5, all at the exact body measure |
| Raster images | 0 |

## The securities discipline this packet is written under

| Rule | How the packet honors it |
|:--|:--|
| No offer or solicitation | The cover, §0 and §6 each carry the sentence. No section states a term as available |
| No general solicitation | No valuation, cap, discount, minimum, or closing date appears anywhere, including in the figures |
| Figures safe out of context | Figures 4, 5 and 6 carry no amount at all, because a figure is the easiest thing in a document to screenshot |
| One consistent size | $2,104,000 is described as the gap under comparison, never as an offering amount |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `final-capital/sections/sec-04-capital-bridge.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | §4, Table 9, and Figure 5's state names |
| `final-capital/sections/sec-03-gate-and-programme.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | §1 and Table 6 |
| `final-capital/capstyle.sty` | [`../../../capitalization-plan`](../../../capitalization-plan) | `fundstyle.sty` |
| `UC-San-Diego/priority-steps.md` §2 and §12 | [`../../../potential-partners`](../../../potential-partners) | §2's developer letter row and §4's open questions |
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../../move-in`](../../../move-in) | §1's federal versus non-federal separation |
| `../briefs/brief-01`, `brief-02`, `brief-03` | This day | §3, §4 and §1 respectively |
| `../diagrams/fig-04`, `fig-05`, `fig-06` | This day | The three figures |
| `../investing/capital-02-corporate-reserve-allocation.md` | This day | §5 and Table 10 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
