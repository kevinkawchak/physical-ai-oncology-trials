# 07Sep26 / packet - The Staged Queue (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-5B3A5E.svg)](..)
[![Accent](https://img.shields.io/badge/Accent-Slate%20Plum%20%235B3A5E-5B3A5E.svg)](fundstyle.sty)
[![Sections](https://img.shields.io/badge/Sections-7-6C757D.svg)](sections)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-6C757D.svg)](sections)
[![Bibliography](https://img.shields.io/badge/Bibliography-56%20entries-6C757D.svg)](references.bib)
[![Compiler](https://img.shields.io/badge/pdfLaTeX%20%2B%20BibTeX-0%20errors-9AA1A8.svg)](#compile-record)
[![Sent](https://img.shields.io/badge/Sent%20or%20entered-nothing-9AA1A8.svg)](.)
[![DOI](https://img.shields.io/badge/DOI-none%20asserted-9AA1A8.svg)](.)

The record of a day on which no counterparty could act, and the queue it
produced. §1 and §5 are the two sections a regulator would be sent; the rest is
internal and is attached to no letter.

## The files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | The shell: cover, badges, decision strip, notices, contents, seven `\input` lines |
| [`fundstyle.sty`](fundstyle.sty) | The style, in the Slate Plum palette. Differs from days 1 to 3 only in the six hex values of its palette block |
| [`references.bib`](references.bib) | 56 entries, every one carrying a resolvable target |
| [`sections/`](sections) | `sec-00` through `sec-06`, one file per section, per Rule 6 |
| `main.pdf` | The compiled packet |
| `07Sep26-packet-LaTeX.zip` | Everything above except the PDF, ready for Overleaf |

## The seven sections

| § | File | What it carries | Floats |
|:--|:--|:--|:--|
| 0 | `sec-00-front.tex` | Abstract, the nothing-sent notice, and the reading path | none |
| 1 | `sec-01-the-closed-day.tex` | Why a closed day has this shape, and the two lanes it forces | Figure 10, Table 16 |
| 2 | `sec-02-action-register.tex` | The release list, the queued orders, and the single approval step | Table 20 |
| 3 | `sec-03-data-room.tex` | Nine folders as typed records, with three access classes | Figure 11, Table 17 |
| 4 | `sec-04-recognition-letters.tex` | What the three letters are, what they are not, and the one place they are mentioned | Table 18 |
| 5 | `sec-05-diligence-and-stops.tex` | Twenty-two questions, four with no answer, and the failure combinations | Figure 12, Table 19 |
| 6 | `sec-06-references.tex` | Positioning, method, and the reference list | none |

## Which sections go to whom

| Recipient | Sections | Why |
|:--|:--|:--|
| FDA Office of Combination Products | 1 and 5 only | The system description and its stop conditions. The capital and outreach sections are not relevant to a classification request |
| A district congressional office | None. The letter is one page | A first letter to a district office should not carry a thirteen-page attachment |
| An investor's analyst, under a confidentiality agreement | 3 and 5 | The data room index and the diligence bank |
| The chief executive | All seven | It is his approval the day asks for |

## How to compile

Upload [`07Sep26-packet-LaTeX.zip`](.) to Overleaf, set the compiler to pdfLaTeX,
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

## The release discipline this packet records

| Rule | How the packet honors it |
|:--|:--|
| Nothing is sent or entered on a closed day | The cover, §0 and §2 each say so; every letter file opens with a `HOLD FOR RELEASE` line |
| Nothing is shown as done | Figure 10's held lane is drawn in the pale shade and labeled; Figure 12's root is a condition, not an event |
| The recognition letters are stated once and precisely | §4 and Table 18, using the approved wording verbatim |
| Unanswered questions are named | §5 marks four of twenty-two as having no answer, and says what each waits on |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../../move-in`](../../../move-in) | §4 and Table 18, the approved wording verbatim |
| `final-move-in/sections/sec-00-front.tex` | [`../../../move-in`](../../../move-in) | §3, the company record rows |
| `final-capital/sections/sec-09-risks-and-limits.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | §5, Figure 12 and Table 19 |
| `final-capital/sections/sec-10-build-method.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | §3, the custody paragraph |
| `UC-San-Diego/priority-steps.md` §10 and §11 | [`../../../potential-partners`](../../../potential-partners) | §1 and §2, the Pre-Request for Designation row |
| `../briefs/brief-01`, `brief-02`, `brief-03` | This day | §3, §4 and §5 |
| `../diagrams/fig-10`, `fig-11`, `fig-12` | This day | The three figures |
| `../investing/capital-04-queued-orders.md` | This day | §2 and Table 20 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
