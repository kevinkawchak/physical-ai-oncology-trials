# 02Sep26 / packet - The Approval Dividend (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-1%20of%205-0E5C63.svg)](..)
[![Accent](https://img.shields.io/badge/Accent-Pacific%20Teal%20%230E5C63-0E5C63.svg)](fundstyle.sty)
[![Sections](https://img.shields.io/badge/Sections-7-6C757D.svg)](sections)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-6C757D.svg)](sections)
[![Bibliography](https://img.shields.io/badge/Bibliography-56%20entries-6C757D.svg)](references.bib)
[![Compiler](https://img.shields.io/badge/pdfLaTeX%20%2B%20BibTeX-0%20errors-9AA1A8.svg)](#compile-record)
[![Overleaf](https://img.shields.io/badge/Overleaf-zip%20included-9AA1A8.svg)](.)
[![DOI](https://img.shields.io/badge/DOI-none%20asserted-9AA1A8.svg)](.)

The document that accompanies the five letters in [`../emails`](../emails). It is
written so that a recipient who opens it without having read anything else in the
repository can decide, from this file alone, whether to take a call.

## The files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | The shell: cover, badges, decision strip, notices, contents, and seven `\input` lines |
| [`fundstyle.sty`](fundstyle.sty) | The style, in the Pacific Teal palette. Differs from the other four days only in the six hex values of its palette block |
| [`references.bib`](references.bib) | 56 entries, every one carrying a resolvable target |
| [`sections/`](sections) | `sec-00` through `sec-06`, one file per section, per Rule 6 |
| `main.pdf` | The compiled packet |
| `02Sep26-packet-LaTeX.zip` | Everything above except the PDF, ready to upload to Overleaf |

## The seven sections

| § | File | What it carries | Floats |
|:--|:--|:--|:--|
| 0 | `sec-00-front.tex` | Abstract, what this packet is for, and how to read it in six minutes | none |
| 1 | `sec-01-what-changed.tex` | The approval, the dated chronology, and the four things it does not change | Figure 1, Table 1 |
| 2 | `sec-02-action-register.tex` | Seven letters, the two replies received and what the first one changes, the single approval step | Figure 2, Table 2 |
| 3 | `sec-03-evidence.tex` | The six checkable quantities, each with its authors' stated limitation | Table 3 |
| 4 | `sec-04-capital.tex` | The corporate reserve as a four-rung ladder against a nine-month horizon | Figure 3, Table 4 |
| 5 | `sec-05-outreach-and-route.tex` | The SBIR route against the five-year program, and the delta between them | Table 5 |
| 6 | `sec-06-references.tex` | Positioning, method note, and the reference list | none |

## How to compile

On Overleaf, upload [`02Sep26-packet-LaTeX.zip`](.), set the compiler to
pdfLaTeX, and run:

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Locally the same four commands work with a TeX Live installation carrying
`titlesec`, `adjustbox`, `xltabular`, `changepage`, `ragged2e`, `needspace`,
`colortbl`, `enumitem`, PGF/TikZ, and the `urlbst` bibliography styles.

## Compile record

Measured on the source in this directory, with `bibtex` run between the first and
second `pdflatex` passes.

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

## The spacing invariant, stated once

Every float in this packet closes with the same two lines:

```latex
\end{appfig}    \vspace{-0.60cm}  \figcaption{...}
\end{apptable}  \vspace{-0.60cm}  \tabcap{...}
```

`appfig` and `apptable` each close with a rigid `\vskip 24.5pt`, and
`\figcaption` and `\tabcap` each open with `\nointerlineskip`, so the distance
from the last rule to the first caption line is exactly
`24.5pt - 0.60cm = 7.44pt` for every figure and every table in the packet,
floating or inline, whatever precedes or follows it on the page.

## Table geometry

Every table is `\begin{tabularx}{\textwidth}` and every fixed column is declared
`>{\raggedright\arraybackslash}p{...}`, so no cell shows a large interword gap
and no table is narrower or wider than the body measure. Column widths are cut to
the longest real cell in each column, so no row carries one deep cell beside four
shallow ones.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `final-capital/capstyle.sty` | [`../../../capitalization-plan`](../../../capitalization-plan) | `fundstyle.sty`: the five TikZ vocabularies, the figure frame, the clickable-DOI machinery |
| `final-new-trial/trialstyle.sty` | [`../../../../new-trial-system`](../../../../new-trial-system) | The `-0.60cm` invariant and the two-line caption convention |
| `final-capital/references.bib` | [`../../../capitalization-plan`](../../../capitalization-plan) | 30 of the 56 bibliography entries, carried unchanged |
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | §3 and Table 3 |
| `final-capital/sections/sec-03-gate-and-programme.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | §5 and Table 5 |
| `applications/app-05-nih-sbir-seed/` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | The $306,000 and $1,300,000 lines throughout |
| `applications/emailed-source/README.md` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | Figure 2 and Table 2 |
| `../diagrams/fig-01`, `fig-02`, `fig-03` | This day | The three figures, drawn from their own specifications |
| `../investing/capital-01-treasury-ladder.md` | This day | §4 and Table 4 |
| `daraxonrasib-llm-story.md` | [`../../..`](../../..) | §1, quoted rather than paraphrased |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
