# 04Sep26 / packet - The Site and Partner Package (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-3%20of%205-2F5D3A.svg)](..)
[![Accent](https://img.shields.io/badge/Accent-Cypress%20Green%20%232F5D3A-2F5D3A.svg)](fundstyle.sty)
[![Sections](https://img.shields.io/badge/Sections-7-6C757D.svg)](sections)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-6C757D.svg)](sections)
[![Bibliography](https://img.shields.io/badge/Bibliography-56%20entries-6C757D.svg)](references.bib)
[![Compiler](https://img.shields.io/badge/pdfLaTeX%20%2B%20BibTeX-0%20errors-9AA1A8.svg)](#compile-record)
[![Agreements](https://img.shields.io/badge/Agreements-none-9AA1A8.svg)](.)
[![DOI](https://img.shields.io/badge/DOI-none%20asserted-9AA1A8.svg)](.)

The document that accompanies the five letters in [`../emails`](../emails),
written so that a site principal investigator, a trials office, or a foundation
reviewer can decide from it alone whether a meeting is worth placing.

## The files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | The shell: cover, badges, decision strip, notices, contents, seven `\input` lines |
| [`fundstyle.sty`](fundstyle.sty) | The style, in the Cypress Green palette. Differs from days 1 and 2 only in the six hex values of its palette block |
| [`references.bib`](references.bib) | 56 entries, every one carrying a resolvable target |
| [`sections/`](sections) | `sec-00` through `sec-06`, one file per section, per Rule 6 |
| `main.pdf` | The compiled packet |
| `04Sep26-packet-LaTeX.zip` | Everything above except the PDF, ready for Overleaf |

## The seven sections

| § | File | What it carries | Floats |
|:--|:--|:--|:--|
| 0 | `sec-00-front.tex` | Abstract, the no-agreement notice, and the reading path | none |
| 1 | `sec-01-two-routes.tex` | Two institutions, seven capability criteria, and the disclosure rule between them | Figure 8, Table 12 |
| 2 | `sec-02-action-register.tex` | The five questions a first meeting settles, and the single approval step | Table 11 |
| 3 | `sec-03-obligations.tex` | Who is responsible for what, and the three corrections every letter carries | Figure 7, Table 15 |
| 4 | `sec-04-foundations.tex` | The funnel, and three foundations by cycle, ceiling, and fit | Figure 9, Table 13 |
| 5 | `sec-05-startup-costs.tex` | The site start-up line against the annual direct cost, with nothing committed | Table 14 |
| 6 | `sec-06-references.tex` | Positioning, method, and the reference list | none |

## How to compile

Upload [`04Sep26-packet-LaTeX.zip`](.) to Overleaf, set the compiler to pdfLaTeX,
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
| Pages | 12 |
| Figures | 3, all TikZ |
| Tables | 5, all at the exact body measure |
| Raster images | 0 |

## The no-agreement discipline this packet is written under

| Rule | How the packet honors it |
|:--|:--|
| No institution is a partner, sponsor, site, or endorser | The cover, §0 and §6 each say so, and Figure 7's cluster titles say it inside the figure |
| Nothing is claimed as agreed | Figure 7 dashes every developer edge and labels the site cluster as conditional |
| No configuration is asserted | No manufacturer, model, arm count, or software version appears anywhere |
| Neither institution is ranked | §1 names both, discloses the parallel approach, and describes neither as preferred |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `UC-San-Diego/README.md` | [`../../../potential-partners`](../../../potential-partners) | §2, the five feasibility questions as Table 11 |
| `UC-San-Diego/priority-steps.md` §3, §4, §6, §8, §12 | [`../../../potential-partners`](../../../potential-partners) | §1, §2, §3 and §5 |
| `Scripps/priority-steps.md` §2, §6, §10 | [`../../../potential-partners`](../../../potential-partners) | §1 and Table 12 |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../move-in`](../../../move-in) | §5, Table 14, and Figure 8's function list |
| `applications/app-06-fnih-amp/` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | §4 and Table 13 |
| `../briefs/brief-01`, `brief-02` | This day | §2 and §1 |
| `../diagrams/fig-07`, `fig-08`, `fig-09` | This day | The three figures |
| `../../03Sep26/briefs/brief-03-use-of-proceeds.md` | [`../../03Sep26`](../../03Sep26) | §5, the $420,000 line |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
