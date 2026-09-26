# 02Oct26 / packet - The Week's Record (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-5%20of%205-8C5A12.svg)](..)
[![Accent](https://img.shields.io/badge/Accent-Sunset%20Cliffs%20Amber%20%238C5A12-8C5A12.svg)](fundstyle.sty)
[![Sections](https://img.shields.io/badge/Sections-7-6C757D.svg)](sections)
[![Figures](https://img.shields.io/badge/Figures-13%20to%2015-6C757D.svg)](../diagrams)
[![Tables](https://img.shields.io/badge/Tables-21%20to%2025-6C757D.svg)](#structure)
[![References](https://img.shields.io/badge/References-41%2C%20all%20linked-6C757D.svg)](references.bib)
[![Compile](https://img.shields.io/badge/pdfLaTeX-0%20errors%2C%200%20overfull-9AA1A8.svg)](#compile-record)
[![Overleaf](https://img.shields.io/badge/Overleaf-zip%20ready-9AA1A8.svg)](02Oct26-packet-LaTeX.zip)

The compiled document of day 5, and the last of the block. It is the attachment
to letter 1, to the NCI SBIR Development Center. It records every letter of the
week against the follow-up rule, explains why only three follow-ups leave today,
states the robotic Phase 1 in numbers a funder can check, and lists the funding
routes the week opened with the next step on each.

## Structure

| § | File | Title | Figure | Table |
|:--|:--|:--|:--|:--|
| Front | [`sections/sec-00-front.tex`](sections/sec-00-front.tex) | Abstract and how to read this |  |  |
| §1 | [`sections/sec-01-the-week-in-one-record.tex`](sections/sec-01-the-week-in-one-record.tex) | The Week in One Record | Figure 13 | Table 21 |
| §2 | [`sections/sec-02-action-register.tex`](sections/sec-02-action-register.tex) | The Action Register | Figure 14 | Table 22 |
| §3 | [`sections/sec-03-the-follow-up-rule.tex`](sections/sec-03-the-follow-up-rule.tex) | The Follow-Up Rule | Figure 15 | Table 23 |
| §4 | [`sections/sec-04-the-robotic-phase-1-in-numbers.tex`](sections/sec-04-the-robotic-phase-1-in-numbers.tex) | The Robotic Phase 1, in Numbers |  | Table 24 |
| §5 | [`sections/sec-05-the-routes-opened.tex`](sections/sec-05-the-routes-opened.tex) | The Routes Opened This Week |  | Table 25 |
| §6 | [`sections/sec-06-references.tex`](sections/sec-06-references.tex) | Positioning, Method, and References |  |  |

## Files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | The cover, the author block with both disclaimers verbatim, the keywords, the contents, and one `\input` per section |
| [`fundstyle.sty`](fundstyle.sty) | The style; this copy carries the Sunset Cliffs Amber palette |
| [`references.bib`](references.bib) | 41 entries, each with a clickable url, and a doi wherever one exists |
| [`sections/`](sections) | Seven section files, one per section |
| `main.pdf` | The compiled document |
| `02Oct26-packet-LaTeX.zip` | `main.tex`, `fundstyle.sty`, `references.bib` and the seven section files, ready to upload to Overleaf |

## How to compile

On Overleaf, upload the zip as a new project, set the compiler to pdfLaTeX, and
recompile. Locally:

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Every package the style loads ships with TeX Live and with Overleaf. No file
outside this directory is read, and no image file of any kind is read.

## Compile record

| Measure | Value |
|:--|:--|
| Pages | 10 |
| LaTeX errors | 0 |
| Overfull boxes | 0 |
| Underfull boxes | 0 |
| Undefined citations or references | 0 |
| BibTeX warnings | 0 |
| Captions at two lines and -0.60cm | 8 of 8 |

## The palette

| Token | Hex | Used for |
|:--|:--|:--|
| `fundaccent` | `#8C5A12` | Headings, table header rows, the cover block, the day's key nodes |
| `fundmid` | `#B98A45` | The cover band and secondary fills |
| `fundpale` | `#F4EADB` | The approval box and light fills |
| `fundgray` family | `#6C757D`, `#9AA1A8`, `#CED4DA`, `#E9ECEF` | Process, closed states, legends |

Sunset Cliffs Amber is the color of the sandstone at Sunset Cliffs in Point Loma,
lit by the last hour of the day. It is given to the day that closes the week.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `08Sep26/packet/fundstyle.sty` | [`../../../auto-fund`](../../../auto-fund) | Every mechanism of `fundstyle.sty` below the palette block |
| Every `emails/` and `linkedin/` file of days 1 to 4 | [`../..`](../..) | Tables 21 and 23, and Figures 13 and 15 |
| `final-protocol/sections/` | [`../../../../trial-protocol`](../../../../trial-protocol) | The design and safety rows of Table 24 |
| `final-ind/sections/sec-02-introduction.tex` | [`../../../../trial-ind`](../../../../trial-ind) | The simulation gate in Table 24 |
| `final-capital/sections/sec-03-gate-and-programme.tex`, `sec-06-clinical-evidence.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | The money and evidence rows of Table 24, and Table 25 |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../move-in`](../../../move-in) | The cost and staffing figures in §4 and §5 |
| `../briefs/`, `../forms/`, `../investing/` | [`../briefs`](../briefs), [`../forms`](../forms), [`../investing`](../investing) | Tables 22 and 25, and Figure 14 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
