# 30Sep26 / packet - The Structured Team (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-3%20of%205-7A1F2B.svg)](..)
[![Accent](https://img.shields.io/badge/Accent-Mission%20Oxblood%20%237A1F2B-7A1F2B.svg)](fundstyle.sty)
[![Sections](https://img.shields.io/badge/Sections-7-6C757D.svg)](sections)
[![Figures](https://img.shields.io/badge/Figures-7%20to%209-6C757D.svg)](../diagrams)
[![Tables](https://img.shields.io/badge/Tables-11%20to%2015-6C757D.svg)](#structure)
[![References](https://img.shields.io/badge/References-39%2C%20all%20linked-6C757D.svg)](references.bib)
[![Compile](https://img.shields.io/badge/pdfLaTeX-0%20errors%2C%200%20overfull-9AA1A8.svg)](#compile-record)
[![Overleaf](https://img.shields.io/badge/Overleaf-zip%20ready-9AA1A8.svg)](30Sep26-packet-LaTeX.zip)

The compiled document of day 3. It is the attachment to letters 1, 2 and 3: to
the Office of Science and Technology Policy, to the Genesis Mission partnerships
office, and to the Moores Clinical Trials Office. It quotes the September 12,
2026 message verbatim, turns its three priorities into rules a counterparty can
check, and places the company inside four structures as a contributor.

## Structure

| § | File | Title | Figure | Table |
|:--|:--|:--|:--|:--|
| Front | [`sections/sec-00-front.tex`](sections/sec-00-front.tex) | Abstract and how to read this |  |  |
| §1 | [`sections/sec-01-the-correspondence.tex`](sections/sec-01-the-correspondence.tex) | The Message, and the Three Rules It Became |  | Table 11 |
| §2 | [`sections/sec-02-action-register.tex`](sections/sec-02-action-register.tex) | The Action Register | Figure 9 | Table 12 |
| §3 | [`sections/sec-03-the-structure-map.tex`](sections/sec-03-the-structure-map.tex) | The Structure Map | Figure 7 | Table 13 |
| §4 | [`sections/sec-04-the-early-rise-weekday.tex`](sections/sec-04-the-early-rise-weekday.tex) | The Early-Rise Weekday | Figure 8 | Table 14 |
| §5 | [`sections/sec-05-authority-inside-the-trial.tex`](sections/sec-05-authority-inside-the-trial.tex) | Authority Inside the Robotic Phase 1 |  | Table 15 |
| §6 | [`sections/sec-06-references.tex`](sections/sec-06-references.tex) | Positioning, Method, and References |  |  |

## Files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | The cover, the author block with both disclaimers verbatim, the keywords, the contents, and one `\input` per section |
| [`fundstyle.sty`](fundstyle.sty) | The style; this copy carries the Mission Oxblood palette |
| [`references.bib`](references.bib) | 39 entries, each with a clickable url, and a doi wherever one exists |
| [`sections/`](sections) | Seven section files, one per section |
| `main.pdf` | The compiled document |
| `30Sep26-packet-LaTeX.zip` | `main.tex`, `fundstyle.sty`, `references.bib` and the seven section files, ready to upload to Overleaf |

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
| `fundaccent` | `#7A1F2B` | Headings, table header rows, the cover block, the day's key nodes |
| `fundmid` | `#A85A63` | The cover band and secondary fills |
| `fundpale` | `#F3E4E6` | The approval box and light fills |
| `fundgray` family | `#6C757D`, `#9AA1A8`, `#CED4DA`, `#E9ECEF` | Process, closed states, legends |

Mission Oxblood is the deep red of the tile roofs of the old mission buildings
along the San Diego coast, a color that stands for an order older than anyone in
it. It is given to the day about working inside established structures.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `08Sep26/packet/fundstyle.sty` | [`../../../auto-fund`](../../../auto-fund) | Every mechanism of `fundstyle.sty` below the palette block |
| [`../../prompts/prompt-auto-fund-2.md`](../../prompts/prompt-auto-fund-2.md) | This block | The September 12 message in §1 |
| `02Sep26/emails/email-06-nci-ctep-gore-reply.txt` | [`../../../auto-fund`](../../../auto-fund) | The first rule, already kept once, in §1 |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../move-in`](../../../move-in) | Table 15 and the firewall in §5 |
| `08Sep26/briefs/brief-02-weekly-cadence.md` | [`../../../auto-fund`](../../../auto-fund) | The weekday themes in Figure 8 |
| `../briefs/` | [`../briefs`](../briefs) | Tables 11, 13 and 14 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
