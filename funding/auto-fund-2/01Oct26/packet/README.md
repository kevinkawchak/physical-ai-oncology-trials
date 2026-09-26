# 01Oct26 / packet - The Pilot Inquiry (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-4B2A7B.svg)](..)
[![Accent](https://img.shields.io/badge/Accent-Pancreatic%20Purple%20%234B2A7B-4B2A7B.svg)](fundstyle.sty)
[![Sections](https://img.shields.io/badge/Sections-7-6C757D.svg)](sections)
[![Figures](https://img.shields.io/badge/Figures-10%20to%2012-6C757D.svg)](../diagrams)
[![Tables](https://img.shields.io/badge/Tables-16%20to%2020-6C757D.svg)](#structure)
[![References](https://img.shields.io/badge/References-37%2C%20all%20linked-6C757D.svg)](references.bib)
[![Compile](https://img.shields.io/badge/pdfLaTeX-0%20errors%2C%200%20overfull-9AA1A8.svg)](#compile-record)
[![Overleaf](https://img.shields.io/badge/Overleaf-zip%20ready-9AA1A8.svg)](01Oct26-packet-LaTeX.zip)

The compiled document of day 4. It is the attachment to letter 1, to NSF's SBIR
program, and is referred to by letters 3 and 4. It answers the NSF inquiry about
a multi-million-dollar pilot exactly: the pilot's published gate, where the
company stands against it, the split that keeps NSF, NIH and private capital
from paying for the same work, and the Phase I that the Project Pitch opens.

## Structure

| § | File | Title | Figure | Table |
|:--|:--|:--|:--|:--|
| Front | [`sections/sec-00-front.tex`](sections/sec-00-front.tex) | Abstract and how to read this |  |  |
| §1 | [`sections/sec-01-the-inquiry-and-the-gate.tex`](sections/sec-01-the-inquiry-and-the-gate.tex) | The Inquiry, and the Gate in Front of the Pilot | Figure 10 | Table 16 |
| §2 | [`sections/sec-02-action-register.tex`](sections/sec-02-action-register.tex) | The Action Register | Figure 11 | Table 17 |
| §3 | [`sections/sec-03-how-the-program-is-split.tex`](sections/sec-03-how-the-program-is-split.tex) | How the Program Is Split |  | Table 18 |
| §4 | [`sections/sec-04-the-phase-i-objectives.tex`](sections/sec-04-the-phase-i-objectives.tex) | The Phase I Technical Objectives |  | Table 19 |
| §5 | [`sections/sec-05-merit-review.tex`](sections/sec-05-merit-review.tex) | The Merit Review Map | Figure 12 | Table 20 |
| §6 | [`sections/sec-06-references.tex`](sections/sec-06-references.tex) | Positioning, Method, and References |  |  |

## Files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | The cover, the author block with both disclaimers verbatim, the keywords, the contents, and one `\input` per section |
| [`fundstyle.sty`](fundstyle.sty) | The style; this copy carries the Pancreatic Purple palette |
| [`references.bib`](references.bib) | 37 entries, each with a clickable url, and a doi wherever one exists |
| [`sections/`](sections) | Seven section files, one per section |
| `main.pdf` | The compiled document |
| `01Oct26-packet-LaTeX.zip` | `main.tex`, `fundstyle.sty`, `references.bib` and the seven section files, ready to upload to Overleaf |

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
| `fundaccent` | `#4B2A7B` | Headings, table header rows, the cover block, the day's key nodes |
| `fundmid` | `#7E62A8` | The cover band and secondary fills |
| `fundpale` | `#EAE4F3` | The approval box and light fills |
| `fundgray` family | `#6C757D`, `#9AA1A8`, `#CED4DA`, `#E9ECEF` | Process, closed states, legends |

Pancreatic Purple is the color of the awareness ribbon for pancreatic cancer, the
disease the program exists for. It is given to the day that asks a science agency
to fund the verification work without which the trial cannot begin.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `08Sep26/packet/fundstyle.sty` | [`../../../auto-fund`](../../../auto-fund) | Every mechanism of `fundstyle.sty` below the palette block |
| [`../../inputs/README.md`](../../inputs/README.md) | This block | The NSF quantities in Table 16 and Figure 10 |
| `final-capital/sections/sec-04-capital-bridge.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | The private capital column of Table 18 and the money in §3 |
| `trial-protocol/`, `trial-ind/` | repository root | Table 19 and Figure 12 |
| `tripartisan-llm-support.md` | [`../../..`](../../..) | The verification practice in §4 |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../move-in`](../../../move-in) | The workforce row of Table 20 and the cost in §5 |
| `../briefs/`, `../forms/` | [`../briefs`](../briefs), [`../forms`](../forms) | Tables 16, 18, 19 and 20 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
