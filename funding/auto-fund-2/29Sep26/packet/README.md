# 29Sep26 / packet - The External Standard Request (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-2%20of%205-34495E.svg)](..)
[![Accent](https://img.shields.io/badge/Accent-Harbor%20Slate%20%2334495E-34495E.svg)](fundstyle.sty)
[![Sections](https://img.shields.io/badge/Sections-7-6C757D.svg)](sections)
[![Figures](https://img.shields.io/badge/Figures-4%20to%206-6C757D.svg)](../diagrams)
[![Tables](https://img.shields.io/badge/Tables-6%20to%2010-6C757D.svg)](#structure)
[![References](https://img.shields.io/badge/References-37%2C%20all%20linked-6C757D.svg)](references.bib)
[![Compile](https://img.shields.io/badge/pdfLaTeX-0%20errors%2C%200%20overfull-9AA1A8.svg)](#compile-record)
[![Overleaf](https://img.shields.io/badge/Overleaf-zip%20ready-9AA1A8.svg)](29Sep26-packet-LaTeX.zip)

The compiled document of day 2. It is the attachment to letters 1 and 2: the
reply to the SEC San Francisco Regional Office, and the routing letter to the
FDA's Oncology AI Program. It answers the office's update with a record rather
than an argument, and says plainly what the record does not yet show.

## Structure

| § | File | Title | Figure | Table |
|:--|:--|:--|:--|:--|
| Front | [`sections/sec-00-front.tex`](sections/sec-00-front.tex) | Abstract and how to read this |  |  |
| §1 | [`sections/sec-01-the-update.tex`](sections/sec-01-the-update.tex) | The Update, and the Record Behind the Request |  | Table 6 |
| §2 | [`sections/sec-02-action-register.tex`](sections/sec-02-action-register.tex) | The Action Register | Figure 4 | Table 7 |
| §3 | [`sections/sec-03-independence-and-simultaneity.tex`](sections/sec-03-independence-and-simultaneity.tex) | Independence and Simultaneity | Figure 5 | Table 8 |
| §4 | [`sections/sec-04-what-a-standard-requires.tex`](sections/sec-04-what-a-standard-requires.tex) | What an External Standard Requires | Figure 6 | Table 9 |
| §5 | [`sections/sec-05-the-evidence.tex`](sections/sec-05-the-evidence.tex) | The Evidence Behind the Robotic Phase 1 |  | Table 10 |
| §6 | [`sections/sec-06-references.tex`](sections/sec-06-references.tex) | Positioning, Method, and References |  |  |

## Files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | The cover, the author block with both disclaimers verbatim, the keywords, the contents, and one `\input` per section |
| [`fundstyle.sty`](fundstyle.sty) | The style; this copy carries the Harbor Slate palette |
| [`references.bib`](references.bib) | 37 entries, each with a clickable url, and a doi wherever one exists |
| [`sections/`](sections) | Seven section files, one per section |
| `main.pdf` | The compiled document |
| `29Sep26-packet-LaTeX.zip` | `main.tex`, `fundstyle.sty`, `references.bib` and the seven section files, ready to upload to Overleaf |

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
| `fundaccent` | `#34495E` | Headings, table header rows, the cover block, the day's key nodes |
| `fundmid` | `#6B7F94` | The cover band and secondary fills |
| `fundpale` | `#E3E8EE` | The approval box and light fills |
| `fundgray` family | `#6C757D`, `#9AA1A8`, `#CED4DA`, `#E9ECEF` | Process, closed states, legends |

Harbor Slate is the blue-gray of the harbor side of the city on a marine-layer
morning; it is given to the day that writes to a regulator, where the tone is
measured and the color should be too.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `08Sep26/packet/fundstyle.sty` | [`../../../auto-fund`](../../../auto-fund) | Every mechanism of `fundstyle.sty` below the palette block |
| Root `README.md` release list | [`../../../../README.md`](../../../../README.md) | Table 6 and the company rows of Table 8 |
| `daraxonrasib-llm-story.md` | [`../../..`](../../..) | §1 and §3 |
| `tripartisan-llm-support.md` | [`../../..`](../../..) | §2 and §4 |
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Table 10 and §5 |
| `../briefs/brief-02-what-an-external-standard-requires.md` | [`../briefs`](../briefs) | Table 9 and Figure 6 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
