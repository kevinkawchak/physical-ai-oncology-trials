# 28Sep26 / packet - The Unsolicited Applicants (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-1%20of%205-2E5E4E.svg)](..)
[![Accent](https://img.shields.io/badge/Accent-Torrey%20Pine%20%232E5E4E-2E5E4E.svg)](fundstyle.sty)
[![Sections](https://img.shields.io/badge/Sections-7-6C757D.svg)](sections)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-6C757D.svg)](#structure)
[![References](https://img.shields.io/badge/References-44%2C%20all%20linked-6C757D.svg)](references.bib)
[![Compile](https://img.shields.io/badge/pdfLaTeX-0%20errors%2C%200%20overfull-9AA1A8.svg)](#compile-record)
[![Overleaf](https://img.shields.io/badge/Overleaf-zip%20ready-9AA1A8.svg)](28Sep26-packet-LaTeX.zip)

The compiled document of day 1. It is the attachment to letter 1 and the
document a program officer, a reviewer, or either applicant would be sent if
they asked what the company is and how it would staff its trial.

## Structure

| § | File | Title | Figures | Tables |
|:--|:--|:--|:--|:--|
| Front | [`sections/sec-00-front.tex`](sections/sec-00-front.tex) | Abstract and how to read this | | |
| 1 | [`sections/sec-01-the-signal.tex`](sections/sec-01-the-signal.tex) | The Signal, and Why It Is Spent First | 1 | 1 |
| 2 | [`sections/sec-02-action-register.tex`](sections/sec-02-action-register.tex) | The Action Register | | 2 |
| 3 | [`sections/sec-03-the-roster.tex`](sections/sec-03-the-roster.tex) | Eleven Roles, and the Six an Applicant Can Fill | 2 | 3 |
| 4 | [`sections/sec-04-obligations.tex`](sections/sec-04-obligations.tex) | Obligations Before and After a First Hire | | 4 |
| 5 | [`sections/sec-05-funding-routes.tex`](sections/sec-05-funding-routes.tex) | Four Routes That Can Fund a First Hire | 3 | 5 |
| 6 | [`sections/sec-06-references.tex`](sections/sec-06-references.tex) | Positioning, Method, and References | | |

## Files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | The cover, the author block with both disclaimers verbatim, the keywords, the contents, and one `\input` per section |
| [`fundstyle.sty`](fundstyle.sty) | The style; this copy carries the Torrey Pine palette |
| [`references.bib`](references.bib) | 44 entries, each with a clickable url, and a doi wherever one exists |
| [`sections/`](sections) | Seven section files, one per section |
| `main.pdf` | The compiled document |
| `28Sep26-packet-LaTeX.zip` | `main.tex`, `fundstyle.sty`, `references.bib` and `sections/`, ready to upload to Overleaf |

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
| Pages | 11 |
| LaTeX errors | 0 |
| Overfull boxes | 0 |
| Underfull boxes | 0 |
| Undefined citations or references | 0 |
| BibTeX warnings | 0 |
| Captions at two lines and -0.60cm | 8 of 8 |

## The palette

| Token | Hex | Used for |
|:--|:--|:--|
| `fundaccent` | `#2E5E4E` | Headings, table header rows, the cover block, open-role tiles |
| `fundmid` | `#5E8C7C` | The cover band and secondary fills |
| `fundpale` | `#E1ECE7` | The approval box and light fills |
| `fundgray` family | `#6C757D`, `#9AA1A8`, `#CED4DA`, `#E9ECEF` | Process, closed roles, legends |

Torrey Pine is the tree that grows on the La Jolla bluffs and nowhere else on
the mainland; it is given to the day about the first people who asked to work
there.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `08Sep26/packet/fundstyle.sty` | [`../../../auto-fund`](../../../auto-fund) | Every mechanism of `fundstyle.sty` below the palette block |
| `final-capital/references.bib` | [`../../../capitalization-plan`](../../../capitalization-plan) | The carried entries for the company's deposited works |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../move-in`](../../../move-in) | Table 3, Figure 2, and §3 |
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | The Phase 1 design numbers in §3 and §5 |
| `../briefs/brief-02-role-fit-and-structured-interview.md` | [`../briefs`](../briefs) | The interview and rubric summary in §4 |
| `../investing/capital-01-contingent-payroll-earmark.md` | [`../investing`](../investing) | The $40,000 earmark in Table 5 and Figure 3 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
