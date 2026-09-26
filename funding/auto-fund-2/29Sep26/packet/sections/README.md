# 29Sep26 / packet / sections - seven section files (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../../README.md)
[![Day](https://img.shields.io/badge/Day-2%20of%205-34495E.svg)](../..)
[![Sections](https://img.shields.io/badge/Sections-7-34495E.svg)](.)
[![Rule 6](https://img.shields.io/badge/Rule%206-one%20.tex%20per%20section-6C757D.svg)](#the-seven-files)
[![Tables](https://img.shields.io/badge/Tables-5%20at%20%5Ctextwidth-6C757D.svg)](#rules-every-file-obeys)
[![Dashes](https://img.shields.io/badge/Em%20or%20double%20dashes-none-9AA1A8.svg)](#rules-every-file-obeys)

One `.tex` file per section, each committed on its own, each read by
[`../main.tex`](../main.tex) through a single `\input`.

## The seven files

| File | Section | Words, about | Figure | Table |
|:--|:--|:--|:--|:--|
| [`sec-00-front.tex`](sec-00-front.tex) | Abstract and how to read this | 330 |  |  |
| [`sec-01-the-update.tex`](sec-01-the-update.tex) | §1 The Update, and the Record Behind the Request | 650 |  | Table 6 |
| [`sec-02-action-register.tex`](sec-02-action-register.tex) | §2 The Action Register | 800 | Figure 4 | Table 7 |
| [`sec-03-independence-and-simultaneity.tex`](sec-03-independence-and-simultaneity.tex) | §3 Independence and Simultaneity | 600 | Figure 5 | Table 8 |
| [`sec-04-what-a-standard-requires.tex`](sec-04-what-a-standard-requires.tex) | §4 What an External Standard Requires | 800 | Figure 6 | Table 9 |
| [`sec-05-the-evidence.tex`](sec-05-the-evidence.tex) | §5 The Evidence Behind the Robotic Phase 1 | 590 |  | Table 10 |
| [`sec-06-references.tex`](sec-06-references.tex) | §6 Positioning, Method, and References | 360 |  |  |

## What each file must support, and the source it rests on

| File | Claim and source |
|:--|:--|
| `sec-01` | That the record is dated and checkable; rests on the eleven identifiers and the repository release list |
| `sec-02` | That only two of the request's next steps are in the company's hands; rests on the day's letters |
| `sec-03` | That the work was independent and simultaneous; rests on the public trial registries and the deposit dates |
| `sec-04` | That three criteria are met, three in part, and two not yet; rests on brief 2 and the capitalization plan's evidence section |
| `sec-05` | That the six quantities support running the Phase 1 and nothing more; rests on the capitalization plan and the deposited protocol |
| `sec-06` | That nothing is a finding, an endorsement, an agreement, or advice; rests on the whole directory |

## Rules every file obeys

| Rule | Value |
|:--|:--|
| Table width | `\begin{tabularx}{\textwidth}`, every fixed column `>{\raggedright\arraybackslash}p{...}` |
| Row geometry | Column widths cut so that no row carries one deep cell beside shallow ones |
| Caption spacing | `\vspace{-0.60cm}` before every `\tabcap` and `\figcaption` |
| Caption geometry | Two lines, balanced to within a few characters |
| Punctuation | Single hyphens only; no em dash, en dash, double dash or triple dash |
| Symbols | The section sign for every internal section reference |
| Dialect | American English, La Jolla usage |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `08Sep26/packet/sections/` | [`../../../../auto-fund`](../../../../auto-fund) | The seven-section shape and the back matter pattern in `sec-06` |
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | The trial design numbers and the six quantities |
| `../../diagrams/` | [`../../diagrams`](../../diagrams) | The three figure specifications |
| `../../../inputs/README.md` | [`../../../inputs`](../../../inputs) | The money frame and the Phase 1 parameters |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
