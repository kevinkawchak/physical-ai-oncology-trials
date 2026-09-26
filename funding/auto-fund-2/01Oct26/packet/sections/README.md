# 01Oct26 / packet / sections - seven section files (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-4B2A7B.svg)](../..)
[![Sections](https://img.shields.io/badge/Sections-7-4B2A7B.svg)](.)
[![Rule 6](https://img.shields.io/badge/Rule%206-one%20.tex%20per%20section-6C757D.svg)](#the-seven-files)
[![Tables](https://img.shields.io/badge/Tables-5%20at%20%5Ctextwidth-6C757D.svg)](#rules-every-file-obeys)
[![Dashes](https://img.shields.io/badge/Em%20or%20double%20dashes-none-9AA1A8.svg)](#rules-every-file-obeys)

One `.tex` file per section, each committed on its own, each read by
[`../main.tex`](../main.tex) through a single `\input`.

## The seven files

| File | Section | Words, about | Figure | Table |
|:--|:--|:--|:--|:--|
| [`sec-00-front.tex`](sec-00-front.tex) | Abstract and how to read this | 350 |  |  |
| [`sec-01-the-inquiry-and-the-gate.tex`](sec-01-the-inquiry-and-the-gate.tex) | §1 The Inquiry, and the Gate in Front of the Pilot | 790 | Figure 10 | Table 16 |
| [`sec-02-action-register.tex`](sec-02-action-register.tex) | §2 The Action Register | 900 | Figure 11 | Table 17 |
| [`sec-03-how-the-program-is-split.tex`](sec-03-how-the-program-is-split.tex) | §3 How the Program Is Split | 540 |  | Table 18 |
| [`sec-04-the-phase-i-objectives.tex`](sec-04-the-phase-i-objectives.tex) | §4 The Phase I Technical Objectives | 610 |  | Table 19 |
| [`sec-05-merit-review.tex`](sec-05-merit-review.tex) | §5 The Merit Review Map | 800 | Figure 12 | Table 20 |
| [`sec-06-references.tex`](sec-06-references.tex) | §6 Positioning, Method, and References | 330 |  |  |

## What each file must support, and the source it rests on

| File | Claim and source |
|:--|:--|
| `sec-01` | That the company does not yet pass the pilot's gate; rests on the published eligibility and the company's record |
| `sec-02` | That registration and drafting must both finish before the Pitch; rests on the two form packs and Figure 11 |
| `sec-03` | That no component is funded twice; rests on Table 18 and letter 4 |
| `sec-04` | That every objective has a checkable threshold; rests on the deposited protocol and the Pitch |
| `sec-05` | That the review map shows its own gaps; rests on brief 3 and the last column of Table 20 |
| `sec-06` | That nothing is an award, an agreement, an eligibility finding, or advice; rests on the whole directory |

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
