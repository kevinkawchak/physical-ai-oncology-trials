# 02Oct26 / packet / sections - seven section files (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../../README.md)
[![Day](https://img.shields.io/badge/Day-5%20of%205-8C5A12.svg)](../..)
[![Sections](https://img.shields.io/badge/Sections-7-8C5A12.svg)](.)
[![Rule 6](https://img.shields.io/badge/Rule%206-one%20.tex%20per%20section-6C757D.svg)](#the-seven-files)
[![Tables](https://img.shields.io/badge/Tables-5%20at%20%5Ctextwidth-6C757D.svg)](#rules-every-file-obeys)
[![Dashes](https://img.shields.io/badge/Em%20or%20double%20dashes-none-9AA1A8.svg)](#rules-every-file-obeys)

One `.tex` file per section, each committed on its own, each read by
[`../main.tex`](../main.tex) through a single `\input`.

## The seven files

| File | Section | Words, about | Figure | Table |
|:--|:--|:--|:--|:--|
| [`sec-00-front.tex`](sec-00-front.tex) | Abstract and how to read this | 310 |  |  |
| [`sec-01-the-week-in-one-record.tex`](sec-01-the-week-in-one-record.tex) | §1 The Week in One Record | 770 | Figure 13 | Table 21 |
| [`sec-02-action-register.tex`](sec-02-action-register.tex) | §2 The Action Register | 680 | Figure 14 | Table 22 |
| [`sec-03-the-follow-up-rule.tex`](sec-03-the-follow-up-rule.tex) | §3 The Follow-Up Rule | 920 | Figure 15 | Table 23 |
| [`sec-04-the-robotic-phase-1-in-numbers.tex`](sec-04-the-robotic-phase-1-in-numbers.tex) | §4 The Robotic Phase 1, in Numbers | 730 |  | Table 24 |
| [`sec-05-the-routes-opened.tex`](sec-05-the-routes-opened.tex) | §5 The Routes Opened This Week | 540 |  | Table 25 |
| [`sec-06-references.tex`](sec-06-references.tex) | §6 Positioning, Method, and References | 350 |  |  |

## What each file must support, and the source it rests on

| File | Claim and source |
|:--|:--|
| `sec-01` | That the week produced sixteen letters from four signals; rests on the files of days 1 to 4 |
| `sec-02` | That no letter leaves if its condition fails; rests on Table 22 and each letter's checklist |
| `sec-03` | That only three follow-ups are earned today; rests on the ledger in Table 23 and the rule in Figure 15 |
| `sec-04` | That every part of the Phase 1 is specified to a checkable number; rests on the deposited protocol, IND and plan |
| `sec-05` | That four routes pay only after an award not yet held; rests on the published eligibility of each |
| `sec-06` | That silence is not a position and nothing is an award, an agreement, or advice; rests on the whole directory |

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
