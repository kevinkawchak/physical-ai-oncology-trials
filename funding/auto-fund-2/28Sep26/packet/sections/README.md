# 28Sep26 / packet / sections - seven section files (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../../README.md)
[![Day](https://img.shields.io/badge/Day-1%20of%205-2E5E4E.svg)](../..)
[![Sections](https://img.shields.io/badge/Sections-7-2E5E4E.svg)](.)
[![Rule 6](https://img.shields.io/badge/Rule%206-one%20.tex%20per%20section-6C757D.svg)](#the-seven-files)
[![Tables](https://img.shields.io/badge/Tables-5%20at%20%5Ctextwidth-6C757D.svg)](#what-each-file-carries)
[![Dashes](https://img.shields.io/badge/Em%20or%20double%20dashes-none-9AA1A8.svg)](#rules-every-file-obeys)

One `.tex` file per section, each committed on its own, each read by
[`../main.tex`](../main.tex) through a single `\input`.

## The seven files

| File | Section | Words, about | Figures | Tables |
|:--|:--|:--|:--|:--|
| [`sec-00-front.tex`](sec-00-front.tex) | Abstract and how to read this | 320 | | |
| [`sec-01-the-signal.tex`](sec-01-the-signal.tex) | §1 The Signal, and Why It Is Spent First | 970 | Figure 1 | Table 1 |
| [`sec-02-action-register.tex`](sec-02-action-register.tex) | §2 The Action Register | 540 | | Table 2 |
| [`sec-03-the-roster.tex`](sec-03-the-roster.tex) | §3 Eleven Roles, and the Six an Applicant Can Fill | 840 | Figure 2 | Table 3 |
| [`sec-04-obligations.tex`](sec-04-obligations.tex) | §4 Obligations Before and After a First Hire | 690 | | Table 4 |
| [`sec-05-funding-routes.tex`](sec-05-funding-routes.tex) | §5 Four Routes That Can Fund a First Hire | 830 | Figure 3 | Table 5 |
| [`sec-06-references.tex`](sec-06-references.tex) | §6 Positioning, Method, and References | 380 | | |

## What each file carries

| File | The claim it must support, and the source it rests on |
|:--|:--|
| `sec-01` | That the inquiries are evidence of a specific, limited kind; rests on the day 1 README's account of the two messages and on the dated record in `funding/daraxonrasib-llm-story.md` |
| `sec-02` | That every action is in one place with its file; rests on the day directory itself |
| `sec-03` | That six roles, 3.10 FTE, are open to an unlicensed applicant; rests on `funding/move-in/final-move-in/sections/sec-14-staffing-and-roles.tex` |
| `sec-04` | That the replies already comply with the rules that apply today; rests on the California and federal sources cited in Table 4 |
| `sec-05` | That two routes can pay a salary and two cannot; rests on the NIH and NSF solicitations and on the day's capital instruction |
| `sec-06` | That nothing in the packet is an agreement, an offer, or advice; rests on the whole directory |

## Rules every file obeys

| Rule | Value |
|:--|:--|
| Table width | `\begin{tabularx}{\textwidth}`, every fixed column `>{\raggedright\arraybackslash}p{...}` |
| Row geometry | Column widths cut so that no row carries one deep cell beside shallow ones |
| Caption spacing | `\vspace{-0.60cm}` before every `\tabcap` and `\figcaption` |
| Caption geometry | Two lines, balanced to within four characters |
| Punctuation | Single hyphens only; no em dash, en dash, double dash or triple dash |
| Symbols | The section sign for every codified or internal section reference |
| Dialect | American English, La Jolla usage |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `08Sep26/packet/sections/` | [`../../../../auto-fund`](../../../../auto-fund) | The seven-section shape and the back matter pattern in `sec-06` |
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | The trial design numbers in `sec-03` and `sec-05` |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../../move-in`](../../../../move-in) | Table 3 in `sec-03` |
| `../../diagrams/` | [`../../diagrams`](../../diagrams) | The three figure specifications |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
