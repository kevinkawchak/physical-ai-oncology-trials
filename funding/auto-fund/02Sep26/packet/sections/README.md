# 02Sep26 / packet / sections - seven section files (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../../README.md)
[![Day](https://img.shields.io/badge/Day-1%20of%205-0E5C63.svg)](../..)
[![Sections](https://img.shields.io/badge/Sections-7-0E5C63.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../../diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-6C757D.svg)](.)
[![Rule 6](https://img.shields.io/badge/Rule%206-one%20.tex%20per%20section-9AA1A8.svg)](#why-one-file-per-section)

One `.tex` file per section, each `\input` from
[`../main.tex`](../main.tex) in the order below.

## Why one file per section

Rule 6 of the master prompt requires one commit for each of `main.tex`, the
`.sty`, the `.bib` and the README, and one commit for each of the paper's `.tex`
sections. A section per file makes that possible, and it makes a correction to
one section reviewable without a diff that spans the whole document.

## The seven files

| § | File | Contents | Floats | Approximate length |
|:--|:--|:--|:--|:--|
| 0 | [`sec-00-front.tex`](sec-00-front.tex) | Abstract, what the packet is for, the six-minute reading path, what is quoted rather than paraphrased | none | 1 page |
| 1 | [`sec-01-what-changed.tex`](sec-01-what-changed.tex) | The approval, the June 2025 to August 2026 chronology, the four things unchanged | Figure 1, Table 1 | 3 pages |
| 2 | [`sec-02-action-register.tex`](sec-02-action-register.tex) | Why no letter is cold, the five letters, the single approval step | Figure 2, Table 2 | 3 pages |
| 3 | [`sec-03-evidence.tex`](sec-03-evidence.tex) | The six checkable quantities, their sources, and what is not claimed | Table 3 | 2 pages |
| 4 | [`sec-04-capital.tex`](sec-04-capital.tex) | Why a ladder rather than a sweep, the six lines, what is excluded | Figure 3, Table 4 | 2 pages |
| 5 | [`sec-05-outreach-and-route.tex`](sec-05-outreach-and-route.tex) | The nine reconciling figures, the delta, the eligibility constraint, the policy frame | Table 5 | 2 pages |
| 6 | [`sec-06-references.tex`](sec-06-references.tex) | Positioning, method and reproducibility, the reference list | none | 2 pages |

## The float conventions every section obeys

| Convention | Value |
|:--|:--|
| Figure carrier | `\begin{appfloat}[!tb]` then `\begin{appfig}[platform / construct]` |
| Table carrier | `\begin{apptable}` then `\begin{tabularx}{\textwidth}{...}` |
| Caption spacing | `\vspace{-0.60cm}` between the float and its caption, always |
| Caption lines | Exactly two, broken by hand, balanced within a small character spread |
| Column declaration | Every fixed column is `>{\raggedright\arraybackslash}p{...}` |
| Section symbol | `\S` wherever a codified reference or an internal section is named |
| Dashes | Single hyphens only. No em dash, no double dash, no triple dash |

## Rule 5 source map

| Used | From | Which section |
|:--|:--|:--|
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-03`, Table 3 in full |
| `final-capital/sections/sec-03-gate-and-programme.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-05`, Table 5 |
| `final-capital/sections/sec-04-capital-bridge.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-05`, the $5,900,000 and 3.67 to 1 lines |
| `final-capital/sections/sec-09-risks-and-limits.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-06`, the positioning constraints |
| `applications/emailed-source/README.md` | [`../../../../pdac-funding-applications`](../../../../pdac-funding-applications) | `sec-02`, Figure 2 and Table 2 |
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../../../move-in`](../../../../move-in) | `sec-04` fund separation, `sec-06` recognition letters |
| `../../diagrams/fig-01`, `fig-02`, `fig-03` | This day | `sec-01`, `sec-02`, `sec-04` |
| `../../investing/capital-01-treasury-ladder.md` | This day | `sec-04`, Table 4 |
| `../../emails/README.md` | This day | `sec-02`, Table 2 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
