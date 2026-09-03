# 08Sep26 / packet / sections - seven section files (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../../README.md)
[![Day](https://img.shields.io/badge/Day-5%20of%205-8A4B2A.svg)](../..)
[![Sections](https://img.shields.io/badge/Sections-7-8A4B2A.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../../diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-6C757D.svg)](.)
[![Closing table](https://img.shields.io/badge/Table%2025-stands%20alone-9AA1A8.svg)](#the-one-rule-specific-to-this-day)

One `.tex` file per section, each `\input` from [`../main.tex`](../main.tex) in
the order below, per Rule 6 of the master prompt.

## The seven files

| § | File | Contents | Floats |
|:--|:--|:--|:--|
| 0 | [`sec-00-front.tex`](sec-00-front.tex) | Abstract, what the block produced in totals, the reading path | none |
| 1 | [`sec-01-the-release.tex`](sec-01-the-release.tex) | The six checks, the session sequence, what was released and what was withheld | Figure 13, Table 21 |
| 2 | [`sec-02-action-register.tex`](sec-02-action-register.tex) | Entry order, settlement basis, the fill record, what to do if a line does not fill, the approval step | Table 24 |
| 3 | [`sec-03-pipeline.tex`](sec-03-pipeline.tex) | Five stages, the four unchaseable items, what thirty days produce | Figure 15, Table 22 |
| 4 | [`sec-04-cadence.tex`](sec-04-cadence.tex) | Five standing functions, three survival rules, what is excluded, how the next block is built | Figure 14, Table 23 |
| 5 | [`sec-05-the-record.tex`](sec-05-the-record.tex) | Twenty-one actions, what is omitted, what is unchanged, what changed | Table 25 |
| 6 | [`sec-06-references.tex`](sec-06-references.tex) | Positioning for the whole block, method, the reference list | none |

## The one rule specific to this day

**Table 25 must stand without any file being opened.** Every row names a day, a
recipient class, and a question in one line. It is the table handed to somebody
who joins the program in a month, and a row that requires a lookup fails that
purpose. The section also states plainly what the table omits, because a record
that is wrong in a small way is not trusted in a large one.

## Float conventions

| Convention | Value |
|:--|:--|
| Figure carrier | `\begin{appfloat}[!tb]` then `\begin{appfig}[platform / construct]` |
| Table carrier | `\begin{apptable}` then `\begin{tabularx}{\textwidth}{...}` |
| Caption spacing | `\vspace{-0.60cm}` between the float and its caption, always |
| Caption lines | Exactly two, broken by hand, balanced within a small character spread |
| Column declaration | Every fixed column is `>{\raggedright\arraybackslash}p{...}` |
| Fit values | Always braced, since a value carrying commas is otherwise split into separate keys |
| Node names | Plain letters and digits only |
| Section symbol | `\S` wherever a codified reference or an internal section is named |
| Dashes | Single hyphens only |

## Rule 5 source map

| Used | From | Which section |
|:--|:--|:--|
| `final-capital/sections/sec-05-twelve-milestones.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-03`, the stage definitions |
| `final-capital/sections/sec-07-operating-plan.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-04`, the standing functions |
| `final-capital/sections/sec-09-risks-and-limits.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-06`, the positioning constraints |
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../../../move-in`](../../../../move-in) | `sec-06`, the recognition wording |
| `UC-San-Diego/priority-steps.md` §4 | [`../../../../potential-partners`](../../../../potential-partners) | `sec-01`, the follow-up interval |
| `../../07Sep26/investing/capital-04-queued-orders.md` | [`../../../07Sep26`](../../../07Sep26) | `sec-01` and `sec-02` |
| `../../briefs/brief-01`, `brief-02` | This day | `sec-03` and `sec-04` |
| `../../diagrams/fig-13`, `fig-14`, `fig-15` | This day | `sec-01`, `sec-04`, `sec-03` |
| `../../investing/capital-05-execution-and-settlement.md` | This day | `sec-02`, Table 24 |
| Days 1 to 4 of this block | [`../../..`](../../..) | `sec-05`, Table 25 in full |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
