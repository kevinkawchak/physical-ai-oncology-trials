# 07Sep26 / packet / sections - seven section files (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-5B3A5E.svg)](../..)
[![Sections](https://img.shields.io/badge/Sections-7-5B3A5E.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../../diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-6C757D.svg)](.)
[![Sent](https://img.shields.io/badge/Shown%20as%20done-nothing-9AA1A8.svg)](#the-one-rule-specific-to-this-day)

One `.tex` file per section, each `\input` from [`../main.tex`](../main.tex) in
the order below, per Rule 6 of the master prompt.

## The seven files

| § | File | Contents | Floats |
|:--|:--|:--|:--|
| 0 | [`sec-00-front.tex`](sec-00-front.tex) | Abstract, the nothing-sent notice, why a closed day is in the schedule, the reading path | none |
| 1 | [`sec-01-the-closed-day.tex`](sec-01-the-closed-day.tex) | The two lanes, the release list, and what is finished today | Figure 10, Table 16 |
| 2 | [`sec-02-action-register.tex`](sec-02-action-register.tex) | Six queued orders, why a basis and not a price, the auction re-check, the approval step | Table 20 |
| 3 | [`sec-03-data-room.tex`](sec-03-data-room.tex) | Nine folders, three access classes, five first lines, custody, what is absent | Figure 11, Table 17 |
| 4 | [`sec-04-recognition-letters.tex`](sec-04-recognition-letters.tex) | Seven things the letters are not, the approved wording, the one place they are mentioned | Table 18 |
| 5 | [`sec-05-diligence-and-stops.tex`](sec-05-diligence-and-stops.tex) | Four unanswered questions, the failure combinations, why the stops are published | Figure 12, Table 19 |
| 6 | [`sec-06-references.tex`](sec-06-references.tex) | Positioning, method, the reference list | none |

## The one rule specific to this day

**No section shows anything as done.** Figure 10's held lane is drawn in the pale
shade and carries a printed note naming the release condition. Figure 11's access
class column carries "Under a CDA" as a value rather than as a footnote. Figure
12's root node is a condition and not an event. A reader who screenshots any of
the three must come away understanding that the day produced preparation and not
action.

## Float conventions

| Convention | Value |
|:--|:--|
| Figure carrier | `\begin{appfloat}[!tb]` then `\begin{appfig}[platform / construct]` |
| Table carrier | `\begin{apptable}` then `\begin{tabularx}{\textwidth}{...}` |
| Caption spacing | `\vspace{-0.60cm}` between the float and its caption, always |
| Caption lines | Exactly two, broken by hand, balanced within a small character spread |
| Column declaration | Every fixed column is `>{\raggedright\arraybackslash}p{...}` |
| Node names | Plain letters and digits. A name generated from a negative or decimal coordinate is read as a subtraction or as a node-and-anchor pair, and both failures are fatal at compile time |
| Section symbol | `\S` wherever a codified reference or an internal section is named |
| Dashes | Single hyphens only |

## Rule 5 source map

| Used | From | Which section |
|:--|:--|:--|
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../../../move-in`](../../../../move-in) | `sec-04`, Table 18 and the approved wording |
| `final-move-in/sections/sec-00-front.tex` | [`../../../../move-in`](../../../../move-in) | `sec-03`, the company record rows |
| `final-capital/sections/sec-09-risks-and-limits.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-05`, Figure 12 and Table 19 |
| `final-capital/sections/sec-10-build-method.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-03`, the custody paragraph |
| `UC-San-Diego/priority-steps.md` §10, §11 | [`../../../../potential-partners`](../../../../potential-partners) | `sec-01` and `sec-05` |
| `../../briefs/brief-01`, `brief-02`, `brief-03` | This day | `sec-03`, `sec-04`, `sec-05` |
| `../../diagrams/fig-10`, `fig-11`, `fig-12` | This day | `sec-01`, `sec-03`, `sec-05` |
| `../../investing/capital-04-queued-orders.md` | This day | `sec-02`, Table 20 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
