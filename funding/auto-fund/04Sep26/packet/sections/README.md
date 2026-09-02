# 04Sep26 / packet / sections - seven section files (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../../README.md)
[![Day](https://img.shields.io/badge/Day-3%20of%205-2F5D3A.svg)](../..)
[![Sections](https://img.shields.io/badge/Sections-7-2F5D3A.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../../diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-6C757D.svg)](.)
[![Agreements](https://img.shields.io/badge/Agreements%20asserted-none-9AA1A8.svg)](#the-one-rule-specific-to-this-day)

One `.tex` file per section, each `\input` from [`../main.tex`](../main.tex) in
the order below, per Rule 6 of the master prompt.

## The seven files

| § | File | Contents | Floats |
|:--|:--|:--|:--|
| 0 | [`sec-00-front.tex`](sec-00-front.tex) | Abstract, the no-agreement notice, why the three days are in this order, the reading path | none |
| 1 | [`sec-01-two-routes.tex`](sec-01-two-routes.tex) | Parallel versus sequential, seven criteria, where functions sit, the deferred introductions, the escalation rule | Figure 8, Table 12 |
| 2 | [`sec-02-action-register.tex`](sec-02-action-register.tex) | The five feasibility questions, the gating determination, what is brought and what is not asked, the approval step | Table 11 |
| 3 | [`sec-03-obligations.tex`](sec-03-obligations.tex) | The three-cluster split, why the site holds the investigator role, the three corrections, four open questions | Figure 7, Table 15 |
| 4 | [`sec-04-foundations.tex`](sec-04-foundations.tex) | The four-gate funnel, three mechanism classes, what a foundation is told about the evidence | Figure 9, Table 13 |
| 5 | [`sec-05-startup-costs.tex`](sec-05-startup-costs.tex) | Four draw stages, why the pool is held short, the reconciliation, what a site is told | Table 14 |
| 6 | [`sec-06-references.tex`](sec-06-references.tex) | Positioning, method, the reference list | none |

## The one rule specific to this day

**No section asserts an agreement, and no figure lets a reader infer one.**
Figure 7's cluster titles carry the words "under an agreement that does not yet
exist" and "no agreement exists" inside the figure rather than in a footnote,
because a figure is screenshotted more often than a footnote is read. Figure 8's
first cluster title says "candidate site, feasibility stage only" for the same
reason.

## Float conventions

| Convention | Value |
|:--|:--|
| Figure carrier | `\begin{appfloat}[!tb]` then `\begin{appfig}[platform / construct]` |
| Table carrier | `\begin{apptable}` then `\begin{tabularx}{\textwidth}{...}` |
| Caption spacing | `\vspace{-0.60cm}` between the float and its caption, always |
| Caption lines | Exactly two, broken by hand, balanced within a small character spread |
| Column declaration | Every fixed column is `>{\raggedright\arraybackslash}p{...}` |
| Fit values | Always braced, because a `fit` value carrying commas is otherwise split into separate keys by the TikZ key parser |
| Section symbol | `\S` wherever a codified reference or an internal section is named |
| Dashes | Single hyphens only |

## Rule 5 source map

| Used | From | Which section |
|:--|:--|:--|
| `UC-San-Diego/README.md` | [`../../../../potential-partners`](../../../../potential-partners) | `sec-02`, Table 11 |
| `UC-San-Diego/priority-steps.md` §3, §4 | [`../../../../potential-partners`](../../../../potential-partners) | `sec-01`, the escalation rule |
| `UC-San-Diego/priority-steps.md` §12 | [`../../../../potential-partners`](../../../../potential-partners) | `sec-03`, the four open questions |
| `Scripps/priority-steps.md` §2, §10 | [`../../../../potential-partners`](../../../../potential-partners) | `sec-01`, Table 12 and the deferred introductions |
| `final-capital/sections/sec-04-capital-bridge.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-03`, the conflict boundary |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../../move-in`](../../../../move-in) | `sec-05` and Figure 8's function list |
| `../../briefs/brief-01`, `brief-02` | This day | `sec-02` and `sec-01` |
| `../../diagrams/fig-07`, `fig-08`, `fig-09` | This day | `sec-03`, `sec-01`, `sec-04` |
| `../../investing/capital-03-site-startup-reserve.md` | This day | `sec-05`, Table 14 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
