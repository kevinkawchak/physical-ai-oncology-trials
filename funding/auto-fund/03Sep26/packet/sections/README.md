# 03Sep26 / packet / sections - seven section files (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../../README.md)
[![Day](https://img.shields.io/badge/Day-2%20of%205-1B3A5C.svg)](../..)
[![Sections](https://img.shields.io/badge/Sections-7-1B3A5C.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../../diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-6C757D.svg)](.)
[![Offer](https://img.shields.io/badge/Offering%20terms-none-9AA1A8.svg)](#the-one-rule-specific-to-this-day)

One `.tex` file per section, each `\input` from [`../main.tex`](../main.tex) in
the order below, per Rule 6 of the master prompt.

## The seven files

| § | File | Contents | Floats |
|:--|:--|:--|:--|
| 0 | [`sec-00-front.tex`](sec-00-front.tex) | Abstract, the securities notice, the line the packet does not cross, the reading path | none |
| 1 | [`sec-01-the-gap.tex`](sec-01-the-gap.tex) | Why $2,104,000, the seven proceeds lines, what federal funds never pay for, the cost of publishing | Table 6 |
| 2 | [`sec-02-action-register.tex`](sec-02-action-register.tex) | The five letters, the signing order and its idle spans, the approval step | Figure 6, Table 7 |
| 3 | [`sec-03-instruments.tex`](sec-03-instruments.tex) | Three instruments on eight attributes, the deciding row, the tranche shape, the recommendation | Figure 4, Table 8 |
| 4 | [`sec-04-firewall-and-position.tex`](sec-04-firewall-and-position.tex) | The four 21 CFR 54.2 triggers, what each instrument does to the firewall, the five states and four guards | Figure 5, Table 9 |
| 5 | [`sec-05-reserve.tex`](sec-05-reserve.tex) | The reserve on two branches, what stays excluded on both, three sources of money | Table 10 |
| 6 | [`sec-06-references.tex`](sec-06-references.tex) | Positioning, method, the reference list | none |

## The one rule specific to this day

**No section states an offering term.** No valuation, cap, discount, minimum, or
closing date appears in any of the seven files, and none appears in any of the
three figures. A figure is the easiest thing in a document to screenshot out of
context, so Figures 4, 5 and 6 carry no amount at all and Figure 4 carries a
printed note saying why.

## Float conventions

| Convention | Value |
|:--|:--|
| Figure carrier | `\begin{appfloat}[!tb]` then `\begin{appfig}[platform / construct]` |
| Table carrier | `\begin{apptable}` then `\begin{tabularx}{\textwidth}{...}` |
| Caption spacing | `\vspace{-0.60cm}` between the float and its caption, always |
| Caption lines | Exactly two, broken by hand, balanced within a small character spread |
| Column declaration | Every fixed column is `>{\raggedright\arraybackslash}p{...}` |
| Section symbol | `\S` wherever a codified reference or an internal section is named |
| Dashes | Single hyphens only |

## Rule 5 source map

| Used | From | Which section |
|:--|:--|:--|
| `final-capital/sections/sec-03-gate-and-programme.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-01`, the gap arithmetic |
| `final-capital/sections/sec-04-capital-bridge.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-03` tranches, `sec-04` firewall |
| `final-capital/sections/sec-09-risks-and-limits.tex` | [`../../../../capitalization-plan`](../../../../capitalization-plan) | `sec-01`, the cost of publishing |
| `UC-San-Diego/priority-steps.md` §2 and §12 | [`../../../../potential-partners`](../../../../potential-partners) | `sec-02` and `sec-04` |
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../../../move-in`](../../../../move-in) | `sec-01` and `sec-05`, fund separation |
| `../../briefs/brief-01`, `brief-02`, `brief-03` | This day | `sec-03`, `sec-04`, `sec-01` |
| `../../diagrams/fig-04`, `fig-05`, `fig-06` | This day | `sec-03`, `sec-04`, `sec-02` |
| `../../investing/capital-02-corporate-reserve-allocation.md` | This day | `sec-05`, Table 10 |
| `../../forms/form-01-reg-d-506b-form-d.md` | This day | `sec-04`, guards 2 and 4 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
