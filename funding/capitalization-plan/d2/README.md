# D2-type figures - Capitalization Plan (v4.5.0)

[![Platform](https://img.shields.io/badge/Platform-D2-3C7DB2.svg)](https://d2lang.com)
[![Figures](https://img.shields.io/badge/Figures-5-00417A.svg)](.)
[![Constructs](https://img.shields.io/badge/Constructs-grid%20%2F%20sql__table%20%2F%20layers%20%2F%20containers-6C757D.svg)](.)
[![Stage](https://img.shields.io/badge/Stage-3%20of%208-6C757D.svg)](../sub-prompts/stage-3-d2)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

Five figure specifications produced by
[`../sub-prompts/stage-3-d2/`](../sub-prompts/stage-3-d2). Each is reproduced in
LaTeX by the `d2*` TikZ vocabulary in `capstyle.sty`. D2 is used wherever the
claim is about **containment or tabulation**: what sits inside what, what is
priced against what, what joins to what.

## Contents

| File | Figure | § | Construct | The question it answers |
|:--|:--|:--|:--|:--|
| [`fig-02-institutional-form-grid.d2.md`](fig-02-institutional-form-grid.d2.md) | 2 | 1 | true grid, 8 by 6 | What kind of institution is a 2.6 FTE firm, on the report's own axes? |
| [`fig-05-asset-register-records.d2.md`](fig-05-asset-register-records.d2.md) | 5 | 2 | sql_table records | What does the company own, license, contract for, and lack? |
| [`fig-08-two-prices-one-programme.d2.md`](fig-08-two-prices-one-programme.d2.md) | 8 | 3 | layers plus measures | What does the same work cost under two mechanisms? |
| [`fig-11-capital-tiers.d2.md`](fig-11-capital-tiers.d2.md) | 11 | 4 | nested containers | Where does every dollar come from, and what separates the tiers? |
| [`fig-16-clinical-evidence-panel.d2.md`](fig-16-clinical-evidence-panel.d2.md) | 16 | 6 | grid plus intervals | Which published quantities can a funder check before writing a cheque? |

## Why five, and how the two money figures differ

A capitalization plan spends most of its argument in tables, so d2 ties mermaid
for the largest allocation. Two of the five draw money, and they were
deliberately built to look nothing alike:

| | Figure 8 | Figure 11 |
|:--|:--|:--|
| Reading direction | Left to right | Bottom to top |
| Organizing axis | Purpose, four budget layers | Source, three capital tiers |
| Money shown | $3,500,000, $1,396,000, $2,104,000 | $1,606,000, unpriced, $5,900,000 |
| The subject | Two prices for one programme | Two firewalls between three tiers |
| The visual device | A total rule spanning three money columns | 16 mm of deliberately empty canvas at each firewall |

A reader who has seen one still learns something from the other, which is the
test the stage sub-prompt sets.

## Anti-defect record

| Defect class | How these five avoid it |
|:--|:--|
| Faked grids | Figures 2 and 16 declare `grid-rows` and `grid-columns` and give every score column the identical width, 20 mm and 17 mm, so the header cannot drift out of register |
| Prose in a record | Figure 5's `sql_table` fields are `name : type` only. Row counts sit in a separate header strip because a count is metadata, not a column |
| Nesting too deep | No container nests more than two deep. Figures 8 and 11 each nest exactly one level |
| Invented style keys | Only `fill`, `stroke`, `stroke-width`, `stroke-dash`, `font-color`, `border-radius` and `bold` appear. No CSS, no class, no gradient |
| Edge overlap | Figure 5 carries three edges and states the bend on the only one that could cross a record. Figure 11's four crossings are vertical, at two fixed x values |
| Corridor collision | Figures 5, 11 and 16 each reserve a corridor of 10 mm or more that carries no node; in Figure 11 the two 16 mm firewall bands are the subject of the figure |

## Arithmetic that must hold

Two of these figures carry sums, and a figure whose numbers do not add is wrong
rather than merely ugly.

| Check | Where | Holds |
|:--|:--|:--|
| $1,600,000 + $720,000 + $780,000 + $400,000 = $3,500,000 | Figure 8, column A | Yes |
| $612,000 + $268,000 + $412,000 + $104,000 = $1,396,000 | Figure 8, column B | Yes |
| $988,000 + $452,000 + $368,000 + $296,000 = $2,104,000 | Figure 8, delta | Yes |
| Column B plus delta equals column A, row by row | Figure 8 | Yes, all four rows |
| $266,000 + $20,000 + $20,000 = $306,000 | Figure 8, Phase I load | Yes |
| $1,130,000 + $85,000 + $85,000 = $1,300,000 | Figure 8, Phase II load | Yes |
| $900,000 + $5,000,000 = $5,900,000 | Figure 11, tier 3 | Yes |
| $5,900,000 divided by $1,606,000 = 3.67 | Figure 11, leverage | Yes, to two places |

## Rule 5 source map

| These figures use | From | For |
|:--|:--|:--|
| `chunk-03` NOVEL PERFORMERS and its Table 1 | `../../science-golden-age` | Figure 2's seven rows and four columns |
| `chunk-08` annex | `../../science-golden-age` | Figure 11's 3:1 leverage target |
| `final-apply/sections/sec-08-budget-and-leverage.tex` | `../../pdac-funding-applications` | Figure 8's four-layer frame and Figure 11's unpriced categories |
| `final-apply/sections/sec-05-trial-evidence.tex` | `../../pdac-funding-applications` | Figure 16's six quantities and four limitations |
| `applications/app-05-nih-sbir-seed/` | `../../pdac-funding-applications` | Figures 8 and 11, the two award amounts |
| `Physical AI Oncology Trial Founding Documents.md` | `../../supplementary` | Figure 5's thirteen owned rows and their DOIs |
| `trial-protocol/`, `trial-ind/`, `trial-phase-2/` | repository root | Figures 5 and 16 |
| `final-apply/applystyle.sty` | `../../pdac-funding-applications` | The `d2*` vocabulary and the quantitative primitives |
