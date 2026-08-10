# Mermaid-type figures - Capitalization Plan (v4.5.0)

[![Platform](https://img.shields.io/badge/Platform-Mermaid-3C7DB2.svg)](https://mermaid.js.org)
[![Figures](https://img.shields.io/badge/Figures-5-00417A.svg)](.)
[![Constructs](https://img.shields.io/badge/Constructs-flowchart%20%2F%20state%20%2F%20sequence%20%2F%20gantt-6C757D.svg)](.)
[![Stage](https://img.shields.io/badge/Stage-1%20of%208-6C757D.svg)](../sub-prompts/stage-1-mermaid)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

Five figure specifications produced by
[`../sub-prompts/stage-1-mermaid/`](../sub-prompts/stage-1-mermaid). Each is
reproduced in LaTeX by the `mm*` TikZ vocabulary in `capstyle.sty`. Mermaid is
used wherever the paper's claim is about **order in time or a decision taken at
a point in time**.

## Contents

| File | Figure | § | Construct | The question it answers |
|:--|:--|:--|:--|:--|
| [`fig-01-clause-eligibility-filter.md`](fig-01-clause-eligibility-filter.md) | 1 | 1 | flowchart LR | Which clause of the report does this company actually qualify under? |
| [`fig-07-phase-gate-state-machine.md`](fig-07-phase-gate-state-machine.md) | 7 | 3 | stateDiagram-v2 | What must hold at month nine for Phase II to begin? |
| [`fig-12-financing-event-sequence.md`](fig-12-financing-event-sequence.md) | 12 | 4 | sequenceDiagram | In what order is a private round executed while a trial enrolls? |
| [`fig-13-twelve-milestone-calendar.md`](fig-13-twelve-milestone-calendar.md) | 13 | 5 | gantt | When does each milestone open, close, and produce its artifact? |
| [`fig-19-august-traction-chain.md`](fig-19-august-traction-chain.md) | 19 | 8 | flowchart LR | What did the July and August 2026 contacts actually unlock? |

## Why five, and why these four constructs

Mermaid's four constructs answer four different temporal questions, and this
paper asks all four. A capitalization plan is mostly an argument about sequence:
what has to happen before what, what decides, and how long each interval is.
Two flowcharts are used rather than one because Figure 1 is a filter, which
narrows, and Figure 19 is a fan, which widens; drawing both as one construct
would hide that they run in opposite directions.

The count is five, tied with d2 for the largest allocation, and the reasoning is
in [`../sub-prompts/README.md`](../sub-prompts/README.md).

## Anti-defect record

| Defect class | How these five avoid it |
|:--|:--|
| Edge overlap | Every bend angle is stated numerically in the figure's own TikZ notes. Figure 19's five converging edges use a symmetric 22/12/0/-12/-22 fan; Figure 7's two gate exits use 20 and 14 rather than a shared value |
| Over-density | No rank carries more than five nodes. Figures 1 and 19 group with `subgraph`; Figure 13's twelve rows are split into two labelled sections by a rule |
| Syntax hallucination | Only Mermaid keywords appear in a ```mermaid fence. No CSS, no hex gradient, no custom font. Labels carry `and` in place of `&` and no unescaped bracket |
| Poor directionality | Four of the five declare `LR` or are natively horizontal. Only the two `subgraph` interiors use `TB`, where the content is genuinely a list |
| Layout instability | Every node carries an absolute coordinate in the TikZ notes, so adding an element later moves nothing already placed |

## Palette

Corporate Blue `#00417A`, lighter `#3C7DB2`, pale `#DCE8F1`, Professional Gray
`#6C757D`, grays `#E9ECEF`, `#CED4DA`, `#9AA1A8`, white `#FFFFFF`. Black is a
stroke and a text colour only. No figure carries a black fill; the audit is a
grep for `padark`, which must return nothing.

## Rule 5 source map

| These figures use | From | For |
|:--|:--|:--|
| `chunk-01`, `chunk-03`, `chunk-04`, `chunk-05` | `../../science-golden-age` | Figure 1's three clauses and four tests |
| `applications/app-05-nih-sbir-seed/` | `../../pdac-funding-applications` | Figures 7 and 13, the two award amounts and the 9 plus 24 month term |
| `final-apply/sections/sec-08-budget-and-leverage.tex` | `../../pdac-funding-applications` | Figure 13's cost column, cut from the four-layer frame |
| `applications/emailed-source/` | `../../pdac-funding-applications` | Figure 19's rows 2 and 3 |
| `UC-San-Diego/` | `../../potential-partners` | Figure 19's rows 1 and 5, and Figure 7's guard G4 |
| `trial-protocol/`, `trial-ind/` | repository root | Figures 12 and 13, the 3+3 escalation and the IND clocks |
| `final-apply/applystyle.sty` | `../../pdac-funding-applications` | The `mm*` vocabulary `capstyle.sty` inherits |
