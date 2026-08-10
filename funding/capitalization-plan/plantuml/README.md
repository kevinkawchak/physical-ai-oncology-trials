# PlantUML-type figures - Capitalization Plan (v4.5.0)

[![Platform](https://img.shields.io/badge/Platform-PlantUML-3C7DB2.svg)](https://plantuml.com)
[![Figures](https://img.shields.io/badge/Figures-3-00417A.svg)](.)
[![Constructs](https://img.shields.io/badge/Constructs-use%20case%20%2F%20state%20%2F%20activity-6C757D.svg)](.)
[![Stage](https://img.shields.io/badge/Stage-2%20of%208-6C757D.svg)](../sub-prompts/stage-2-plantuml)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

Three figure specifications produced by
[`../sub-prompts/stage-2-plantuml/`](../sub-prompts/stage-2-plantuml). Each is
reproduced in LaTeX by the `uml*` TikZ vocabulary in `capstyle.sty`. PlantUML is
used wherever the claim is about **permission**: who may act, what guard must
hold first, and what runs concurrently.

## Contents

| File | Figure | § | Construct | The question it answers |
|:--|:--|:--|:--|:--|
| [`fig-06-sponsor-site-boundary.puml.md`](fig-06-sponsor-site-boundary.puml.md) | 6 | 2 | use case, two boundaries | What may the company do alone, and what needs an institution? |
| [`fig-10-capital-firewall-guards.puml.md`](fig-10-capital-firewall-guards.puml.md) | 10 | 4 | state with guards | Which capital positions may be held while a participant is on study? |
| [`fig-15-milestone-evidence-activity.puml.md`](fig-15-milestone-evidence-activity.puml.md) | 15 | 5 | activity, fork and join | What three parties do at once to close one milestone? |

## Why three

The paper makes exactly three claims about permission, and each needs a
different UML construct. A use case answers who is inside which boundary; a
guarded state machine answers what condition licenses a change of position; an
activity with a fork answers what proceeds in parallel. Adding a fourth would
mean restating one of these three in a construct that fits it worse.

The three sit in three different sections, §2, §4 and §5, so no page carries two
of them and the reader never has to hold two permission models at once.

## Anti-defect record

| Defect class | How these three avoid it |
|:--|:--|
| Edge overlap | Figure 10's four firewall crossings are outbound and return pairs 4 mm apart at two fixed x values. Figure 15's single return edge is routed at x = -0.30, outside every branch column |
| Over-density | Figure 6 splits twelve use cases across two boundaries of six, in two columns of three. Figure 15 splits ten steps across three branches. No rank carries more than five |
| Syntax hallucination | Every block is closed: `fork` with `end fork`, `if` with `endif`, `@startuml` with `@enduml`. `skinparam` sets colour, font and shadow only |
| Guards in labels | Every guard is written on the transition as `[condition]`, never inside a state name. Figure 10's guards sit 3.4 mm above the line with a white fill |
| Boundary collision | Figure 6's three boundaries are separated by 9 mm corridors that carry no node, only crossing association lines |
| Poor directionality | Figures 6 and 10 are drawn left to right. Figure 15 is the one figure in the paper drawn top down, because a fork and join is genuinely hierarchical in time |

## Palette

Corporate Blue `#00417A`, lighter `#3C7DB2`, pale `#DCE8F1`, Professional Gray
`#6C757D`, grays `#E9ECEF`, `#CED4DA`, `#9AA1A8`, white `#FFFFFF`. No black
fill anywhere. The two fault-tree gate glyphs used elsewhere in the paper take
`pagraym` and `pagrayl` for the same reason.

## Rule 5 source map

| These figures use | From | For |
|:--|:--|:--|
| `trial-protocol/`, `trial-ind/`, `trial-phase-2/` | repository root | Figure 6's sponsor scope and Figure 15's retention obligations |
| `UC-San-Diego/` | `../../potential-partners` | Figure 6's site scope and the missing CTA |
| `final-apply/sections/sec-05-trial-evidence.tex` | `../../pdac-funding-applications` | Figure 6's ISGPS and DSMB assignments |
| `final-apply/sections/sec-06-physical-ai-governance.tex` | `../../pdac-funding-applications` | Figure 15's hash-and-replay branch |
| `final-apply/sections/sec-08-budget-and-leverage.tex` | `../../pdac-funding-applications` | Figure 10's cost-share framing |
| `Physical-AI-Oncology-Trial-Competition-Proposal.zip` | `../../supplementary/source-files` | Figure 6's January 13, 2026 baseline |
| 21 CFR parts 54 and 312; 13 CFR 121.702 | codified | Figures 6, 10 and 15, every guard and every retention term |
| `final-apply/applystyle.sty` | `../../pdac-funding-applications` | The `uml*` vocabulary and the `\umlactor` macro |
