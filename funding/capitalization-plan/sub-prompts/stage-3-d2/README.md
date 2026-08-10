## Stage 3 sub-prompt - d2-type figures

[![Stage](https://img.shields.io/badge/Stage-3%20of%208-00417A.svg)](.)
[![Platform](https://img.shields.io/badge/Platform-D2-3C7DB2.svg)](https://d2lang.com)
[![Figures](https://img.shields.io/badge/Figures-5-6C757D.svg)](../../d2)
[![Output](https://img.shields.io/badge/Output-..%2Fd2-9AA1A8.svg)](../../d2)

### Instruction

Produce five d2-type figure specifications in
`funding/capitalization-plan/d2/`, one file per figure, one commit per file,
committed the moment each file is written.

D2 is chosen wherever the claim is about **containment or tabulation**: what
sits inside what, what is priced against what, what joins to what. A
capitalization plan makes this claim more often than any other, which is why
the count ties mermaid at five.

| Figure | Section | Construct | Perspective no other figure takes |
|:--|:--|:--|:--|
| 2 | §1 The Novel-Performer Case | true grid, seven rows by five columns | This company scored against the report's own institutional-form table, cell by cell |
| 5 | §2 The Entity and the Asset | sql_table records | The asset register as typed records: owner, instrument, date, status, encumbrance |
| 8 | §3 The $1.6M Gate | layers plus paired measures | The same work priced twice, layer by layer, with the delta carried as a third column |
| 11 | §4 Capital Bridge | nested containers | Three capital tiers as three physically separated containers with the firewall as the gap |
| 16 | §6 The Clinical Evidence | grid plus interval measures | Six published quantities a funder can check, each with its interval and its source |

### Required contents of each file

Identical to stage 1, with the source fence marked ```d2.

### Anti-defect requirements

- A grid is a grid. Set `grid-rows` and `grid-columns` and let every cell be a
  real cell; do not fake a table with free-floating boxes, which is what
  produces the misaligned column an author then has to nudge by hand.
- A `sql_table` shape carries `field: type` rows only. Prose belongs in a
  `near`-anchored note, not in a field name.
- Containers may nest two deep and no deeper. Three-deep nesting in D2 shrinks
  the innermost label below legibility once the figure is fitted to the measure.
- Every declared style key must exist in D2: `fill`, `stroke`,
  `stroke-width`, `font-color`, `border-radius`, `shadow`. Do not invent
  `background`, `color`, or a CSS class.
- In the TikZ notes, state the cell width and row pitch numerically. Figure 2's
  grid must use one column width for all five columns so the header row cannot
  drift out of register with the body.
- Figure 8 and Figure 11 both draw money. They must not look alike: Figure 8 is
  a three-column ledger read left to right, Figure 11 is a three-tier stack read
  bottom to top with a visible gap where the firewall sits.
- Do not copy `funding/pdac-funding-applications/d2/fig-11-budget-layers`.
  That figure splits cash from contributed value. Figure 8 splits one programme
  across two prices, which is a different claim about the same dollars.
