## Stage 7 sub-prompt - full-capital

[![Stage](https://img.shields.io/badge/Stage-7%20of%208-00417A.svg)](.)
[![Output](https://img.shields.io/badge/Output-..%2Ffull--capital-3C7DB2.svg)](../../full-capital)
[![Figures](https://img.shields.io/badge/Figures-20%20drawn-6C757D.svg)](../../full-capital)
[![Tables](https://img.shields.io/badge/Tables-19-6C757D.svg)](../../full-capital)
[![Commits](https://img.shields.io/badge/Commits-16-9AA1A8.svg)](.)

### Instruction

Resolve every `\draftinstr` from stage 6 by reading the file it names, and draw
all twenty figures in TikZ. Nothing bracketed survives this stage.

### Deliverables and commit order

The same sixteen-commit order as stage 6, against
`funding/capitalization-plan/full-capital/`.

### Column-width method, taken from the parent work

`funding/pdac-funding-applications/final-apply` sets every table to
`\textwidth` and allocates columns by measuring the longest unbreakable token in
each, not by dividing the measure evenly. The method is:

1. Every table is `tabularx` at `\textwidth`. Exactly one column is `X` and it
   is the column carrying the longest prose. All others are fixed `p{}`.
2. A fixed column's width is the width of its widest atomic cell content plus
   `2\tabcolsep`, rounded up to the next 0.1 cm. A currency column holding
   `$1,300,000` needs 1.9 cm; a two-digit index column needs 0.7 cm; a
   month range such as `10 to 13` needs 1.5 cm.
3. Every fixed column is prefixed `>{\raggedright\arraybackslash}`, without
   exception, so no cell shows a stretched interword gap.
4. The sum of the fixed widths plus `2\tabcolsep` per column must leave the `X`
   column at least 3.2 cm, or the table is re-cut with one fewer column.

### Figure verification, run twice

Every figure is checked against this list, then checked again after the whole
stage compiles. Both passes are recorded in the stage README.

- **a) No text, box, or arrow overlap.** For every pair of nodes on the same
  rank, the gap between their bounding boxes is at least 4 mm. For every edge,
  no unrelated node lies within 2 mm of the edge path. Every edge label carries
  `fill=protowhite` and `inner sep=1.5pt` so it punches a hole in the line it
  labels rather than sitting on top of it.
- **b) Curved arrows.** Any `to[bend left=N]` or `bend right=N` states `N`
  explicitly. `N` is 12 to 22 for a short hop between adjacent nodes and 28 to
  40 for a return edge that must clear one intervening node. A bend below 10 is
  indistinguishable from a straight line and is replaced by one; a bend above 45
  re-enters the node band above and is replaced by an orthogonal route.
- **c) Spacing between boxes.** Horizontal pitch is at least the node text width
  plus 6 mm. Vertical pitch is at least the node height plus 5 mm. Cluster
  `inner sep` is 6 to 7 pt and no node touches its cluster border.

### Complexity floor

Every figure must carry, at minimum: a title or an in-figure panel label, at
least eight labelled elements, at least one quantity taken from an author
source, and one italic in-figure note stating what the figure does not show.
A figure that carries only boxes and arrows is under-specified for this paper
and is redrawn.

### Anti-defect requirements carried from stages 1 to 5

The five platform sub-prompts each list the defects specific to their
vocabulary. All of them apply again here, in TikZ:

- No rank carries more than five nodes; deeper structure goes into a `fit`
  cluster rather than into a longer row.
- Direction is chosen per figure. A timeline, a ledger, or a pipeline is drawn
  left to right. Only a genuine hierarchy is drawn top down.
- Node coordinates are absolute and are listed in the figure's own comment
  header, so adding an element later moves nothing that was already placed.
- Every `\foreach` list ends without a trailing comma, and every macro used
  inside one is expandable at the point TikZ reads it.
