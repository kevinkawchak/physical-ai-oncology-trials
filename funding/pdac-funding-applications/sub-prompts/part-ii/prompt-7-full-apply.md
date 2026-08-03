## prompt-7-full-apply

**Stage.** PART II, Stage 7 of 8. **Output.** `funding/pdac-funding-applications/full-apply/`.

### Objective

Turn the skeleton into the full paper. Every `\draftinstr` is resolved and
deleted; every placeholder figure becomes a complete TikZ figure; every table is
populated with author-source quantitative data.

### Content requirements

1. Resolve each bracketed instruction against the exact file it names, and say
   in prose what that file supplies.
2. All twenty figures drawn in full: six mermaid-type, three plantuml-type, four
   d2-type, three diagrams-python-type, four graphviz-type.
3. At least twelve tables, each exactly the body-text width, each column
   `>{\raggedright\arraybackslash}p{...}`, with column widths tuned to the text
   they actually carry rather than split evenly.
4. Body text at roughly 75,000 characters.

### Figure verification, run twice (explicit requirement)

For every one of the twenty figures, verify and record:

- **a)** No text box overlaps another box, and no arrow passes through a label.
  Check by comparing each node's placement against its neighbours' widths.
- **b)** Where a curved edge is used, the looseness is stated explicitly - `bend
  left=NN`, `bend right=NN`, or a `to[out=,in=,looseness=]` triple - and the
  value is small enough that the curve does not re-enter another node.
- **c)** Box-to-box spacing is at least 6mm on the minor axis and 10mm on the
  major axis, so no two frames touch.

Run the pass once, fix, then run it again from the top. Figures must be equally
complex and equally complete throughout: a figure late in the paper may not be
thinner than one early in the paper.

### Column-width method carried from the parent work

Set every table with `xltabular` at `\textwidth`, give the widest prose column
the residual `X`, and give fixed columns the width their longest realistic cell
needs plus one `\tabcolsep`. Header rows are Corporate Blue with white bold
text; body rows are white with a light rule.

### Commits

Twelve section commits plus main, style, bib, README, error-fix, and zip. Push
each immediately.
