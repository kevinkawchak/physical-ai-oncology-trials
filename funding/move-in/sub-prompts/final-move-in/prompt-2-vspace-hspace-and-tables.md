# Stage 3, sub-prompt 2 - `\vspace`, `\hspace`, and the table measure audit

## Goal

Apply the author's vertical and horizontal spacing methods so that the fifteen
documents read as one system, and prove by measurement that every table is
exactly the width of the body text.

## The author's spacing vocabulary

| Instrument | Where the author uses it | Value in this paper |
|:--|:--|:--|
| `\apptable` wrapper | Above and below every table | `\addvspace{0.28em}` before, `\addvspace{0.34em}` after, with a closing `\unskip` |
| `\tabcap` | Under every table | A centered italic block at 0.94 of the measure, `\addvspace{0.30em}` after |
| `\mvrule{width}` | Between cover blocks and before the reference list | 0.9 pt, 0.6 pt and 0.5 pt, in that order down the cover |
| `\vspace{-0.35em}` | Immediately before a reference list that follows a rule | Removes the double gap a rule plus a `\bmhead` would otherwise leave |
| `\vspace{0.2em}` and `\vspace{-0.1em}` | Around a rule inside back matter | Keeps the rule optically centered between the two blocks it separates |
| `\hspace` | Never in body prose | Used only inside a table cell to hold a leader, and only where a `p` column cannot |
| `\enspace` | Between cover badges and after the keyword label | The parent's value, unchanged |
| `\;\vert\;` separators | Between cover identity fields | Set with thin spaces so the rule reads as one line |

The invariant is that spacing is applied through a named macro. A bare
`\vspace` in a section file is a stage 3 defect, with two exceptions that the
parent build also allows: the pair around a back-matter rule, and the negative
skip before the reference list.

## The table measure audit

For every table in the paper, compute:

```
sum(fixed column widths) + 2 * ncols * tabcolsep  <=  textwidth
```

with `\tabcolsep` at 4.6 pt and `\textwidth` at 477.6 pt. The `Y` column
absorbs the remainder. Three failures are looked for specifically, because each
one appeared in a parent build:

1. A bold header wider than any body cell in its column.
2. A column holding an unbreakable token, a DOI or a file path, narrower than
   that token.
3. A table that reports overfull by exactly one interword space, 2.74 pt, which
   is the `\apptable` closing `\unskip` missing.

## Acceptance

- `grep -c 'Overfull' main.log` is 0.
- Every table's left and right edges align with the body measure in the
  compiled PDF.
- `grep -n '\\vspace' sections/*.tex` returns only the permitted exceptions.

## Commit

Folded into the seventeen section commits of stage 3, then verified in the
error pass.
