# Stage 2, sub-prompt 2 - column width optimization

## Goal

Every table must be exactly the width of the body text and must look right at
that width. This sub-prompt states the author method taken from
`funding/pdac-funding-applications` and applies it to every table in the paper.

## The author method

1. **The measure is fixed.** `\textwidth` under the geometry block of
   `movestyle.sty`. Every table is a `tabularx` set to `{\textwidth}`, so the
   engine absorbs rounding into the `X` column rather than letting the table
   drift past the margin.
2. **One `X` column per table, and it is the prose column.** The column whose
   cells carry sentences takes `Y`, which is `X` with
   `>{\raggedright\arraybackslash}`. Fixed columns carry labels, dates, dollar
   figures and identifiers.
3. **Every fixed column is `>{\raggedright\arraybackslash}p{...}`.** Without the
   prefix a narrow `p` column justifies, and a justified 2 cm column shows word
   gaps wide enough to read as a defect. The audit is a grep: every `p{` in a
   section file must be preceded by `\raggedright\arraybackslash`.
4. **Width is set from the longest unbreakable token, not from the average.**
   A column holding `10.5281/zenodo.21887807` needs the width of that string at
   the body size, or the DOI overflows. A column holding `2026` needs 1.0 cm.
   Measure the longest token, then add `2\tabcolsep` plus 0.05 cm of slack.
5. **The bold header cell counts.** A header set in bold is wider than the same
   words in the body face. Four columns in the parent build overflowed for this
   reason alone. Where the header is the widest cell, either widen the column or
   shorten the header; do not let it decide the width by accident.
6. **`\arraystretch` at 1.16 and `\tabcolsep` at 4.6 pt.** These are the parent
   values and they are not changed, because a table that matches the parent's
   vertical rhythm sits correctly against the parent's paragraph spacing.
7. **`\apptable` wraps every table.** Its closing `\unskip` removes the
   interword space that the newline after `\end{tabularx}` would otherwise
   contribute. Without it every table sets one space, 0.25 em or 2.74 pt, past
   the right margin and reports as overfull.

## Width budget at this geometry

The body measure is 6.6 inches, 16.76 cm, 477.6 pt. A five-column table with
`\tabcolsep` at 4.6 pt spends 46 pt on gutters, leaving 431.6 pt, 15.14 cm, for
content. Typical allocations:

| Shape | Fixed columns | `Y` column |
|:--|:--|:--|
| Number, label, prose | 0.8 cm, 3.2 cm | remainder |
| Role, FTE, salary, prose | 4.2 cm, 1.3 cm, 2.0 cm | remainder |
| Document, part, statute, prose | 1.0 cm, 1.2 cm, 3.0 cm | remainder |
| Source, date, result, control, limitation | 2.6 cm, 1.5 cm, 1.6 cm, 1.5 cm | remainder |
| Term, expansion, term, expansion | 1.5 cm, `Y`, 1.5 cm, `Y` | two `Y` columns |

## Acceptance

- No `Overfull \hbox` in `main.log` attributable to a table.
- `grep -o 'p{[0-9.]*cm}' sections/*.tex | wc -l` equals the count of
  `raggedright\arraybackslash` occurrences immediately preceding them.
- Every table's first column left edge aligns with the body text left edge in
  the compiled PDF.

## Commit

Folded into the seventeen section commits of sub-prompt 1, then verified in the
stage error pass.
