# Figure 6 - Where Each Criterion Is Carried

**Platform.** D2. **Native construct.** A grid with shaded status cells.

## Perspective no other figure in this day gives

Table 9 says whether each of the eight criteria is met. It cannot say which part
of the record carries it. The grid does: eight criteria down the side, five parts
of the record across the top, and a fill in each cell for fully, in part, or not
yet. The pattern answers a question the table cannot, which is that the first
three criteria are carried everywhere and the last two only by replies the
company does not yet have.

## Native source

```d2
grid: {
  grid-rows: 9
  grid-columns: 6
  h0: Criterion; h1: "2025 papers"; h2: "Protocols and IND"
  h3: "Applications and plans"; h4: "Repository and builds"; h5: "Outside replies"
  c1: "1. Persistent identifiers"; a1.class: full; b1.class: full; c1x.class: full; d1.class: part; e1.class: none
  c2: "2. Dated versions";         a2.class: full; b2.class: full; c2x.class: full; d2.class: full; e2.class: none
  c3: "3. Open method";            a3.class: full; b3.class: part; c3x.class: part; d3.class: full; e3.class: none
  c4: "4. Credibility assessment"; a4.class: full; b4.class: none; c4x.class: none; d4.class: part; e4.class: none
  c5: "5. Regulatory mapping";     a5.class: none; b5.class: full; c5x.class: part; d5.class: part; e5.class: none
  c6: "6. Independent review";     a6.class: part; b6.class: part; c6x.class: part; d6.class: part; e6.class: none
  c7: "7. Outcome validation";     a7.class: part; b7.class: none; c7x.class: none; d7.class: none; e7.class: none
  c8: "8. Adoption, confirmed";    a8.class: none; b8.class: none; c8x.class: none; d8.class: none; e8.class: none
}
classes: {
  full: {style.fill: "#34495E"}
  part: {style.fill: "#6B7F94"}
  none: {style.fill: "#E9ECEF"}
}
```

## TikZ construction

| Element | Style | Geometry |
|:--|:--|:--|
| Column headers | `d2cellh`, 19 mm wide | y = 0, at x = 3.30, 5.30, 7.30, 9.30, 11.30 |
| Criterion header and labels | `d2cellh`, `d2celll`, 38 mm wide | x = 0.60 |
| Status cells | `d2cell` with fill `fundaccent`, `fundmid` or `fundgrayl` | Rows 0.68 cm apart from y = -0.78 |
| Legend | `\legkey` three times | y = -6.35 |

The fills are written as one `\foreach` row per criterion, five fill names per
row in column order, so a correction to one cell is a one-word edit.

## Why each shade was given

| Criterion | The judgment behind the row |
|:--|:--|
| 1 | Every deposited work has a DOI; the repository itself is versioned but not deposited as one work |
| 3 | The 2025 papers and the repository carry code; the protocols and applications carry method in prose |
| 4 | The credibility score belongs to the 2025 digital twin; the build records carry partial checks |
| 5 | The IND and protocols are mapped to regulation; the applications and repository in part |
| 6 | Every part has model review; no part has human external review |
| 7 | Only the 2025 simulation has been set against a trial, as a chronology |
| 8 | No part is independently confirmed; the outside replies column is where confirmation would appear |

## Caption, exactly as set

> Figure 6. The eight criteria against five parts of the record, shaded by whether each
> part carries the criterion fully, in part, or not yet; the last two rows are open.

The two lines are 85 and 82 characters.

## Where it is used

`../packet/sections/sec-04-what-a-standard-requires.tex`, at the head of §4.1.
