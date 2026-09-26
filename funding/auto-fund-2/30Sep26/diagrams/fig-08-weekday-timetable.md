# Figure 8 - The Early-Rise Week as a Timetable

**Platform.** D2. **Native construct.** A grid of blocks against weekdays.

## Perspective no other figure in this day gives

Table 14 gives one day, block by block. It does not show the week. The grid does:
nine blocks down the side, Monday to Friday across the top, and the two federal
windows filled with each day's theme. A new team member sees in one look that
every day has the same shape and only the federal windows change.

## Native source

```d2
week: {
  grid-rows: 10
  grid-columns: 6
  h0: "Pacific time"; h1: Monday; h2: Tuesday; h3: Wednesday; h4: Thursday; h5: Friday
  r1: "5:00 Start"        ; r1*: "Read replies"
  r2: "5:30 Approval"     ; r2*: "Approve"
  r3: "6:00 Federal"      ; m3: "Program offices"; t3: "Regulators, capital"; w3: "Sites, structures"; th3: "Pilots, solicitations"; f3: "Follow-ups, record"
  r4: "8:00 Technical"    ; r4*: "Technical work"
  r5: "11:00 West Coast"  ; r5*: "Institutions"
  r6: "12:00 Break"       ; r6*: "Break"
  r7: "12:30 Federal"     ; same themes as row r3
  r8: "2:00 Build"        ; r8*: "Commit"
  r9: "3:30 Record"       ; r9*: "Record"
}
```

The `r1*` notation stands for the five identical cells of that row, one per
weekday; D2 itself needs each cell written out.

## TikZ construction

| Element | Style | Geometry |
|:--|:--|:--|
| Header row | `d2cellh`, 24 mm wide | y = 0; columns at x = 0.00, 2.55, 4.95, 7.35, 9.75, 12.15 |
| Time labels | `d2celll`, 24 mm | x = 0.00; rows 0.60 cm apart from y = -0.68 |
| Identical blocks | `d2cell` with `fundpale`, `d2cellg`, or `d2cellk` fills | Drawn by one `\foreach` over the five weekdays |
| Federal windows | `d2cell` with `fundaccent` fill and white text | Rows at y = -1.88 and -4.28, one theme per weekday |
| Legend | `\legkey` three times | y = -6.25 |

## Caption, exactly as set

> Figure 8. The early-rise week as a timetable grid, with the two federal windows
> carrying each day's theme and every other block identical from Monday to Friday.

The two lines are 79 and 80 characters.

## Where it is used

`../packet/sections/sec-04-the-early-rise-weekday.tex`, after Table 14.
