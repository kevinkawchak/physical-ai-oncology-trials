# Figure 3 - The Reserve Against Three Claims on It

**Platform.** D2. **Native construct.** A grid container beside an interval strip
on a shared horizontal scale.

## Perspective no other figure in this day gives

Figures 1 and 2 carry no money. This one carries all of it, and it carries the
one relationship the Treasury ladder exists to express: that a reserve is not a
number but a set of maturities placed against dated claims. A D2 grid holds the
claims as a small table, and a measure strip beside it puts the same claims on a
nine-month rule, so the reader sees the table and the timing in one view.

## Native source

```d2
reserve: {
  shape: grid
  grid-columns: 3
  h1: Claim; h2: Timing; h3: Rung that meets it
  c1: Operating burn, quarter 1; t1: Month 3; r1: Rung A
  c2: Phase I milestone 1 to 3; t2: Month 6; r2: Rung B
  c3: Phase I milestones 4, 5; t3: Month 9; r3: Rung C
  c4: Contingency, unclaimed; t4: Month 12; r4: Rung D
}
horizon: {
  shape: rectangle
  label: "Nine-month SBIR Phase I horizon, four rungs, one liquid sleeve"
}
reserve -> horizon: same scale
```

## TikZ construction

Left panel is a four-row, three-column grid on a 0.76 cm row pitch. Right panel
is a 12-month rule with four rung markers and a shaded nine-month band. The two
panels are separated by a 10 mm corridor that no element spans.

| Element | Style | Geometry |
|:--|:--|:--|
| Header row, three cells | `d2cellh` | Widths 42 mm, 18 mm, 22 mm; `y = 0` |
| Claim rows 1 to 4 | `d2celll` on column 1, `d2cell` on 2, `d2cellk` on 3 | `y = -0.76` to `y = -3.04` |
| Panel title | `ptitle` | `(0,0.62)` |
| Month rule | `axisx` from 0 to 4.8 | At `(9.6,-3.35)` in the shifted scope |
| Month ticks and labels | 0.4 cm per month | Months 0, 3, 6, 9, 12 |
| Nine-month band | `mmband` | From month 0 to month 9, 0.5 cm deep, behind the markers |
| Rung markers A to D | `d2cellk`, 14 mm wide | At months 3, 6, 9, 12 on alternating heights |
| Liquid sleeve marker | `legkey` with `fundpale` | Below the rule, spanning the whole band |
| Share labels | `pnote` | 20, 20, 20, 20, 15 and 5 percent, right of the panel |

Edge routing: there are no edges. The two panels are related by shared month
positions rather than by an arrow, which is why the corridor between them is left
empty and no connector is drawn across it.

## Value provenance

| Value in the figure | Source |
|:--|:--|
| Four rungs at 20 percent each, sleeve at 15, residual at 5 | `../investing/capital-01-treasury-ladder.md`, the five-line table |
| Maturity targets of 3, 6, 9 and 12 months | Same |
| Nine-month horizon | `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed` |
| Phase I milestone grouping | `../briefs/brief-02-sbir-phase-i-readiness.md`, the five-milestone table |

No dollar amount appears in this figure. The reserve is sized as a share rather
than in dollars in the instruction it comes from, and a figure that invented a
balance would state a number the instruction deliberately does not.

## Caption, exactly as printed

```
Figure 3. The corporate reserve as four maturities and one liquid sleeve,
placed against the dated claims each rung is cut to meet, at one scale.
```

Line 1 is 72 characters, line 2 is 70 characters.

## Sources read

- `funding/auto-fund/02Sep26/investing/capital-01-treasury-ladder.md`
- `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed`
- `funding/capitalization-plan/final-capital/capstyle.sty`, for the `d2*` styles
