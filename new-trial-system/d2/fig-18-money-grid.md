# Figure 18 - Three asks, four cost layers, two overhead regimes

**Type.** d2-type, grid. **Section.** §6, Funding Proposals.
**Perspective.** *The same four layers of direct work priced three ways and
carried under two overhead regimes, so a funder sees what changes and what does
not when the mechanism changes.* No other figure in this paper carries the
capitalization numbers; Figure 17 plots when the funding artifacts appeared,
Figure 19 routes dollars to work packages, and Figure 20 draws the machinery
that produced the applications.

**Caption (2 balanced lines, 73 and 75 characters, numbered as printed).**

```
Figure 18. One four-layer budget priced three ways, and the same direct work
under a 15.0 percent company load against a 57 percent university rate.
```

## D2 source

```d2
grid: {
  grid-rows: 8
  grid-columns: 5
  style.fill: "#FFFFFF"

  h0: "Cost layer" { style: { fill: "#800020"; font-color: "#FFFFFF" } }
  h1: "SBIR Phase I, 9 months" { style: { fill: "#800020"; font-color: "#FFFFFF" } }
  h2: "SBIR Phase II, 24 months" { style: { fill: "#800020"; font-color: "#FFFFFF" } }
  h3: "Five year direct program" { style: { fill: "#800020"; font-color: "#FFFFFF" } }
  h4: "Same work, university route" { style: { fill: "#800020"; font-color: "#FFFFFF" } }

  l1: "Simulation and verification" { style.fill: "#C9C9C9" }
  a1: "in Phase I scope" { style.fill: "#FFFFFF" }
  b1: "carried forward" { style.fill: "#E2D6D9" }
  c1: "carried forward" { style.fill: "#E2D6D9" }
  d1: "same direct work" { style.fill: "#FFFFFF" }

  l2: "Regulatory and protocol" { style.fill: "#C9C9C9" }
  a2: "in Phase I scope" { style.fill: "#FFFFFF" }
  b2: "carried forward" { style.fill: "#E2D6D9" }
  c2: "carried forward" { style.fill: "#E2D6D9" }
  d2: "same direct work" { style.fill: "#FFFFFF" }

  l3: "Site and device readiness" { style.fill: "#C9C9C9" }
  a3: "not in scope" { style.fill: "#FFFFFF" }
  b3: "in Phase II scope" { style.fill: "#E2D6D9" }
  c3: "carried forward" { style.fill: "#E2D6D9" }
  d3: "same direct work" { style.fill: "#FFFFFF" }

  l4: "Clinical conduct" { style.fill: "#C9C9C9" }
  a4: "not in scope" { style.fill: "#FFFFFF" }
  b4: "partial" { style.fill: "#E2D6D9" }
  c4: "full" { style.fill: "#E2D6D9" }
  d4: "same direct work" { style.fill: "#FFFFFF" }

  l5: "Award" { style: { fill: "#A32A3C"; font-color: "#FFFFFF" } }
  a5: "306000 dollars" { style.fill: "#E2D6D9" }
  b5: "1300000 dollars" { style.fill: "#E2D6D9" }
  c5: "3500000 dollars" { style.fill: "#E2D6D9" }
  d5: "2137000 dollars" { style.fill: "#C9C9C9" }

  l6: "Direct work inside the award" { style: { fill: "#A32A3C"; font-color: "#FFFFFF" } }
  a6: "part of 1396000" { style.fill: "#E2D6D9" }
  b6: "part of 1396000" { style.fill: "#E2D6D9" }
  c6: "3500000 dollars" { style.fill: "#E2D6D9" }
  d6: "1396000 dollars" { style.fill: "#C9C9C9" }

  l7: "Overhead and fee load" { style: { fill: "#A32A3C"; font-color: "#FFFFFF" } }
  a7: "15.0 percent" { style.fill: "#E2D6D9" }
  b7: "15.0 percent" { style.fill: "#E2D6D9" }
  c7: "direct only" { style.fill: "#E2D6D9" }
  d7: "57 percent" { style.fill: "#C9C9C9" }
}
```

## TikZ construction table

Absolute coordinates. Canvas 14.8 by 6.4 cm. A true grid: one row height, five
fixed column widths, no edges.

| Element | Style token | Placement |
|:--|:--|:--|
| Header row | `d2cellh`, height 0.66 cm | y = 0 |
| Column 1 width | 3.80 cm | Cost layer names |
| Columns 2 to 5 width | 2.75 cm each | Values |
| Layer rows 1 to 4 | `d2cellg` in column 1, `d2cell` in columns 2 and 5, `d2celll` in columns 3 and 4 | y = -0.66 down to -2.64, pitch 0.66 cm |
| Money rows 5 to 7 | `d2cellk` in column 1, `d2celll` in columns 2 to 4, `d2cellg` in column 5 | y = -3.30 down to -4.62 |
| Band separator | Charcoal rule, 0.7 pt | Between row 4 and row 5, full grid width |
| Header separator | Charcoal rule, 0.7 pt | Below the header row |
| Bridge strip | `d2mid`, `text width=58mm` | x = 0, y = -5.40 |
| Ratio strip | `d2soft`, `text width=52mm` | x = 6.60, y = -5.40 |
| In-figure note | `pnote` | x = 0, y = -6.05, `text width=140mm` |

The grid is cut once, by a single charcoal rule between row 4 and row 5, which
separates the scope rows from the money rows. Column 5 is the only column
filled in the neutral Mist Gray across the money band, because it is the
counterfactual: the same direct work routed through an institution rather than
through the company.

## Cell values and their sources

| Value | Figure | Source |
|:--|:--|:--|
| SBIR Phase I award | 306,000 dollars over 9 months, 5 milestones | Capitalization plan cover ledger |
| SBIR Phase II award | 1,300,000 dollars over 24 months, 7 milestones | Capitalization plan cover ledger |
| Five-year direct program | 3,500,000 dollars over 60 months, direct | Capitalization plan cover ledger |
| Total award, both phases | 1,606,000 dollars over 33 months | Capitalization plan executive summary |
| Direct work inside the award | 1,396,000 dollars | Capitalization plan executive summary |
| Direct work outside the award | 2,104,000 dollars | Capitalization plan abstract |
| Company overhead and fee load | 15.0 percent against an SBIR allowance of 40 percent not claimed | Capitalization plan executive summary |
| University facilities and administrative rate | 57 percent, giving 2,137,000 dollars for the same direct work | Capitalization plan executive summary |
| Private capital above the firewall | 5,900,000 dollars, 3.67 to one on cash alone | Capitalization plan executive summary |
| Full-time equivalents at full Phase II staffing | 2.6 | Capitalization plan executive summary |

## Edge routing

A grid carries no edges. The two strips beneath the grid are the only non-cell
objects; they sit 0.78 cm below the last row, are 0.85 cm apart horizontally,
and touch no cell border. Cell text is capped at 26 characters and set at
`\tiny` with `align=center`; where a value needs a unit it is written as a word,
so no cell contains a currency symbol that would widen the column beyond the
fixed 2.75 cm.

## Repository sources

- `funding/capitalization-plan/final-capital/publication/LaTeX Source Files.zip` - every money value in the table above, the four-layer budget, the two overhead regimes, and the 3.67 to one bridge ratio
- `funding/pdac-funding-applications/final-apply/publication/LaTeX Source Files.zip` - the ten prior applications, of which application 05 is the SBIR mechanism this grid prices
- `funding/RFA-RM-27-001-v2/LaTeX Source Files.zip` - the NIH budget, milestones and sustainability section the grid is checked against
