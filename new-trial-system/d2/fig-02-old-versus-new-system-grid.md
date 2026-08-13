# Figure 2 - Prior system against new system, ten axes

**Type.** d2-type, grid. **Section.** §1, Introduction.
**Perspective.** *Ten operating axes on which the prior pancreatic cancer trial
system and the new system differ, with a measured value in each cell rather than
an adjective, so the incompatibility claim is a table a reader can check.* No
other figure in this paper scores the two systems as a whole; Figure 22 scores
only the peer review layer, and Figure 1 establishes the demand without
describing either system's mechanics.

**Caption (2 balanced lines, 72 and 70 characters, numbered as printed).**

```
Figure 2. Ten operating axes on which the prior and new trial systems differ,
each cell carrying a measured value from the author's deposited record.
```

## D2 source

```d2
grid: {
  grid-rows: 11
  grid-columns: 3
  style.fill: "#FFFFFF"

  h0: "Operating axis" { style: { fill: "#800020"; font-color: "#FFFFFF" } }
  h1: "Prior system" { style: { fill: "#800020"; font-color: "#FFFFFF" } }
  h2: "New system" { style: { fill: "#800020"; font-color: "#FFFFFF" } }

  a0: "Document production unit" { style.fill: "#C9C9C9" }
  a1: "Team-month" { style.fill: "#FFFFFF" }
  a2: "Prompt-hour" { style.fill: "#E2D6D9" }

  b0: "IND assembly time" { style.fill: "#C9C9C9" }
  b1: "Months across a group" { style.fill: "#FFFFFF" }
  b2: "Four days, one author" { style.fill: "#E2D6D9" }

  c0: "Peer review entry point" { style.fill: "#C9C9C9" }
  c1: "After completion" { style.fill: "#FFFFFF" }
  c2: "During development" { style.fill: "#E2D6D9" }

  d0: "Review latency" { style.fill: "#C9C9C9" }
  d1: "Seven to eight weeks best case" { style.fill: "#FFFFFF" }
  d2: "Same day, hour scale" { style.fill: "#E2D6D9" }

  e0: "Reviewer count per round" { style.fill: "#C9C9C9" }
  e1: "Two to three humans" { style.fill: "#FFFFFF" }
  e2: "Three model manufacturers" { style.fill: "#E2D6D9" }

  f0: "Virtual trial cost" { style.fill: "#C9C9C9" }
  f1: "Above 120000 dollars per run" { style.fill: "#FFFFFF" }
  f2: "36330 dollars projected" { style.fill: "#E2D6D9" }

  g0: "Provenance record" { style.fill: "#C9C9C9" }
  g1: "Acknowledgement paragraph" { style.fill: "#FFFFFF" }
  g2: "Commit history plus DOI" { style.fill: "#E2D6D9" }

  i0: "Figure format" { style.fill: "#C9C9C9" }
  i1: "Raster, not reusable" { style.fill: "#FFFFFF" }
  i2: "Machine readable, reusable" { style.fill: "#E2D6D9" }

  j0: "Revision granularity" { style.fill: "#C9C9C9" }
  j1: "Whole manuscript" { style.fill: "#FFFFFF" }
  j2: "One file, one commit" { style.fill: "#E2D6D9" }

  k0: "Author count" { style.fill: "#C9C9C9" }
  k1: "Group, plus a CRO" { style.fill: "#FFFFFF" }
  k2: "One, plus the models" { style.fill: "#A32A3C" }
}
```

## TikZ construction table

Absolute coordinates. Canvas 14.6 by 8.4 cm. A true grid: every cell is the
same height, and the three column widths are stated once and never varied.

| Element | Style token | Placement |
|:--|:--|:--|
| Header row | `d2cellh`, height 0.62 cm | y = 0; three cells at x = 0, 4.60, 9.60 |
| Column 1 width | 4.60 cm | Axis names |
| Column 2 width | 5.00 cm | Prior system values |
| Column 3 width | 5.00 cm | New system values |
| Axis cells, rows 1 to 10 | `d2cellg` | Column 1, y = -0.62 to -6.20, pitch 0.62 cm |
| Prior cells, rows 1 to 10 | `d2cell` | Column 2, same rows |
| New cells, rows 1 to 9 | `d2celll` | Column 3, rows 1 to 9 |
| New cell, row 10 | `d2cellk` | Column 3, row 10 only, the single emphasis cell |
| Grid rule | `d2cell` border, 0.4 pt | Uniform, drawn per cell, so no double-weight line appears |
| Column header separator | Charcoal rule, 0.7 pt | Below the header row only |
| Ratio strip | `d2mid`, `text width=44mm` | Below the grid at x = 4.60, y = -7.05, one node |
| In-figure note | `pnote` | x = 0, y = -7.75, `text width=140mm` |

Row 10 is the only cell in the grid with the lighter burgundy fill, because it
is the axis the other nine follow from: a single author with model assistance
is what makes a prompt-hour the production unit.

## Cell values and their sources

| Axis | Prior value | New value | Source of the new value |
|:--|:--|:--|:--|
| Document production unit | Team-month | Prompt-hour | Build record of every stage in this repository |
| IND assembly time | Months across a group | Four days, one author | `trial-ind/final-ind`, deposit Jul 1, 2026 |
| Peer review entry point | After completion | During development | AI peer review study, Abstract |
| Review latency | Seven to eight weeks best case | Same day | AI peer review study, Introduction |
| Reviewer count per round | Two to three humans | Three model manufacturers | AI peer review study, Methods |
| Virtual trial cost | Above 120,000 dollars per run | 36,330 dollars projected | `funding/capitalization-plan/final-capital`, Table 17 |
| Provenance record | Acknowledgement paragraph | Commit history plus DOI | Every stage of this repository |
| Figure format | Raster, not reusable | Machine readable, reusable | The five specification directories in `new-trial-system` |
| Revision granularity | Whole manuscript | One file, one commit | The commit rule in `new-trial-system/sub-prompts/README.md` |
| Author count | Group, plus a CRO | One, plus the models | `new-trial-system/abstracts/README.md`, every entry single-authored |

## Edge routing

A grid carries no edges by construction, which is why the grid was chosen: the
comparison is positional. The only non-cell object is the ratio strip beneath
the grid, which sits 0.85 cm below the last row and is centered on the boundary
between columns 2 and 3, so it touches no cell border. Cell text is capped at
30 characters per line and set at `\tiny` with `align=center`, so no label
overflows its column and no two adjacent labels appear to run together.

## Repository sources

- `new-trial-system/abstracts/README.md` - single authorship across every deposited work from 2024 to 2026
- `trial-ind/final-ind/publication/LaTeX Source Files.zip` - the IND assembly the four-day figure refers to
- `new-trial-system/inputs/AI_Peer_Review_Acceleration_of_LLM_Generated_Glioblastoma_Clinical_Trial_Patient_Matching_ML__FDA_ICH_ISO__and_FastAPI.zip` - review entry point, latency, and reviewer count
- `funding/capitalization-plan/final-capital/publication/LaTeX Source Files.zip` - the projected 36,330 dollar virtual trial against the above-120,000 dollar benchmark
- `new-trial-system/sub-prompts/README.md` - the one-file one-commit rule
