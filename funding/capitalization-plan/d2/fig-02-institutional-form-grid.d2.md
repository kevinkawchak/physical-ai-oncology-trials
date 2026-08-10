# Figure 2 - This company scored against the report's own institutional-form table

**Type.** d2-type, true grid, eight rows by six columns. **Section.** §1, The
Novel-Performer Case. **Perspective.** *The report's own seven activities and
four institutional forms, with a fifth column the report does not carry.* No
other figure compares this company to anything; Figure 1 selects a clause and
Figure 3 decomposes a cost, and neither asks what kind of institution this is.

**Caption (3 balanced lines, 64 to 69 characters, numbered as printed).**

```
Figure 2. The report's own seven activities, its four institutional
forms, and a fifth column it does not carry. An SBIR firm of 2.6
people is well suited to two of the seven and unsuited to two others.
```

## D2 source

```d2
grid: {
  grid-rows: 8
  grid-columns: 6
  style: {stroke: "#3C7DB2"; stroke-width: 1}

  h0: "Activity"          {style: {fill: "#00417A"; font-color: "#FFFFFF"; bold: true}}
  h1: "University"        {style: {fill: "#00417A"; font-color: "#FFFFFF"; bold: true}}
  h2: "Corporate lab"     {style: {fill: "#00417A"; font-color: "#FFFFFF"; bold: true}}
  h3: "Federal lab"       {style: {fill: "#00417A"; font-color: "#FFFFFF"; bold: true}}
  h4: "New institutions"  {style: {fill: "#00417A"; font-color: "#FFFFFF"; bold: true}}
  h5: "SBIR firm, 2.6 FTE" {style: {fill: "#3C7DB2"; font-color: "#FFFFFF"; bold: true}}

  a0: "Curiosity driven, investigator led"  {style: {fill: "#DCE8F1"}}
  a1: "Well suited"     {style: {fill: "#FFFFFF"}}
  a2: "Less suited"     {style: {fill: "#E9ECEF"}}
  a3: "Partly suited"   {style: {fill: "#FFFFFF"}}
  a4: "By design"       {style: {fill: "#FFFFFF"}}
  a5: "Partly suited"   {style: {fill: "#DCE8F1"}}

  b0: "Larger scale, engineering intensive" {style: {fill: "#DCE8F1"}}
  b1: "Less suited"     {style: {fill: "#E9ECEF"}}
  b2: "Partly suited"   {style: {fill: "#FFFFFF"}}
  b3: "Partly suited"   {style: {fill: "#FFFFFF"}}
  b4: "By design"       {style: {fill: "#FFFFFF"}}
  b5: "Less suited"     {style: {fill: "#E9ECEF"}}

  c0: "Long horizon platform and tools"     {style: {fill: "#DCE8F1"}}
  c1: "Less suited"     {style: {fill: "#E9ECEF"}}
  c2: "Partly suited"   {style: {fill: "#FFFFFF"}}
  c3: "Well suited"     {style: {fill: "#FFFFFF"}}
  c4: "By design"       {style: {fill: "#FFFFFF"}}
  c5: "Partly suited"   {style: {fill: "#DCE8F1"}}

  d0: "Public goods data and infrastructure" {style: {fill: "#DCE8F1"}}
  d1: "Partly suited"   {style: {fill: "#FFFFFF"}}
  d2: "Less suited"     {style: {fill: "#E9ECEF"}}
  d3: "Partly suited"   {style: {fill: "#FFFFFF"}}
  d4: "By design"       {style: {fill: "#FFFFFF"}}
  d5: "Well suited"     {style: {fill: "#00417A"; font-color: "#FFFFFF"}}

  e0: "Mission driven, public good science"  {style: {fill: "#DCE8F1"}}
  e1: "Partly suited"   {style: {fill: "#FFFFFF"}}
  e2: "Less suited"     {style: {fill: "#E9ECEF"}}
  e3: "Partly suited"   {style: {fill: "#FFFFFF"}}
  e4: "By design"       {style: {fill: "#FFFFFF"}}
  e5: "Well suited"     {style: {fill: "#00417A"; font-color: "#FFFFFF"}}

  f0: "Proprietary product development"      {style: {fill: "#DCE8F1"}}
  f1: "Less suited"     {style: {fill: "#E9ECEF"}}
  f2: "Well suited"     {style: {fill: "#FFFFFF"}}
  f3: "Less suited"     {style: {fill: "#E9ECEF"}}
  f4: "By design"       {style: {fill: "#FFFFFF"}}
  f5: "Partly suited"   {style: {fill: "#DCE8F1"}}

  g0: "Workforce training and apprenticeship" {style: {fill: "#DCE8F1"}}
  g1: "Well suited"     {style: {fill: "#FFFFFF"}}
  g2: "Partly suited"   {style: {fill: "#FFFFFF"}}
  g3: "Less suited"     {style: {fill: "#E9ECEF"}}
  g4: "By design"       {style: {fill: "#FFFFFF"}}
  g5: "Less suited"     {style: {fill: "#E9ECEF"}}
}
```

## The fifth column, row by row

The four left-hand columns are the report's own scoring, restated in words
because the report's `+`, `≠` and `×` glyphs do not survive a LaTeX measure at
this size. The fifth column is new, and each cell carries a reason.

| Activity | SBIR firm, 2.6 FTE | Why |
|:--|:--|:--|
| Curiosity driven, investigator led | Partly suited | The founder is the investigator, but a milestone schedule forecloses curiosity by design |
| Larger scale, engineering intensive | Less suited | 2.6 FTE cannot staff an engineering-intensive programme; the site supplies the theatre and the platform |
| Long horizon platform and tools | Partly suited | The interlock rig and the VVUQ suite are platform work, but 33 months is not a long horizon |
| Public goods data and infrastructure | Well suited | Every one of the twelve milestone artifacts is deposited publicly, by plan |
| Mission driven, public good science | Well suited | One mission, one stop condition, and a closure budget in year five |
| Proprietary product development | Partly suited | Commercialization is permitted, but the plan holds no exclusive licence to anything |
| Workforce training and apprenticeship | Less suited | No trainees, no apprenticeship, and no intention to add either inside 33 months |

Two well suited, three partly suited, two less suited. A firm that claimed to be
well suited to all seven would be claiming to be a university, a corporate lab,
a federal lab and an FRO at once, which is the claim the report is written
against.

## TikZ construction notes

Canvas 14.6 by 7.0 cm. One true grid, not free-floating boxes.

| Element | Style token | Placement |
|:--|:--|:--|
| Column 0 cells | `d2celll`, `text width=34mm`, `minimum height=7.2mm` | x = 0 to 3.60 |
| Columns 1 to 5 cells | `d2cell`, `minimum width=20mm`, `minimum height=7.2mm` | Column pitch 2.10 cm from x = 3.60 |
| Header row | `d2cellh` for columns 0 to 4; `d2cellk` for column 5 | y = 0 |
| Body rows | Fill by score: `d2cell` for well or partly, `d2cellg` for less suited | y = -0.78 to -6.24, pitch 0.78 cm |
| Column 5 emphasis | `d2key` for the two well-suited cells | Rows d and e only |
| Column rule | `protoblue`, 0.8 pt | Vertical, between columns 4 and 5, at x = 11.85 |
| Legend | `\legkey` | Three swatches beneath the grid at y = -6.85, x = 0, 3.4, 6.8 |
| In-figure note | `pnote`, `text width=132mm` | x = 0, y = -7.45 |

Grid discipline: all five score columns take the identical 20 mm width, so the
header row cannot drift out of register with the body. Cell height is uniform at
7.2 mm and row pitch is 7.8 mm, leaving a 0.6 mm rule gap that reads as a grid
line rather than as a gap. No cell carries more than two words, which is the
only way a 20 mm cell stays legible at `\tiny`.

The vertical rule at x = 11.85 is the figure's one piece of emphasis: it marks
the boundary between the report's own columns and the column this paper adds.

## Repository sources

- `funding/science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md` - the NOVEL PERFORMERS section and its Table 1, the source of the seven activity rows and the four institutional columns
- `funding/capitalization-plan/diagrams-python/fig-18-operating-topology.md` - the 2.6 FTE the fifth column is scored against
- `funding/capitalization-plan/mermaid/fig-13-twelve-milestone-calendar.md` - the twelve public deposits that make row d well suited
- `funding/pdac-funding-applications/final-apply/sections/sec-09-build-method.tex` - the no-trainee, no-apprenticeship operating fact behind row g
