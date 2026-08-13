# Figure 22 - Peer review economics on six axes

**Type.** d2-type, grid. **Section.** §7, AI Peer Review.
**Perspective.** *The economics of one review round under both regimes, with the
ratio computed in each cell, so the claim that the prior system cannot absorb
the new system's output volume is arithmetic rather than assertion.* No other
figure in this paper computes ratios; Figure 21 shows the two clocks, Figure 23
shows the concurrency inside one new-system round, and Figure 2 compares the
systems at large without isolating review.

**Caption (2 balanced lines, 71 and 73 characters, numbered as printed).**

```
Figure 22. Six economic axes of one review round under both regimes, with
the ratio in each cell and the volume the prior system would have to absorb.
```

## D2 source

```d2
grid: {
  grid-rows: 8
  grid-columns: 4
  style.fill: "#FFFFFF"

  h0: "Economic axis" { style: { fill: "#800020"; font-color: "#FFFFFF" } }
  h1: "Prior system, human review" { style: { fill: "#800020"; font-color: "#FFFFFF" } }
  h2: "New system, AI review" { style: { fill: "#800020"; font-color: "#FFFFFF" } }
  h3: "Ratio" { style: { fill: "#800020"; font-color: "#FFFFFF" } }

  a0: "Latency, one round" { style.fill: "#C9C9C9" }
  a1: "7 to 8 weeks best case" { style.fill: "#FFFFFF" }
  a2: "Same day" { style.fill: "#E2D6D9" }
  a3: "about 50 to 1" { style.fill: "#A32A3C" }

  b0: "Typical journal processing" { style.fill: "#C9C9C9" }
  b1: "Several months, 1 to 2 rounds" { style.fill: "#FFFFFF" }
  b2: "Hours, unlimited rounds" { style.fill: "#E2D6D9" }
  b3: "about 300 to 1" { style.fill: "#A32A3C" }

  c0: "Reviewers per round" { style.fill: "#C9C9C9" }
  c1: "2 to 3 humans" { style.fill: "#FFFFFF" }
  c2: "3 model manufacturers" { style.fill: "#E2D6D9" }
  c3: "1 to 1 by count" { style.fill: "#C9C9C9" }

  d0: "Entry point" { style.fill: "#C9C9C9" }
  d1: "After completion" { style.fill: "#FFFFFF" }
  d2: "During development" { style.fill: "#E2D6D9" }
  d3: "not comparable" { style.fill: "#C9C9C9" }

  e0: "Marginal cost per round" { style.fill: "#C9C9C9" }
  e1: "Reviewer time, unpriced" { style.fill: "#FFFFFF" }
  e2: "Inference cost, tens of dollars" { style.fill: "#E2D6D9" }
  e3: "orders of magnitude" { style.fill: "#A32A3C" }

  f0: "Artifacts absorbed per year" { style.fill: "#C9C9C9" }
  f1: "2 to 6 per author" { style.fill: "#FFFFFF" }
  f2: "over 30 deposited in 2026" { style.fill: "#E2D6D9" }
  f3: "about 6 to 1" { style.fill: "#A32A3C" }

  g0: "Corrections applied before release" { style: { fill: "#A32A3C"; font-color: "#FFFFFF" } }
  g1: "None, review follows release" { style.fill: "#C9C9C9" }
  g2: "Every round, before deposit" { style.fill: "#E2D6D9" }
  g3: "the decisive axis" { style: { fill: "#800020"; font-color: "#FFFFFF" } }
}
```

## TikZ construction table

Absolute coordinates. Canvas 14.6 by 6.2 cm. One row height, four fixed column
widths, no edges.

| Element | Style token | Placement |
|:--|:--|:--|
| Header row | `d2cellh`, height 0.64 cm | y = 0 |
| Column 1 width | 4.40 cm | Axis names |
| Columns 2 and 3 width | 4.00 cm each | Regime values |
| Column 4 width | 2.20 cm | Ratio |
| Axis cells, rows 1 to 6 | `d2cellg` | Column 1, y = -0.64 down to -3.84, pitch 0.64 cm |
| Prior cells | `d2cell` | Column 2, same rows |
| New cells | `d2celll` | Column 3, same rows |
| Ratio cells | `d2cellk` where a ratio exists, `d2cellg` where it does not | Column 4, same rows |
| Row 7, decisive axis | `d2cellk` in column 1, `d2cellg` in column 2, `d2celll` in column 3, `d2cell` burgundy fill in column 4 | y = -4.48 |
| Header separator | Charcoal rule, 0.7 pt | Below the header row |
| Decisive-row separator | Charcoal rule, 0.7 pt | Above row 7 |
| Volume strip | `d2mid`, `text width=56mm` | x = 0, y = -5.30 |
| Method strip | `d2soft`, `text width=48mm` | x = 6.20, y = -5.30 |
| In-figure note | `pnote` | x = 0, y = -5.95, `text width=138mm` |

Row 7 is separated from the six economic rows by its own charcoal rule, because
it is not an economic axis: it is the reason the economics matter. A regime
whose review arrives after release cannot apply a correction to the released
object, whatever its cost or latency.

## Cell values and their sources

| Axis | Value | Source |
|:--|:--|:--|
| Latency, best case | 7 to 8 weeks for a prior-system round | AI peer review study, Introduction, citing the 1990s to 2000s electronic transition |
| Typical journal processing | Several months, 1 to 2 rounds; faster online journals about one month | AI peer review study, Introduction |
| Reviewers per round | 2 to 3 humans against 3 model manufacturers | AI peer review study, Methods, triple review by Sonnet, GPT 5.1 and Grok 4.1 |
| Entry point | After completion against during development | AI peer review study, Abstract |
| Marginal cost per round | Reviewer time unpriced against inference cost in tens of dollars | AI peer review study, efficiency metric reference value of 35 dollars |
| Artifacts absorbed per year | Over 30 works deposited across 2026 | `new-trial-system/abstracts/README.md` |
| Corrections before release | Every round, before deposit | AI peer review study, Abstract and Conclusions |

## Edge routing

A grid carries no edges. The two strips beneath the grid sit 0.82 cm below row
7, are 1.80 cm apart horizontally, and touch no cell border. Ratio cells carry
at most 18 characters so the 2.20 cm column never widens. Where an axis has no
meaningful ratio, the cell says so in words rather than carrying a dash, because
a dash in a ratio column reads as a missing value rather than as a deliberate
one.

## Repository sources

- `new-trial-system/inputs/AI_Peer_Review_Acceleration_of_LLM_Generated_Glioblastoma_Clinical_Trial_Patient_Matching_ML__FDA_ICH_ISO__and_FastAPI.zip` - latency, reviewer count, entry point, the efficiency reference values, and the review-during-development principle
- `new-trial-system/abstracts/README.md` - the 2026 deposit count behind the artifacts-per-year axis
- `funding/RFA-RM-27-001-v2/LaTeX Source Files.zip` - the scheduled mid-project review milestones that make the entry-point row operational for a funding application
