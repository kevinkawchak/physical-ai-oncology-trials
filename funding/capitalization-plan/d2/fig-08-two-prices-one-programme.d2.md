# Figure 8 - One programme, four layers, two prices, and the delta

**Type.** d2-type, layers with paired measures. **Section.** §3, The $1.6M Gate
and the $3.5M Programme. **Perspective.** *The identical four-layer budget
priced once as a five-year envelope and once as an SBIR award, with the
shortfall carried as a third column rather than absorbed.* No other figure
prices anything twice; Figure 10 stacks capital by source, which says where
money comes from and never what it buys.

**Caption (3 balanced lines, 64 to 71 characters, numbered as printed).**

```
Figure 8. One programme, four layers, and two prices for the same work.
The SBIR route reaches 1,396,000 of direct effort inside a 1,606,000
award; the third column is the 2,104,000 that it does not reach.
```

## D2 source

```d2
direction: right

programme: "One programme, four layers" {
  style: {fill: "#FFFFFF"; stroke: "#3C7DB2"; stroke-width: 2; border-radius: 6}

  L1: "Clinical conduct\nsite, pharmacy, monitoring" {style: {fill: "#00417A"; font-color: "#FFFFFF"}}
  L2: "IND maintenance and\nsafety reporting"        {style: {fill: "#3C7DB2"; font-color: "#FFFFFF"}}
  L3: "Interlock rig, logging,\naudit replay"        {style: {fill: "#DCE8F1"; font-color: "#00417A"}}
  L4: "Verification package\nand archive"            {style: {fill: "#E9ECEF"; font-color: "#000000"}}
}

fiveyear: "Price A, five-year envelope" {
  style: {fill: "#FFFFFF"; stroke: "#00417A"; border-radius: 6}
  A1: "1,600,000"  {style: {fill: "#FFFFFF"}}
  A2: "720,000"    {style: {fill: "#FFFFFF"}}
  A3: "780,000"    {style: {fill: "#FFFFFF"}}
  A4: "400,000"    {style: {fill: "#FFFFFF"}}
  AT: "3,500,000 direct, 60 months" {style: {fill: "#00417A"; font-color: "#FFFFFF"; bold: true}}
}

sbir: "Price B, SBIR route" {
  style: {fill: "#FFFFFF"; stroke: "#3C7DB2"; border-radius: 6}
  B1: "612,000"  {style: {fill: "#DCE8F1"}}
  B2: "268,000"  {style: {fill: "#DCE8F1"}}
  B3: "412,000"  {style: {fill: "#DCE8F1"}}
  B4: "104,000"  {style: {fill: "#DCE8F1"}}
  BT: "1,396,000 direct, 33 months" {style: {fill: "#3C7DB2"; font-color: "#FFFFFF"; bold: true}}
}

delta: "The delta, not reached" {
  style: {fill: "#E9ECEF"; stroke: "#6C757D"; border-radius: 6}
  D1: "988,000"  {style: {fill: "#E9ECEF"}}
  D2: "452,000"  {style: {fill: "#E9ECEF"}}
  D3: "368,000"  {style: {fill: "#E9ECEF"}}
  D4: "296,000"  {style: {fill: "#E9ECEF"}}
  DT: "2,104,000 direct, 27 months" {style: {fill: "#9AA1A8"; font-color: "#000000"; bold: true}}
}

programme -> fiveyear: "priced as one grant"
fiveyear -> sbir: "less what the award reaches"
sbir -> delta: "equals what it does not"
```

## The reconciliation, to the dollar

| Layer | Price A, 5 years | Inside the SBIR award | Delta |
|:--|:--|:--|:--|
| Clinical conduct: site, pharmacy, monitoring | $1,600,000 | $612,000 | $988,000 |
| IND maintenance and safety reporting | $720,000 | $268,000 | $452,000 |
| Interlock rig, logging, audit replay | $780,000 | $412,000 | $368,000 |
| Verification package and archive | $400,000 | $104,000 | $296,000 |
| **Total, direct** | **$3,500,000** | **$1,396,000** | **$2,104,000** |

The four-layer split is reused verbatim from
`funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex`.
Nothing in the left column is re-derived here.

## Why $1,606,000 of award holds only $1,396,000 of work

| Component | Phase I | Phase II | Total |
|:--|:--|:--|:--|
| Direct costs | $266,000 | $1,130,000 | $1,396,000 |
| Indirect, 7.5 percent of direct | $20,000 | $85,000 | $105,000 |
| Fee, 7 percent of direct plus indirect | $20,000 | $85,000 | $105,000 |
| **Total award** | **$306,000** | **$1,300,000** | **$1,606,000** |

The overhead and fee load is $210,000 on $1,396,000 of direct work, that is
15.0 percent. NIH permits an SBIR indirect rate of up to 40 percent of direct
costs without a negotiated rate agreement; this plan claims 7.5 percent, and the
7.5 percent covers insurance, audit and accounting only.

## TikZ construction notes

Canvas 14.6 by 6.6 cm. A three-column ledger read left to right, deliberately
unlike Figure 10, which is a bottom-to-top stack.

| Element | Style token | Placement |
|:--|:--|:--|
| Layer labels L1 to L4 | `d2key`, `d2mid`, `d2soft`, `d2gray`, `text width=44mm` | x = 0, y = 0, -1.05, -2.10, -3.15 |
| Layer container | `d2cont`, `fit` L1 to L4 | `inner sep=7pt` |
| Price A column | `d2cell`, `minimum width=21mm`, `minimum height=8.4mm` | x = 5.35, same four y values |
| Price B column | `d2cellg` with `pablue2` fill, same size | x = 7.75 |
| Delta column | `d2cellg`, same size | x = 10.15 |
| Column headers | `d2title` for A and B, `d2title2` for the delta | Anchored south, 1.2 mm above each column's first cell |
| Totals row | `d2cellh`, `d2cellk`, `d2cell` with `pagrayd` fill | y = -4.35, one per money column |
| Total rule | `pagrayd`, 0.5 pt | Horizontal at y = -3.90, spanning the three money columns only |
| Ratio bars | `\hbarrow` | Three bars at x = 12.75, showing 100, 39.9 and 60.1 percent of the programme |
| Arithmetic check | `pnote` | x = 5.35, y = -5.10, stating that B plus delta equals A on every row |
| In-figure note | `pnote`, `text width=134mm` | x = 0, y = -5.70 |

Column discipline: the three money columns take the identical 21 mm width and
the identical 8.4 mm height, so a reader comparing a row across the three
columns compares equal areas. The total rule spans only the money columns, not
the layer labels, so it reads as an arithmetic rule and not as a table border.

The ratio bars on the right are the figure's one non-tabular element. They are
drawn to a common 3.2 cm full scale, so 39.9 percent is 1.28 cm and 60.1 percent
is 1.92 cm, and the two sum to the full scale exactly.

## Repository sources

- `funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex` - the four-layer $3,500,000 frame, reused verbatim
- `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed/` - the $306,000 and $1,300,000 award amounts
- `funding/RFA-RM-27-001-v2/` - the $700,000 per year, no cost share, no programme income budget statement
- NIH SBIR and STTR indirect-cost policy, the 40 percent allowance this plan does not claim
- 2 CFR 200.414, the de minimis indirect rate the company would otherwise use
