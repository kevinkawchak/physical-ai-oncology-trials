# Figure 3 - The same direct work priced under three overhead regimes

**Type.** graphviz-type, record nodes side by side. **Section.** §1, The
Novel-Performer Case. **Perspective.** *What $1,396,000 of identical direct work
costs a funder through a university, through the full SBIR allowance, and
through this company's budgeted rate.* No other figure decomposes a cost;
Figure 8 compares two programme scopes, which is a different question about a
different pair of numbers.

**Caption (three balanced lines, 64 to 65 characters).**

```
The same 1,396,000 of direct work priced three ways. A university
F and A rate at 57 percent adds 741,000, the full SBIR allowance
adds 695,000, and the rate this plan claims adds 210,000 in all.
```

## Graphviz source

```dot
digraph indirect {
  rankdir=LR;
  ranksep=0.9;
  nodesep=0.55;
  node [shape=record, fontname="Times", fontsize=10,
        style=filled, fillcolor="#FFFFFF", color="#000000"];
  edge [fontname="Times", fontsize=9, color="#6C757D"];

  work [label="{Direct work, identical in all three|\
Personnel, 2.6 FTE | $842,000|\
Site clinical conduct | $288,000|\
Compute and rig hardware | $96,000|\
Monitoring and DSMB | $86,000|\
Consultants and travel | $84,000|\
TOTAL DIRECT | $1,396,000}",
        fillcolor="#DCE8F1", color="#00417A"];

  uni [label="{Route A, university, 57 percent F and A|\
Direct | $1,396,000|\
Less equipment, off MTDC | -$96,000|\
MTDC base | $1,300,000|\
F and A at 57 percent | $741,000|\
Fee | none|\
TOTAL TO THE FUNDER | $2,137,000}",
        fillcolor="#E9ECEF", color="#6C757D"];

  full [label="{Route B, SBIR at the 40 percent allowance|\
Direct | $1,396,000|\
Indirect at 40 percent | $558,000|\
Subtotal | $1,954,000|\
Fee at 7 percent | $137,000|\
TOTAL TO THE FUNDER | $2,091,000}",
        fillcolor="#E9ECEF", color="#6C757D"];

  plan [label="{Route C, this plan, 7.5 percent|\
Direct | $1,396,000|\
Indirect at 7.5 percent | $105,000|\
Subtotal | $1,501,000|\
Fee at 7 percent | $105,000|\
TOTAL TO THE FUNDER | $1,606,000}",
        fillcolor="#00417A", fontcolor="#FFFFFF", color="#00417A"];

  work -> uni  [label="through an institution"];
  work -> full [label="allowance claimed in full"];
  work -> plan [label="allowance not claimed"];

  uni  -> plan [label="premium $531,000, 33.1 percent", style=dashed, constraint=false];
  full -> plan [label="premium $485,000, 30.2 percent", style=dashed, constraint=false];
}
```

## The arithmetic, checked

| Line | Route A, university | Route B, full allowance | Route C, this plan |
|:--|:--|:--|:--|
| Direct costs | $1,396,000 | $1,396,000 | $1,396,000 |
| Equipment removed from the base | -$96,000 | not applicable | not applicable |
| Overhead base | $1,300,000 MTDC | $1,396,000 direct | $1,396,000 direct |
| Overhead rate | 57 percent | 40 percent | 7.5 percent |
| Overhead | $741,000 | $558,000 | $105,000 |
| Fee at 7 percent | none | $137,000 | $105,000 |
| **Total to the funder** | **$2,137,000** | **$2,091,000** | **$1,606,000** |
| Premium over route C | $531,000 | $485,000 | zero |
| Premium as a percentage | 33.1 | 30.2 | zero |

Every column starts from the same $1,396,000 and every total is the sum of its
own column. $741,000 is 57 percent of $1,300,000. $558,000 is 40 percent of
$1,396,000. $137,000 is 7 percent of $1,954,000. $105,000 is 7.5 percent of
$1,396,000, and the fee is 7 percent of $1,501,000.

## Why the three rates are the three rates

| Rate | Where it comes from | Why it applies or does not |
|:--|:--|:--|
| 57 percent MTDC | A typical negotiated research F and A rate at a US research university | It is what the same work costs if a university is the applicant |
| 40 percent of direct | The NIH SBIR and STTR indirect allowance available without a negotiated rate agreement | It is available to this company and is not claimed |
| 7.5 percent of direct | This plan's own figure, covering insurance, audit and accounting only | It is what the company can document |
| 15 percent MTDC | The 2 CFR 200.414 de minimis rate | Not used, because the SBIR allowance is more favourable and still is not claimed |

The company has no negotiated indirect-cost rate agreement, no facilities to
recover, and no administrative layer to fund. That is a structural fact about a
2.6 FTE firm and not a concession; the figure shows what it is worth to a
funder.

## TikZ construction notes

Canvas 14.6 by 8.4 cm. Four `gvbox` records in a row, drawn as ruled cells
stacked vertically inside a single outline, which is how Graphviz renders a
record node.

| Element | Style token | Placement |
|:--|:--|:--|
| Work record, six rows | `gvcells` body, `gvcellh` header | x = 0, rows at y = 0 down to -3.0, pitch 0.50 cm |
| Route A record, six rows | `gvcellg` body, `gvcellh` header | x = 3.75 |
| Route B record, five rows | `gvcellg` body, `gvcellh` header | x = 7.50 |
| Route C record, five rows | `gvcells` body, `gvcellh` header, blue total row | x = 11.25 |
| Cell width | Label 21 mm plus value 12 mm | Uniform across all four records |
| Total rows | `gvcellh` for route C, `gvcell` with `pagraym` for A and B | Bottom row of each record, separated by a 0.6 pt rule |
| Solid edges | `gvedgeb` | Work to each of the three routes, at y = -1.50, horizontal |
| Premium edges | `gvedged` | A to C and B to C, both `bend right=26`, routed beneath all four records |
| Premium labels | `\tiny`, `fill=protowhite`, `inner sep=1.5pt` | At the midpoint of each dashed edge |
| Ratio bars | `\vbarcol` | Three columns at x = 1.30, 2.30, 3.30 in a panel at y = -6.60, heights 2.14, 2.09, 1.61 cm |
| In-figure note | `pnote`, `text width=134mm` | x = 0, y = -7.85 |

Record discipline: every record uses the same 21 mm label column and 12 mm value
column, so the four totals sit on one vertical line at x offset 33 mm within
each record and can be compared without reading a label.

Edge routing: the three solid edges are horizontal and at one y, so they cannot
cross. The two dashed premium edges are the only edges that travel backwards;
both take `bend right=26` and are routed below y = -3.60, clear of every record
by at least 6 mm.

## Repository sources

- `funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex` - the $700,000 per year, no cost share frame the direct column is cut from
- `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed/` - the $306,000 and $1,300,000 awards route C sums to
- `funding/science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md` - the deference to incumbents that consume the funding, which is the finding this figure prices
- NIH SBIR and STTR indirect-cost allowance, 40 percent of direct costs without a negotiated rate
- 2 CFR 200.414, the de minimis indirect rate
