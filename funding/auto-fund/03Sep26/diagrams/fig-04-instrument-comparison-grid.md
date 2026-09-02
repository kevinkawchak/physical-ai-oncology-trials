# Figure 4 - Three Instruments on the Same Eight Attributes

**Platform.** D2. **Native construct.** Three container columns in a grid, with
one attribute label column and no edges between the containers.

## Perspective no other figure in this day gives

Figure 5 draws the financing as a process and Figure 6 draws it as a signing
order. Neither answers the question the day actually asks, which is which of
three instruments to choose. A comparison across a fixed attribute set is a
column layout, and D2's container grid is the only vocabulary in the set that
keeps three columns aligned on a shared row scale without drawing a single
connector between them.

The absence of edges is the point. Three instruments are alternatives, not
stages, and an arrow between them would suggest a sequence that does not exist.

## Native source

```d2
compare: {
  grid-columns: 4
  a0: Attribute; a1: Post-money SAFE; a2: Convertible note; a3: Priced preferred
  r1: Time to first close;      s1: Days;        n1: Days to weeks; p1: Weeks to months
  r2: Legal cost to issuer;     s2: Lowest;      n2: Low;           p2: Highest
  r3: Accrues interest;         s3: No;          n3: Yes;           p3: No
  r4: Has a maturity date;      s4: No;          n4: Yes;           p4: No
  r5: Ownership at signing;     s5: None;        n5: None;          p5: Immediate
  r6: Cap table clarity later;  s6: Good;        n6: Good;          p6: Best
  r7: Governance given away;    s7: None;        n7: None;          p7: Board seat usual
  r8: SBIR ownership answer;    s8: Unchanged;   n8: Unchanged;     p8: Can change
}
```

## TikZ construction

A nine-row, four-column grid. Row pitch is 0.68 cm; the header row is 0.72 cm
deep. The three instrument columns are equal at 26 mm so that no column reads as
favored by width; the attribute column is 40 mm because its labels are the
longest cells in the figure.

| Element | Style | Geometry |
|:--|:--|:--|
| Header, attribute column | `d2cellh`, 40 mm | `(0,0)` |
| Header, three instrument columns | `d2cellh`, 26 mm | `(3.15,0)`, `(5.85,0)`, `(8.55,0)` |
| Attribute rows 1 to 8 | `d2celll`, 40 mm | `y = -0.68` to `y = -5.44` |
| Value cells, rows 1 to 7 | `d2cell`, 26 mm | Same rows, three columns |
| Value cells, row 8 | `d2cellk` on columns 1 and 2, `d2cellg` on column 3 | The row that decides the choice, marked by fill |
| Row 8 emphasis rule | `fundmid`, 0.9 pt | A rule above row 8 across the full grid width |
| Note | `pnote`, `text width=118mm` | Below the grid, two lines |

Edge routing: there are no edges, by design. The three instruments are
alternatives rather than stages, and the grid's shared row scale is the only
relationship the figure asserts.

## Value provenance

| Value in the figure | Source |
|:--|:--|
| All eight attributes and all 24 cells | `../briefs/brief-01-instrument-comparison.md`, the eight-attribute table |
| The row 8 emphasis | 13 CFR 121.702, through `../../02Sep26/forms/form-02-sba-company-registry.md` |

No valuation, cap, discount, minimum, or closing date appears anywhere in this
figure. Those are offering terms, no offering exists, and a figure is the easiest
thing in a document to screenshot out of context.

## Caption, exactly as printed

```
Figure 4. Three candidate instruments across eight attributes at one raise
size, with the row that decides the choice for this company marked in fill.
```

Line 1 is 72 characters, line 2 is 74 characters.

## Sources read

- `funding/auto-fund/03Sep26/briefs/brief-01-instrument-comparison.md`
- `funding/auto-fund/02Sep26/forms/form-02-sba-company-registry.md`
- `funding/capitalization-plan/final-capital/capstyle.sty`, for the `d2*` styles
