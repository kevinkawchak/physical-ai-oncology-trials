# Figure 11 - Three capital tiers and the two gaps between them

**Type.** d2-type, nested containers. **Section.** §4, Non-Dilutive to Dilutive
Bridge. **Perspective.** *Where every dollar in the plan comes from, stacked by
tier, with the firewall drawn as physical separation rather than as a policy
sentence.* No other figure draws capital by source; Figure 8 draws it by
purpose, and Figure 10 draws the rules that govern movement between these tiers.

**Caption (three balanced lines, 62 to 64 characters).**

```
Three capital tiers and the two gaps between them. Federal money
at the base, contributed value unpriced in the middle, private
capital above, and 5,900,000 against 1,606,000 is 3.67 to one.
```

## D2 source

```d2
direction: up

tier1: "Tier 1, non-dilutive federal, 1,606,000" {
  style: {fill: "#FFFFFF"; stroke: "#00417A"; stroke-width: 2; border-radius: 6}
  P1: "SBIR Phase I\n306,000, months 1 to 9"    {style: {fill: "#00417A"; font-color: "#FFFFFF"}}
  P2: "SBIR Phase II\n1,300,000, months 10 to 33" {style: {fill: "#00417A"; font-color: "#FFFFFF"}}
  P3: "Foundation cash\n0, none sought in this plan" {style: {fill: "#E9ECEF"}}
}

gapA: "Firewall 1, 13 CFR 121.702 ownership test" {
  style: {fill: "#FFFFFF"; stroke: "#3C7DB2"; stroke-dash: 4}
}

tier2: "Tier 2, contributed value, unpriced" {
  style: {fill: "#E9ECEF"; stroke: "#6C757D"; border-radius: 6}
  C1: "Investigational drug supply"      {style: {fill: "#FFFFFF"}}
  C2: "Theatre and robotic platform time" {style: {fill: "#FFFFFF"}}
  C3: "Pathology and specimen handling"  {style: {fill: "#FFFFFF"}}
  C4: "Regulatory cross reference"       {style: {fill: "#FFFFFF"}}
}

gapB: "Firewall 2, 21 CFR part 54 disclosure" {
  style: {fill: "#FFFFFF"; stroke: "#00417A"; stroke-dash: 4}
}

tier3: "Tier 3, dilutive private, 5,900,000" {
  style: {fill: "#FFFFFF"; stroke: "#3C7DB2"; stroke-width: 2; border-radius: 6}
  S1: "Seed SAFE\n900,000, after M8"       {style: {fill: "#DCE8F1"; font-color: "#00417A"}}
  S2: "Option pool\n10 percent, after M10" {style: {fill: "#DCE8F1"; font-color: "#00417A"}}
  S3: "Series A\n5,000,000, after M12"     {style: {fill: "#3C7DB2"; font-color: "#FFFFFF"}}
}

tier1 -> gapA: "no equity may cross downward"
gapA -> tier2: "contributed value is never cash"
tier2 -> gapB: "no investigator interest may cross"
gapB -> tier3: "raised only after a milestone closes"
```

## The three tiers, with their instruments and dates

| Tier | Instrument | Amount | Earliest | Gate that must close first |
|:--|:--|:--|:--|:--|
| 1 | SBIR Phase I award | $306,000 | Month 1 | None; this is the entry point |
| 1 | SBIR Phase II award | $1,300,000 | Month 10 | The four guards of Figure 7 |
| 1 | Foundation cash | $0 | Not sought | Not applicable |
| 2 | Investigational drug supply | Unpriced | Month 6 | Supply agreement, absent today |
| 2 | Theatre and robotic time | Unpriced | Month 12 | CTA, absent today |
| 2 | Pathology and specimen handling | Unpriced | Month 12 | CTA, absent today |
| 2 | Regulatory cross reference | Unpriced | Month 6 | Letter of authorization, absent |
| 3 | Seed SAFE | $900,000 | Month 20 | M8, dose level 1 cleared |
| 3 | Option pool, 10 percent | Non-cash | Month 28 | M10, dose level 2 cleared |
| 3 | Series A | $5,000,000 | Month 33 | M12, closeout deposited |

Tier 2 carries no dollar figure, deliberately, and for the same reason the
parent work gives: no agreement exists, so any valuation would be invented, and
an invented cost-share figure converts a real structural claim into an
unverifiable number.

## Leverage against the annex target

| Quantity | Value |
|:--|:--|
| Federal, tier 1 | $1,606,000 |
| Private cash, tier 3 | $5,900,000 |
| Private to federal ratio | 3.67 to 1 |
| Annex target | at least 3 to 1 |

The FY 2028 annex asks agencies to target at least a 3:1 leverage of private to
federal investment. This plan reaches 3.67:1 on cash alone, before any tier 2
value is counted, which is the argument for counting tier 2 at zero.

## TikZ construction notes

Canvas 14.0 by 8.8 cm. A bottom-to-top stack, deliberately unlike Figure 8's
left-to-right ledger.

| Element | Style token | Placement |
|:--|:--|:--|
| Tier 1 members | `d2key`, `d2key`, `d2gray`, `text width=38mm` | y = -7.10, x = 0, 4.55, 9.10 |
| Tier 1 container | `d2cont`, `fit` P1 to P3 | `inner sep=7pt`, title anchored south west |
| Firewall rule 1 | `pablue1`, 1 pt, dashed | Full width at y = -5.35 |
| Firewall label 1 | `\scriptsize\sffamily\bfseries`, `text=pablue1`, `fill=protowhite` | Anchored west on the rule at x = 0.15 |
| Tier 2 members | `d2cell`, `text width=30mm`, `minimum height=9mm` | y = -3.70, x = 0, 3.55, 7.10, 10.65 |
| Tier 2 container | `d2cont2`, `fit` C1 to C4 | `inner sep=7pt` |
| Firewall rule 2 | `protoblue`, 1.1 pt, dashed | Full width at y = -2.05 |
| Firewall label 2 | Same, `text=protoblue` | Anchored west at x = 0.15 |
| Tier 3 members | `d2soft`, `d2soft`, `d2mid`, `text width=38mm` | y = -0.40, x = 0, 4.55, 9.10 |
| Tier 3 container | `d2cont`, `fit` S1 to S3 | `inner sep=7pt` |
| Tier totals | `d2cellh` for tier 1 and 3, `d2cellg` for tier 2 | Anchored east at x = 13.90, one per tier band |
| Crossing edges | `d2edged`, vertical | Four only, at x = 1.90 and x = 11.60, two per rule |
| Leverage bar | `\hbarrow` pair | x = 0, y = 0.85, showing 1,606,000 against 5,900,000 on a common scale |
| In-figure note | `pnote`, `text width=132mm` | x = 0, y = -8.30 |

Gap discipline: the two firewall rules are the figure's subject, so the vertical
clearance around them is deliberate and equal. Each rule has 8 mm of empty
canvas above and 8 mm below before the nearest container edge, and no label,
node, or edge label sits inside that 16 mm band except the rule's own caption at
the left margin.

Exactly four edges cross the two rules, two per rule, at x = 1.90 and x = 11.60.
Both crossings are vertical, so the crossing point is unambiguous.

## Repository sources

- `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed/` - the tier 1 amounts
- `funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex` - the four unpriced contributed categories and the reason they are unpriced
- `funding/science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md` - the at least 3:1 private to federal leverage target
- `funding/capitalization-plan/mermaid/fig-13-twelve-milestone-calendar.md` - M8, M10 and M12, the three milestones that gate tier 3
- 13 CFR 121.702 and 21 CFR part 54, the two firewalls
