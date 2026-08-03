# Figure 3 - The ten applications as a scored matrix

**Type.** d2-type, grid. **Section.** §2, The Ten Applications.
**Perspective.** *All ten side by side on four attributes.* Figure 1 shows where
the ten come from; this shows what they differ in, and the answer is that they
differ in term and mechanism, not in the science.

**Caption (three balanced lines, 61 to 65 characters each).**

```
Ten applications on four attributes. The science column is constant
by construction: the same trial is described ten times, and only the
mechanism, the term, and the ask change.
```

## D2 source

```d2
grid: {
  grid-columns: 5
  header-recipient: "Recipient" { style.fill: "#00417A"; style.font-color: "#FFFFFF" }
  header-mechanism: "Mechanism" { style.fill: "#00417A"; style.font-color: "#FFFFFF" }
  header-term:      "Term"      { style.fill: "#00417A"; style.font-color: "#FFFFFF" }
  header-ask:       "Ask"       { style.fill: "#00417A"; style.font-color: "#FFFFFF" }
  header-view:      "Lead"      { style.fill: "#00417A"; style.font-color: "#FFFFFF" }

  r01: "01 NIH Pioneer";      m01: "person-based";   t01: "5 years";  a01: "$3.5M";   v01: "surgical"
  r02: "02 ARPA-H";           m02: "milestone";      t02: "36 months"; a02: "$2.1M";  v02: "surgical"
  r03: "03 NSF TIP X-Labs";   m03: "organization";   t03: "5 years";  a03: "$3.5M";   v03: "surgical"
  r04: "04 DOE Genesis";      m04: "mission";        t04: "5 years";  a04: "$3.5M";   v04: "surgical"
  r05: "05 NIH SEED SBIR";    m05: "small business"; t05: "9 + 24 mo"; a05: "$1.6M";  v05: "surgical"
  r06: "06 FNIH AMP";         m06: "consortium";     t06: "5 years";  a06: "$3.5M";   v06: "medical"
  r07: "07 HHMI";             m07: "person-based";   t07: "7 years";  a07: "$0.7M/yr"; v07: "medical"
  r08: "08 NCI CTEP";         m08: "concept review"; t08: "5 years";  a08: "$3.5M";   v08: "medical"
  r09: "09 Convergent FRO";   m09: "time-bound";     t09: "5 years";  a09: "$3.5M";   v09: "medical"
  r10: "10 UCSD Moores";      m10: "institutional";  t10: "one meeting"; a10: "none"; v10: "medical"
}
```

## TikZ construction notes

| Element | Style token | Placement |
|:--|:--|:--|
| Header row | `d2cellh` | Five columns: 4.0cm, 2.9cm, 2.5cm, 2.2cm, 2.3cm; total 13.9cm |
| Recipient column | `d2celll`, left aligned | Only left-aligned column, because it carries the longest strings |
| Mechanism, term, ask | `d2cell` | Centred; the ask column uses `d2cellk` for the two that differ from $3.5M |
| Lead column | `d2soft` surgical, `d2cellg` medical | The two-tone split makes the Set A / Set B boundary visible without a rule |
| Row 10 | `d2cellg` throughout | The only row whose ask is not money |

Every cell is placed by anchor from its left neighbour, so the five columns
cannot drift, and all eleven rows share one `minimum height` so the grid reads
as a table rather than as eleven separate bands.

## Repository sources

- `funding/pdac-funding-applications/applications/README.md` - the recipient table and the perspective split
- Each `funding/pdac-funding-applications/applications/app-*/README.md` - the ask and term rows
