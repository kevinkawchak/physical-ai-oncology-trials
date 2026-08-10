# Figure 16 - Six published quantities, with the interval on each

**Type.** d2-type, grid with interval measures. **Section.** §6, The Clinical
Evidence a Funder Is Buying. **Perspective.** *Every number a funder would
check before writing a cheque, each with its comparator, its interval, and the
limitation its own authors stated.* No other figure carries clinical
quantities; Figure 17 traces where each one goes, which is provenance rather
than magnitude.

**Caption (3 balanced lines, 66 to 69 characters, numbered as printed).**

```
Figure 16. Six quantities a reviewer can check against a published
source, each beside its comparator and its stated limitation. Two are
in silico, one is a digital twin, and three are trial or registry.
```

## D2 source

```d2
panel: {
  grid-rows: 7
  grid-columns: 5
  style: {stroke: "#3C7DB2"}

  h0: "Quantity"    {style: {fill: "#00417A"; font-color: "#FFFFFF"; bold: true}}
  h1: "Test arm"    {style: {fill: "#00417A"; font-color: "#FFFFFF"; bold: true}}
  h2: "Comparator"  {style: {fill: "#00417A"; font-color: "#FFFFFF"; bold: true}}
  h3: "Tier"        {style: {fill: "#00417A"; font-color: "#FFFFFF"; bold: true}}
  h4: "Date"        {style: {fill: "#00417A"; font-color: "#FFFFFF"; bold: true}}

  q1: "RASolute 302 median OS, RAS G12" {style: {fill: "#DCE8F1"}}
  a1: "13.2 months" {style: {fill: "#00417A"; font-color: "#FFFFFF"}}
  b1: "6.6 months"  {style: {fill: "#FFFFFF"}}
  c1: "Trial"       {style: {fill: "#FFFFFF"}}
  d1: "2026-05"     {style: {fill: "#FFFFFF"}}

  q2: "QSP simulation median OS, 10 arms" {style: {fill: "#DCE8F1"}}
  a2: "12.8 months" {style: {fill: "#3C7DB2"; font-color: "#FFFFFF"}}
  b2: "5.4 months"  {style: {fill: "#FFFFFF"}}
  c2: "In silico"   {style: {fill: "#E9ECEF"}}
  d2s: "2025-08"    {style: {fill: "#FFFFFF"}}

  q3: "Digital twin median OS, 1000 patients" {style: {fill: "#DCE8F1"}}
  a3: "12.1 months" {style: {fill: "#3C7DB2"; font-color: "#FFFFFF"}}
  b3: "not applicable" {style: {fill: "#E9ECEF"}}
  c3: "Twin"        {style: {fill: "#E9ECEF"}}
  d3: "2025-10"     {style: {fill: "#FFFFFF"}}

  q4: "Digital twin PFS hazard ratio" {style: {fill: "#DCE8F1"}}
  a4: "0.31"        {style: {fill: "#3C7DB2"; font-color: "#FFFFFF"}}
  b4: "not applicable" {style: {fill: "#E9ECEF"}}
  c4: "Twin"        {style: {fill: "#E9ECEF"}}
  d4: "2025-10"     {style: {fill: "#FFFFFF"}}

  q5: "VVUQ credibility score, 55 tests" {style: {fill: "#DCE8F1"}}
  a5: "81.9"        {style: {fill: "#3C7DB2"; font-color: "#FFFFFF"}}
  b5: "V and V 40 gate" {style: {fill: "#FFFFFF"}}
  c5: "Twin"        {style: {fill: "#E9ECEF"}}
  d5: "2025-10"     {style: {fill: "#FFFFFF"}}

  q6: "Empirical triplicate grade 3 plus AE" {style: {fill: "#DCE8F1"}}
  a6: "8.0 percent"  {style: {fill: "#3C7DB2"; font-color: "#FFFFFF"}}
  b6: "25.0 percent" {style: {fill: "#FFFFFF"}}
  c6: "In silico"    {style: {fill: "#E9ECEF"}}
  d6: "2025-07"      {style: {fill: "#FFFFFF"}}
}
```

## The intervals drawn beside the grid

Three quantities carry an interval and three do not. The three that do are drawn
as `\ciband` rules on a common months axis; the three that do not are marked as
point estimates from a deterministic run, which is what they are.

| Quantity | Point | Interval drawn | Why |
|:--|:--|:--|:--|
| RASolute 302 median OS, test | 13.2 | Yes, from the published trial | A randomized readout carries one |
| RASolute 302 median OS, control | 6.6 | Yes, from the published trial | Same |
| QSP simulation median OS | 12.8 | No | A ten-arm deterministic ODE run has no sampling interval |
| Digital twin median OS | 12.1 | No | 1000 synthetic patients, no patient-specific parameters |
| Digital twin PFS hazard ratio | 0.31 | No | Same |
| Empirical triplicate grade 3 plus | 8.0 percent | No | Triplicate, not a sampled cohort |

The ratio the simulation produced in 2025 was 2.4-fold, and the ratio the trial
reported in 2026 was 2.0-fold. That is a chronology observation and a
hypothesis-supporting one. It is not a validation claim, and three differences
are material: 1000 simulated against 241 enrolled, a combination against a
single agent, and KRAS G12C against a primarily G12D and G12V population.

## The stated limitations, carried in the same row

| Source | Limitation, in its own authors' words |
|:--|:--|
| Empirical triplicate, 100,000 patients | The KRAS G12C log favours experimental while the KRAS-mutant report favours control; trial-to-trial variability |
| QSP simulation, 10 arms, 250 ODEs | Assumes no acquired resistance and ideal pharmacodynamics; grade 3 plus consistently high in all arms |
| Digital twin, 1000 patients | No patient-specific PK, PD, or tumour growth parameters; simple Emax model, no immune compartments |
| RASolute 302 | Metastatic and previously treated; silent on the resectable setting |

## TikZ construction notes

Canvas 14.6 by 7.8 cm. A seven by five grid on the left, an interval panel on
the right, separated by a 10 mm corridor.

| Element | Style token | Placement |
|:--|:--|:--|
| Column 0 cells | `d2celll`, `text width=32mm`, `minimum height=7.0mm` | x = 0 |
| Columns 1 to 4 | `d2cell`, `minimum width=17mm`, `minimum height=7.0mm` | Pitch 1.80 cm from x = 3.45 |
| Header row | `d2cellh` | y = 0 |
| Test arm cells | `d2cellk` for the five simulated, `d2key` for RASolute 302 | Column 1 |
| Tier cells | `d2cellg` for in silico and twin, `d2cell` for trial | Column 3 |
| Body rows | y = -0.76 to -4.56, pitch 0.76 cm | Six rows |
| Interval axis | `\axisx{0}{4.2}{-4.95}` plus tick labels at 0, 5, 10, 15 months | Right panel, x offset 11.10 |
| Intervals | `\ciband` | Two rules, test and control, at y = -1.30 and y = -2.30 |
| Point estimates | `\legkey` marker plus a value label | Three at y = -3.10, -3.70, -4.30 |
| Panel titles | `\ptitle` | Two, one per panel, anchored west at y = 0.75 |
| Ratio callout | `d2cellk`, `text width=22mm` | Right panel, y = -5.60, carrying 2.4-fold against 2.0-fold |
| In-figure note | `pnote`, `text width=134mm` | x = 0, y = -6.55 |

Grid discipline: all four value columns take the identical 17 mm width. Column 0
takes 32 mm because it is the only column carrying prose, which is the same
one-wide-column rule the paper's tables follow.

The two panels are separated by 10 mm of empty canvas, and no element spans the
corridor. The interval panel is drawn to a 4.2 cm axis for 0 to 16.8 months, so
13.2 months is 3.30 cm and 6.6 months is exactly half that, which makes the
two-fold relationship legible without reading a number.

## Repository sources

- `funding/pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex` - all six quantities, their comparators, and the four stated limitations, carried unchanged
- RASolute 302, New England Journal of Medicine, 2026, DOI 10.1056/NEJMoa2605555
- `10.5281/zenodo.17001137` - the QSP simulation, 10 arms, 250 ODEs per patient
- `10.5281/zenodo.15735068` - the daraxonrasib identification meta-analysis
- ASME V&V 40 and ICH M15, the credibility framework the 81.9 score is aligned to
- `trial-protocol/` - the resectable setting the RASolute 302 row is silent on
