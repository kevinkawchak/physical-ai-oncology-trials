# Figure 12 - What Phase 2 inherits, replaces, and adds

**Type.** d2-type, containers. **Section.** §4, Trial Protocol.
**Perspective.** *The Phase 2 document set drawn as three containers over the
Phase 1 document set, so a reader can see which of the twelve protocol sections
were carried unchanged, which were replaced outright, and which exist only
because the study became randomized.* No other figure in this paper compares the
two protocol documents; Figure 11 compares the clinical progression they
describe, and Figure 10 stays inside one participant in one of them.

**Caption (2 balanced lines, 72 and 75 characters, numbered as printed).**

```
Figure 12. The twelve Phase 1 protocol sections sorted into what Phase 2
carried unchanged, what it replaced, and what randomization made necessary.
```

## D2 source

```d2
direction: down

p1: "Phase 1 protocol, v1.0.0, June 21 2026" {
  style: { fill: "#FFFFFF"; stroke: "#800020" }
  s0: "Compliance"
  s1: "Summary and schema"
  s2: "Introduction"
  s3: "Objectives and endpoints"
  s4: "Design"
  s5: "Population"
  s6: "Intervention"
  s7: "Discontinuation"
  s8: "Assessments"
  s9: "Statistics"
  s10: "Oversight"
  s11: "Additional"
}

carried: "Carried unchanged into Phase 2" {
  style: { fill: "#E2D6D9"; stroke: "#A32A3C" }
  c0: "Compliance"
  c1: "Introduction"
  c2: "Discontinuation"
  c3: "Additional"
}

replaced: "Replaced outright" {
  style: { fill: "#C9C9C9"; stroke: "#6B6B6B" }
  r0: "Summary and schema"
  r1: "Objectives and endpoints"
  r2: "Design"
  r3: "Statistics"
}

added: "Added because the study randomized" {
  style: { fill: "#A32A3C"; stroke: "#800020"; font-color: "#FFFFFF" }
  a0: "Blinded independent central review"
  a1: "Stratified permuted block randomization"
  a2: "Interim analysis and alpha spending"
  a3: "Multicenter site qualification"
}

p1 -> carried: "4 of 12"
p1 -> replaced: "4 of 12"
p1 -> added: "new"
```

## TikZ construction table

Absolute coordinates. Canvas 15.0 by 9.6 cm. Nesting depth two: an outer
container holds leaf boxes, and no container holds another container.

| Element | Style token | Placement |
|:--|:--|:--|
| Phase 1 container | `d2cont`, `fit` its twelve leaves | Top band, x = 0 to 14.60, y = 0 to -2.05 |
| Phase 1 container title | `d2title` | Anchored north west inside the container, 2 mm inset |
| Twelve Phase 1 leaves | `d2gray`, `text width=20mm`, height 0.60 cm | Two rows of six, x = 0.30 + 2.38k for k = 0 to 5, y = -0.70 and y = -1.50 |
| Carried container | `d2cont`, `fit` four leaves | x = 0 to 4.55, y = -3.20 to -6.35 |
| Replaced container | `d2cont2`, `fit` four leaves | x = 5.05 to 9.60, y = -3.20 to -6.35 |
| Added container | `d2cont`, burgundy stroke 0.9 pt | x = 10.10 to 14.60, y = -3.20 to -6.35 |
| Container titles | `d2title` for carried and added, `d2title2` for replaced | Anchored north west, 2 mm inset |
| Carried leaves | `d2soft`, `text width=33mm`, height 0.62 cm | x = 0.30, y = -3.85, -4.55, -5.25, -5.95 |
| Replaced leaves | `d2gray2`, same size | x = 5.35, same four y values |
| Added leaves | `d2key`, same size | x = 10.40, same four y values |
| Three descent edges | `d2edgeb` for carried and added, `d2edged` for replaced | From the Phase 1 container's south edge at x = 2.28, 7.33 and 12.35 to each container's north edge |
| Edge count labels | `d2edge` label, white fill | Midpoint of each descent edge |
| Version strip | `d2mid`, `text width=60mm` | x = 4.20, y = -7.35 |
| In-figure note | `pnote` | x = 0, y = -8.05, `text width=142mm` |

The three lower containers are the same width, 4.55 cm, and start at the same
y, so the sort reads as a partition of one set rather than as three unrelated
groups. The 0.50 cm gutter between containers is wider than the 0.20 cm gutter
between leaves inside a container, which is what makes the nesting legible.

## Cell values and their sources

| Container | Members | Basis |
|:--|:--|:--|
| Carried unchanged | Compliance, Introduction, Discontinuation, Additional | Both protocols share `sec-00`, `sec-02`, `sec-07` and `sec-11` structure and regulatory basis |
| Replaced outright | Summary and schema, Objectives and endpoints, Design, Statistics | Phase 1 is open-label single-arm 3+3 at n = 18; Phase 2 is randomized 1:1 at n = 220 with a PFS primary |
| Added by randomization | BICR, stratified permuted-block randomization, interim analysis with alpha spending, multicenter site qualification | Present in the Phase 2 document only |

The version strip records both documents: Phase 1 protocol v1.0.0 deposited
June 21, 2026 at `doi:10.5281/zenodo.20780121`, and Phase 2 protocol v1.1.0
deposited June 23, 2026 at `doi:10.5281/zenodo.20807027`. Two days separate the
two deposits, which is the quantitative point the figure exists to make.

## Edge routing

Three edges only, all descending, each leaving the Phase 1 container's south
edge at a distinct x that lies directly above the target container's horizontal
center, so no edge is oblique and none can cross another. The vertical run is
1.15 cm and passes through empty canvas. Count labels sit at the midpoint of
each run with a white fill. Leaf boxes inside a container are separated by a
0.08 cm vertical gutter, and the container's `inner sep` of 7 pt keeps every
leaf clear of the container stroke.

## Repository sources

- `trial-protocol/final-protocol/publication/LaTeX Source Files.zip` - the twelve Phase 1 sections, the v1.0.0 version line, and the June 21 deposit DOI
- `trial-phase-2/final-protocol/publication/author/LaTeX Source Files.zip` - the Phase 2 sections, the v1.1.0 version line, the June 23 deposit DOI, and the predicate declaration naming Phase 1
- `new-trial-system/abstracts/README.md` - both deposit dates, from the June 21 and June 23, 2026 entries
