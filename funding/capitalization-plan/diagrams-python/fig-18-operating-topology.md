# Figure 18 - Where the 2.6 FTE, the compute, and the site functions sit

**Type.** diagrams (python)-type, three clusters across two trust boundaries.
**Section.** §7, Small-Business Operating Plan. **Perspective.** *Who is
employed, who is contracted, who is contributed, and which of the three sits
inside the hospital's network.* No other figure maps employment and contract
status; Figure 6 maps authority, which says who may act and never who is on a
payroll.

**Caption (three balanced lines, 62 to 64 characters).**

```
Three clusters and two trust boundaries. The company employs 2.3
of the 2.6 FTE, contracts 0.3, and pays for none of the six site
functions. No identified data crosses the left trust boundary.
```

## Specification, as a diagrams-library graph

No `.py` file is emitted.

```python
# illustrative only, not executed, no .py file is written to the repository
with Diagram("ChemicalQDevice operating topology, Phase II", direction="LR"):
    with Cluster("ChemicalQDevice premises, San Diego, no PHI"):
        ceo = Node("CEO and sponsor lead, 0.8 FTE")
        eng = Node("Verification engineer, 1.0 FTE")
        crc = Node("Clinical research coordinator, 0.5 FTE")
        gpu = Node("Two GPU workstation nodes")
        arc = Node("Artifact store and hash manifest")
    with Cluster("Contracted, fixed fee, not employed"):
        bio = Node("Biostatistician, 0.2 FTE")
        reg = Node("Regulatory consultant, 0.1 FTE")
        mon = Node("Independent monitor, per visit")
    with Cluster("UC San Diego Moores, PHI, contributed"):
        srg = Node("Operating surgeon")
        thr = Node("Theatre and robotic platform")
        phm = Node("Investigational pharmacy")
        pth = Node("Pathology and specimens")
        llm = Node("On-premises advisory model")
        dsm = Node("DSMB, independent")
    eng >> Edge(label="deploys, signed build") >> llm
    crc >> Edge(label="de-identified CRF") >> arc
    mon >> Edge(label="source verification") >> srg
    bio >> Edge(label="locked analysis plan", style="dashed") >> arc
```

## The 2.6 FTE, by employment status

| Role | FTE | Status | Paid from | Inside which boundary |
|:--|:--|:--|:--|:--|
| CEO and sponsor lead | 0.8 | Employed | Phase II personnel | Company |
| Verification engineer | 1.0 | Employed | Phase II personnel | Company |
| Clinical research coordinator | 0.5 | Employed | Phase II personnel | Company |
| Biostatistician | 0.2 | Contracted, fixed fee | Phase II consultants | Neither |
| Regulatory consultant | 0.1 | Contracted, fixed fee | Phase II consultants | Neither |
| **Total** | **2.6** | 2.3 employed, 0.3 contracted | | |

At Phase I staffing the figure is 1.1 FTE: the CEO at 0.6 and the verification
engineer at 0.5. The coordinator is not hired until a participant can be
consented, which is month 10.

## The six site functions, none on the company payroll

| Function | Who supplies it | How it is paid | Instrument required |
|:--|:--|:--|:--|
| Operating surgeon and assistant | UC San Diego Moores | Site clinical conduct line, per participant | Executed CTA |
| Theatre and robotic platform time | UC San Diego Moores | Contributed, unpriced | Executed CTA |
| Investigational pharmacy | UC San Diego Moores | Site clinical conduct line | Executed CTA |
| Pathology and specimen handling | UC San Diego Moores | Pharmacy and laboratory line | Executed CTA |
| On-premises advisory model host | UC San Diego Moores | Contributed, unpriced | Executed CTA |
| DSMB | Independent, site convened | Independent monitoring line | Charter, then CTA |

The advisory model runs on the hospital's side of the boundary, not the
company's. That is a deliberate architectural choice and not a convenience: the
model reads telemetry that is part of the medical record, so moving it to the
company's premises would move identified data with it.

## Glyph assignment

| Tile | Glyph macro | Tile style | Cluster |
|:--|:--|:--|:--|
| CEO and sponsor lead | `\glyphuser` | `dgtiled` | Company |
| Verification engineer | `\glyphgear` | `dgtiled` | Company |
| Clinical research coordinator | `\glyphteam` | `dgtile` | Company |
| Two GPU workstation nodes | `\glyphcpu` | `dgtilem` | Company |
| Artifact store and manifest | `\glyphdb` | `dgtile` | Company |
| Biostatistician | `\glyphchart` | `dgtileg` | Contracted |
| Regulatory consultant | `\glyphdoc` | `dgtileg` | Contracted |
| Independent monitor | `\glyphmon` | `dgtileg` | Contracted |
| Operating surgeon | `\glyphscalpel` | `dgtile` | Site |
| Theatre and robotic platform | `\glyphrobot` | `dgtilem` | Site |
| Investigational pharmacy | `\glyphpill` | `dgtile` | Site |
| Pathology and specimens | `\glyphflask` | `dgtile` | Site |
| On-premises advisory model | `\glyphai` | `dgtiled` | Site |
| DSMB, independent | `\glyphshield` | `dgtileg` | Site |

## TikZ construction notes

Canvas 14.6 by 9.8 cm. Three clusters left to right, two trust boundaries drawn
as full-height dashed rules in the corridors between them.

| Element | Style token | Placement |
|:--|:--|:--|
| Company tiles | `\dgnodew`, `\dgnode` | x = 0 and 2.70; y = 0, -2.20, -4.40 (five tiles, last row single) |
| Company cluster | `dgcluster`, `fit` all five tiles and labels | `inner sep=7pt` |
| Trust boundary A | `protoblue`, 1.1 pt, dashed | Vertical at x = 5.15, from y = 0.95 to y = -7.10 |
| Contracted tiles | `\dgnodeg` | x = 6.35; y = 0, -2.20, -4.40 |
| Contracted cluster | `dgcluster2`, `fit` three tiles and labels | `inner sep=7pt` |
| Trust boundary B | `protoblue`, 1.1 pt, dashed | Vertical at x = 8.35 |
| Site tiles | `\dgnode`, `\dgnodew`, `\dgnodeg` | x = 9.55 and 12.25; y = 0, -2.20, -4.40 |
| Site cluster | `dgcluster2`, `fit` all six tiles and labels | `inner sep=7pt` |
| Boundary labels | `\scriptsize\sffamily\bfseries`, `text=protoblue`, rotated 90 | Anchored on each rule at y = -3.55, `fill=protowhite` |
| PHI markers | `\glyphlock` at 0.7 scale plus a `\tiny` caption | One inside the site cluster, one struck with `\pxmark` inside the company cluster |
| FTE totals | `d2cellk`, `minimum width=16mm` | Anchored north east on the company and contracted clusters |
| Crossing edges | `dgedgeb` solid, `dgedged` dashed | Four only, at y = -0.75, -2.95, -5.15 and -6.30 |
| In-figure note | `pnote`, `text width=132mm` | x = 0, y = -8.90 |

Tile pitch is 27 mm horizontally and 22 mm vertically, both at the stage floor.
Every label sits 5.4 mm beneath its tile centre and every cluster `fit` names
both node and label.

Boundary discipline: the two rules are vertical and full height, and the only
ink crossing them is the four labelled edges, each at a distinct y. No tile,
label, or cluster border touches either rule; the nearest cluster edge is 11 mm
from boundary A and 12 mm from boundary B.

## Repository sources

- `funding/pdac-funding-applications/final-apply/sections/sec-06-physical-ai-governance.tex` - the on-premises model placement and the trust-boundary argument
- `funding/pdac-funding-applications/final-apply/sections/sec-09-build-method.tex` - the single-operator working method the Phase I staffing reflects
- `funding/potential-partners/UC-San-Diego/` - the six site functions
- `trial-protocol/` - the DSMB charter and the independent monitoring obligation
- `funding/capitalization-plan/d2/fig-08-two-prices-one-programme.d2.md` - the personnel and consultant lines these FTEs are paid from
