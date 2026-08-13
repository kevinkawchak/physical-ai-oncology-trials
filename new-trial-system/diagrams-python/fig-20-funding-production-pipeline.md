# Figure 20 - The production pipeline and its typed exits

**Type.** diagrams (python)-type, clustered infrastructure. **Section.** §6,
Funding Proposals. **Perspective.** *The machinery that turned one evidence
store into fourteen funding artifacts, and the four typed exits at which an
artifact leaves the pipeline, so a funder can see that the applications are
outputs of a repeatable process rather than fourteen separate efforts.* No other
figure in this paper draws the production machinery; Figure 17 plots when the
artifacts appeared, Figure 18 prices them, and Figure 19 routes their dollars.

**Caption (2 balanced lines, 74 and 72 characters, numbered as printed).**

```
Figure 20. One evidence store, one production pipeline, and the four typed
exits at which a funding artifact leaves it for a named recipient class.
```

## Python diagrams source

```python
from diagrams import Diagram, Cluster, Edge
from diagrams.generic.blank import Blank

with Diagram("Funding artifact production", show=False, direction="LR"):
    with Cluster("Evidence store"):
        ev = [Blank("Protocols and IND"),
              Blank("Simulation record"),
              Blank("Legislative drafts"),
              Blank("Peer review record")]

    with Cluster("Pipeline"):
        pl = [Blank("Master prompt"),
              Blank("Sub-prompt schedule"),
              Blank("Draft, full, final stages"),
              Blank("Defect and format pass")]

    with Cluster("Typed exits"):
        ex = [Blank("Person-based mechanism"),
              Blank("Organization mechanism"),
              Blank("Partnership mechanism"),
              Blank("Small-business mechanism")]

    with Cluster("Deposit"):
        dp = [Blank("Zenodo DOI"),
              Blank("Repository path"),
              Blank("Commit provenance")]

    for n in ev:
        n >> Edge(label="reads") >> pl[0]
    pl[0] >> pl[1] >> pl[2] >> pl[3]
    for n in ex:
        pl[3] >> Edge(label="emits") >> n
    for n in ex:
        n >> Edge(style="dashed") >> dp[0]
```

## TikZ construction table

Absolute coordinates. Canvas 15.2 by 9.4 cm. Four clusters left to right,
because the claim is a pipeline.

| Element | Style token | Placement |
|:--|:--|:--|
| Evidence tiles | `dgnodeg` with `dgtileg`, glyphs `\glyphdoc`, `\glyphcpu`, `\glyphbank`, `\glyphteam` | x = 0.85, y = 1.20, -0.70, -2.60, -4.50; pitch 1.90 cm |
| Pipeline tiles | `dgnodew` with `dgtiled`, glyphs `\glyphai`, `\glyphgear`, `\glyphlink`, `\glyphshield` | x = 5.05, same four y values |
| Typed exit tiles | `dgnode` with `dgtilem`, glyphs `\glyphuser`, `\glyphserver`, `\glyphhand`, `\glyphflask` | x = 9.85, same four y values |
| Deposit tiles | `dgnode` with `dgtile`, glyphs `\glyphlink`, `\glyphdb`, `\glyphlock` | x = 14.10, y = 0.25, -1.65, -3.55 |
| Four cluster frames | `dgcluster2` for evidence, `dgcluster` for pipeline, exits and deposit | `fit` over each column's tiles and labels, `inner sep=7pt` |
| Cluster titles | `dgctitle2` for evidence, `dgctitle` for the other three | Anchored north west, 1.5 mm inset |
| Evidence to pipeline | `dgedge` | Four edges from each evidence tile's east anchor converging on the pipeline's first tile west anchor |
| Pipeline internal chain | `dgedgeb`, 0.9 pt | Three vertical edges at x = 5.05 between successive pipeline tiles |
| Pipeline to exits | `dgedgeb` | Four edges from the pipeline's last tile east anchor to each exit tile's west anchor |
| Exits to deposit | `dgedged` | Four dashed edges from each exit tile's east anchor to the deposit cluster's west boundary waypoint |
| Count badges | `dgctitle` on a `dgtileg` chip | One per exit tile, 6 mm right of the tile, carrying the application count |
| In-figure note | `pnote` | x = 0.35, y = -6.30, `text width=144mm` |

The four columns share one vertical pitch, 1.90 cm, so the pipeline reads as
four aligned ranks. Every cluster holds four tiles except the deposit cluster's
three, which is inside the five-tile limit.

## Typed exit table

| Exit | Mechanism class | Applications routed | Source |
|:--|:--|:--|:--|
| Person-based | Funds an individual investigator | 01 NIH Pioneer Award, 07 HHMI Investigator | `funding/pdac-funding-applications` |
| Organization | Funds an institution or an organization type | 02 ARPA-H, 03 NSF TIP X-Labs, 09 Convergent FRO | same |
| Partnership | Funds a consortium or a cost-shared partnership | 04 DOE Genesis Mission, 06 FNIH AMP, 08 NCI CTEP, 10 UC San Diego Moores | same |
| Small business | Funds a firm | 05 NIH SEED SBIR, and the capitalization plan that rewrites it | `funding/capitalization-plan` |

Nine of the ten applications address mechanisms that fund a person or an
institution. One, application 05, addresses the only mechanism in the set built
for a company, and it was the shortest of the ten; the capitalization plan
exists to rewrite that one exit properly.

## Glyph table

| Tile | Pictogram | Why this glyph |
|:--|:--|:--|
| Protocols and IND | `\glyphdoc` | Regulatory documents |
| Simulation record | `\glyphcpu` | Compute-produced evidence |
| Legislative drafts | `\glyphbank` | An institutional instrument |
| Peer review record | `\glyphteam` | Multiple reviewers |
| Master prompt, sub-prompt schedule | `\glyphai`, `\glyphgear` | The instruction and the mechanism that executes it |
| Draft, full, final stages | `\glyphlink` | A chain in which each stage reads the one before |
| Defect and format pass | `\glyphshield` | A guard before release |
| Person, organization, partnership, small business | `\glyphuser`, `\glyphserver`, `\glyphhand`, `\glyphflask` | A person, an institution, a shared undertaking, and a firm doing bench work |
| Zenodo DOI, repository path, commit provenance | `\glyphlink`, `\glyphdb`, `\glyphlock` | A resolver, a store, and an immutable record |

## Edge routing

Fifteen edges. The four inbound edges converge on one anchor from four
distinct approach angles between 22 and 41 degrees, through the clear corridor
between x = 2.35 and x = 3.55 where no tile sits. The three pipeline chain
edges are vertical at x = 5.05 and pass through the 0.55 cm gap between
consecutive tile-label pairs. The four outbound edges diverge from one anchor
at the mirror of the inbound angles, through the clear corridor between x =
6.55 and x = 8.35. The four dashed deposit edges all terminate on a single
stated waypoint at the vertical center of the deposit cluster's west boundary,
so they converge outside the cluster rather than crossing inside it. No edge
passes through a tile or a label box.

## Repository sources

- `funding/pdac-funding-applications/final-apply/publication/LaTeX Source Files.zip` - the ten applications and the mechanism class of each
- `funding/capitalization-plan/final-capital/publication/LaTeX Source Files.zip` - the small-business exit, the eight-stage build the pipeline column renders, and the thirteen deposited assets in the evidence store
- `funding/RFA-RM-27-001-v2/LaTeX Source Files.zip` - the NIH application that exits through the organization class
- `new-trial-system/abstracts/README.md` - the fourteen artifacts and their deposit dates
