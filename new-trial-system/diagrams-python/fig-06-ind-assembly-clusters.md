# Figure 6 - The IND as an assembled object

**Type.** diagrams (python)-type, clustered infrastructure. **Section.** §3,
IND. **Perspective.** *Which clusters of existing source material fed which IND
module, and the three modules that had no prior-system counterpart to copy and were
therefore composed rather than adapted.* No other figure in this paper shows the
IND's inputs; Figure 7 shows when it was assembled, Figure 8 shows the
regulatory crosswalk it satisfies, and Figure 9 shows what would have to fail
for it to be refused.

**Caption (2 balanced lines, 72 and 71 characters, numbered as printed).**

```
Figure 6. Four source clusters feeding twelve IND modules, and the three
modules that had no prior-system counterpart and were composed instead.
```

## Python diagrams source

```python
from diagrams import Diagram, Cluster, Edge
from diagrams.generic.blank import Blank

with Diagram("IND assembly", show=False, direction="LR"):
    with Cluster("Regulatory source"):
        reg = [Blank("21 CFR 312 adapted"),
               Blank("ICH E6 R3 adapted"),
               Blank("ReGARDD IND template"),
               Blank("FDA 1571 instructions")]

    with Cluster("Clinical source"):
        clin = [Blank("Phase 1 protocol v1.0.0"),
                Blank("Phase 2 protocol v1.1.0"),
                Blank("Dutch 1000-case cohort"),
                Blank("RASolute 302 readout")]

    with Cluster("Simulation source"):
        sim = [Blank("QSP metastatic PDAC"),
               Blank("100000 patient in silico"),
               Blank("Digital twin proposals"),
               Blank("VVUQ test pipeline")]

    with Cluster("Device source"):
        dev = [Blank("Eight-arm robotic stack"),
               Blank("Heartbeat broadcast bus"),
               Blank("Force and no-fly limits")]

    ind = Blank("IND v1.0, twelve modules")

    composed = [Blank("Physical AI Subpart J text"),
                Blank("Autonomy disclosure"),
                Blank("Verification cost ledger")]

    for n in reg:
        n >> Edge(label="adapted") >> ind
    for n in clin:
        n >> Edge(label="carried") >> ind
    for n in sim:
        n >> Edge(label="cited") >> ind
    for n in dev:
        n >> Edge(label="specified") >> ind
    for n in composed:
        ind >> Edge(style="dashed", label="composed") >> n
```

## TikZ construction table

Absolute coordinates. Canvas 15.0 by 9.8 cm. Four clusters in two columns on
the left, one subject node in the center, one cluster of composed modules on
the right.

| Element | Style token | Placement |
|:--|:--|:--|
| Regulatory cluster tiles | `dgnode` with `dgtile`, glyphs `\glyphdoc`, `\glyphdoc`, `\glyphdoc`, `\glyphlock` | x = 0.75 and 2.95, y = 0.55 and -1.35; 2 by 2 block |
| Clinical cluster tiles | `dgnode` with `dgtilem`, glyphs `\glyphscalpel`, `\glyphscalpel`, `\glyphchart`, `\glyphpill` | x = 0.75 and 2.95, y = -3.75 and -5.65 |
| Simulation cluster tiles | `dgnodeg` with `dgtileg`, glyphs `\glyphcpu`, `\glyphdb`, `\glyphai`, `\glyphgear` | x = 5.55 and 7.75, y = 0.55 and -1.35 |
| Device cluster tiles | `dgnode` with `dgtile`, glyphs `\glyphrobot`, `\glyphsignal`, `\glyphstop` | x = 5.55, 7.75 and 6.65, y = -3.75 and -5.65 |
| Four cluster frames | `dgcluster` for regulatory and device, `dgcluster2` for clinical and simulation | `fit` over each block's tiles and labels, `inner sep=7pt` |
| Cluster titles | `dgctitle` for burgundy clusters, `dgctitle2` for gray clusters | Anchored north west, 1.5 mm inset |
| Subject node | `dgnodew` with `dgtiled`, glyph `\glyphdoc`, `line width=1pt` | x = 10.60, y = -2.55, the only tile outside a cluster |
| Subject halo | Burgundy ring, 0.6 pt, radius 7.5 mm | Centered on the subject tile, to distinguish it |
| Composed cluster tiles | `dgnode` with `dgtilem`, glyphs `\glyphshield`, `\glyphuser`, `\glyphbank` | x = 13.85, y = -0.55, -2.55, -4.55; pitch 2.00 cm |
| Composed cluster frame | `dgcluster`, burgundy dashed | `fit` its three tiles and labels |
| Inbound edges | `dgedge` for gray clusters, `dgedgeb` for burgundy clusters | From each cluster's east boundary waypoint to the subject tile's west anchor |
| Outbound edges | `dgedged` | From the subject tile's east anchor to each composed tile's west anchor |
| Edge labels | `dgedge` label, white fill | One per cluster, placed on the cluster's single boundary waypoint edge |
| In-figure note | `pnote` | x = 0, y = -7.35, `text width=142mm` |

Every cluster carries at most four tiles and the composed cluster three, so no
frame exceeds the density limit. All tile labels are two lines or fewer at
23 mm.

## Glyph table

| Tile | Pictogram | Why this glyph |
|:--|:--|:--|
| Adapted 21 CFR 312, ICH E6(R3), ReGARDD template | `\glyphdoc` | These are documents adapted verbatim in structure, not systems |
| FDA 1571 instructions | `\glyphlock` | A form with fixed required fields, which the glyph's closed shackle reads as |
| Phase 1 and Phase 2 protocols | `\glyphscalpel` | Both are operative protocols, not analyses |
| Dutch 1000-case cohort | `\glyphchart` | A published statistical baseline |
| RASolute 302 readout | `\glyphpill` | A drug trial result |
| QSP simulation, in silico trial | `\glyphcpu`, `\glyphdb` | Compute and dataset respectively |
| Digital twin proposals, VVUQ pipeline | `\glyphai`, `\glyphgear` | A model and a mechanism |
| Robotic stack, broadcast bus, stop limits | `\glyphrobot`, `\glyphsignal`, `\glyphstop` | The device, its network, and its safety interlock |
| IND v1.0 | `\glyphdoc`, white on burgundy | The subject is itself a document |
| Physical AI Subpart J, autonomy disclosure, cost ledger | `\glyphshield`, `\glyphuser`, `\glyphbank` | Protection, a person, and money |

## Edge routing

Four inbound edges and three outbound edges, seven in total, which is well
below the density at which a converging fan becomes illegible. Each cluster
contributes exactly one edge, leaving from a single stated waypoint at the
vertical center of its east boundary, so four edges converge on the subject
tile's west anchor at four distinct approach angles between 18 and 34 degrees
from horizontal. None passes through a tile or a label box, because the
corridor between the left cluster columns and the subject node, from x = 8.95
to x = 10.15, contains no tile. The three outbound edges are dashed, leave the
subject's east anchor, and rise or fall to their composed tile within a 3.25 cm
horizontal run; the middle one is horizontal and the outer two are straight
obliques, so none crosses another.

## Repository sources

- `trial-ind/final-ind/publication/LaTeX Source Files.zip` - the twelve modules, their content, and every cited source
- `trial-protocol/final-protocol/publication/LaTeX Source Files.zip` - the Phase 1 protocol carried into the IND as the proposed clinical research
- `trial-phase-2/final-protocol/publication/author/LaTeX Source Files.zip` - the Phase 2 protocol carried as the general investigational plan's forward path
- `national-platform/21cfr312_adapt` and `national-platform/ich_e6r3_adapt` - the adapted regulatory text
- `unification` and `digital-twins` - the simulation and VVUQ sources cited in the IND's pharmacology and toxicology module
