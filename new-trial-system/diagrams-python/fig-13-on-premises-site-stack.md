# Figure 13 - The on-premises site stack, with its inference boundary

**Type.** diagrams (python)-type, clustered infrastructure. **Section.** §4,
Trial Protocol. **Perspective.** *The four layers of equipment a participant is
operated within, and the one boundary that makes the protocol's on-premises
claim checkable: no inference request crosses it.* No other figure in this paper
draws the site; Figure 10 stays inside the participant's state, Figure 11 draws
the study's escalation, and Figure 12 compares the two protocol documents.

**Caption (2 balanced lines, 73 and 74 characters, numbered as printed).**

```
Figure 13. The four-layer on-premises site stack, and the single boundary
across which no inference request, model weight, or patient record passes.
```

## Python diagrams source

```python
from diagrams import Diagram, Cluster, Edge
from diagrams.generic.blank import Blank

with Diagram("On-premises site stack", show=False, direction="TB"):
    with Cluster("Layer 1, oversight"):
        ov = [Blank("Attending surgeon"), Blank("Study coordinator"), Blank("Independent monitor")]

    with Cluster("Layer 2, control"):
        ct = [
            Blank("On-premises LLM host"),
            Blank("Advisory arbiter"),
            Blank("Heartbeat bus at 10 kHz"),
            Blank("Emergency stop under 3 ms"),
        ]

    with Cluster("Layer 3, device"):
        dv = [
            Blank("Eight-arm robotic platform"),
            Blank("Per-arm force cap 3 N"),
            Blank("Cross-arm cap 18 N"),
            Blank("Vascular no-fly gating"),
        ]

    with Cluster("Layer 4, record"):
        rc = [Blank("Operative log store"), Blank("Pathology and imaging"), Blank("Verification hash registry")]

    outside = Blank("Any external network")

    ov[0] >> Edge(label="oversees") >> ct[1]
    ct[1] >> Edge(label="advises") >> dv[0]
    dv[0] >> Edge(label="writes") >> rc[0]
    rc[2] >> Edge(style="dashed", label="attests") >> ov[2]
    ct[0] >> Edge(style="dotted", color="firebrick", label="no path") >> outside
```

## TikZ construction table

Absolute coordinates. Canvas 14.4 by 11.2 cm. Four stacked clusters, one
external node beyond a drawn boundary.

| Element | Style token | Placement |
|:--|:--|:--|
| Layer 1 tiles | `dgnode` with `dgtile`, glyphs `\glyphuser`, `\glyphteam`, `\glyphmon` | y = 0; x = 1.55, 5.15, 8.75; pitch 3.60 cm |
| Layer 2 tiles | `dgnodew` with `dgtiled`, glyphs `\glyphai`, `\glyphgear`, `\glyphsignal`, `\glyphstop` | y = -2.65; x = 1.55, 4.35, 7.15, 9.95; pitch 2.80 cm |
| Layer 3 tiles | `dgnode` with `dgtilem`, glyphs `\glyphrobot`, `\glyphhand`, `\glyphhand`, `\glyphshield` | y = -5.30; x = 1.55, 4.35, 7.15, 9.95 |
| Layer 4 tiles | `dgnodeg` with `dgtileg`, glyphs `\glyphdb`, `\glyphchart`, `\glyphlock` | y = -7.95; x = 1.55, 5.15, 8.75 |
| Four cluster frames | `dgcluster` for layers 2 and 3, `dgcluster2` for layers 1 and 4 | `fit` over each row's tiles and labels, `inner sep=7pt` |
| Cluster titles | `dgctitle` for layers 2 and 3, `dgctitle2` for layers 1 and 4 | Anchored north west, 1.5 mm inset |
| Building boundary | Burgundy rule, 1.2 pt, long dashes | Vertical at x = 11.85, running the full canvas height |
| Boundary label | `dgctitle`, rotated 90 degrees | On the boundary at y = -4.00, reading `site boundary` |
| External node | `dgnodeg` with `dgtileg`, glyph `\glyphcloud` | x = 13.55, y = -2.65, the only tile right of the boundary |
| Vertical chain edges | `dgedgeb`, 0.9 pt | Layer 1 to 2 at x = 1.55, layer 2 to 3 at x = 1.55, layer 3 to 4 at x = 1.55 |
| Attestation edge | `dgedged` | From layer 4's rightmost tile up to layer 1's rightmost tile at x = 8.75 |
| Prohibited path | Charcoal, 1 pt, dotted, terminated by `\pxmark` | From layer 2's leftmost tile east to x = 11.85, crossed out on the boundary |
| Legend | `legkey`, two swatches | x = 0.35, y = -9.55 |
| In-figure note | `pnote` | x = 0.35, y = -10.25, `text width=136mm` |

The four clusters are separated by a uniform 2.65 cm vertical pitch, which is
1.05 cm more than the tallest tile-plus-label pair, so no cluster frame touches
another. The building boundary is the only long-dashed rule in the paper and
carries a rotated label, so it cannot be mistaken for a cluster edge.

## Glyph table

| Tile | Pictogram | Why this glyph |
|:--|:--|:--|
| Attending surgeon | `\glyphuser` | One named person with authority |
| Study coordinator | `\glyphteam` | A role discharged by more than one person |
| Independent monitor | `\glyphmon` | A watching function, not a doing function |
| On-premises LLM host | `\glyphai` | The model itself |
| Advisory arbiter | `\glyphgear` | A mechanism that combines inputs into one advisory |
| Heartbeat bus | `\glyphsignal` | A periodic broadcast |
| Emergency stop | `\glyphstop` | The interlock |
| Robotic platform | `\glyphrobot` | The device |
| Force caps, per arm and cross arm | `\glyphhand` | A limit on contact |
| No-fly gating | `\glyphshield` | A protected region |
| Operative log store | `\glyphdb` | A record store |
| Pathology and imaging | `\glyphchart` | Measured results |
| Verification hash registry | `\glyphlock` | An immutable attestation |
| Any external network | `\glyphcloud` | The thing the boundary excludes |

## Edge routing

Five edges only. The three chain edges run vertically at a single x of 1.55
through the uniform cluster gutter, so none can cross a tile. The attestation
edge runs vertically at x = 8.75 from layer 4 to layer 1, a 7.95 cm rise
through the right-hand corridor; the corridor is clear because layers 2 and 3
place tiles at x = 7.15 and x = 9.95 and none at x = 8.75, leaving 1.20 cm of
clearance on either side. The prohibited path runs horizontally at y = -2.65
from the LLM host tile to the boundary and is struck through with `\pxmark` at
the crossing point, which is the figure's only crossed edge and its whole
point. No edge passes through a label box, because every label is 23 mm wide
and centered under its tile, while every vertical run is at least 1.20 cm from
the nearest label center.

## Repository sources

- `trial-protocol/final-protocol/publication/LaTeX Source Files.zip` - the on-premises requirement, the 10 kHz heartbeat bus, the 3 ms cross-arm emergency stop budget, the 3 N per-arm and 18 N cross-arm force caps, and the vascular no-fly gating
- `trial-phase-2/final-protocol/publication/author/LaTeX Source Files.zip` - the multicenter site qualification that generalizes this stack to eight centers
- `trial-ind/final-ind/publication/LaTeX Source Files.zip` - the device description filed under the 21 CFR 812 pathway
- `national-platform/new_trial_psl` - the site establishment and documentation package the record layer implements
