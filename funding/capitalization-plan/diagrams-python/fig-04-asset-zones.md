# Figure 4 - Owned, licensed, contracted and absent, as four physical zones

**Type.** diagrams (python)-type, four clusters with glyph tiles. **Section.**
§2, The Entity and the Asset. **Perspective.** *The shape of what the company
holds, with the empty zone drawn at full size rather than omitted.* No other
figure shows the asset base as an area; Figure 5 states the same four classes as
typed records, which gives the terms but not the proportion.

**Caption (three balanced lines, 63 to 65 characters).**

```
Four zones and thirteen tiles. The owned zone holds everything the
company made; the contracted zone is drawn at full size and holds
nothing, because an empty zone is the finding this figure reports.
```

## Specification, as a diagrams-library graph

No `.py` file is emitted. The listing below is the node and cluster graph in
`diagrams` form and is illustrative of the structure; it is never executed.

```python
# illustrative only, not executed, no .py file is written to the repository
with Diagram("ChemicalQDevice asset zones, August 2026", direction="LR"):
    with Cluster("Owned outright, no encumbrance"):
        owned = [
            Node("Phase 1 protocol"),      Node("Phase 2 protocol"),
            Node("IND package, drafted"),  Node("QSP and VVUQ suite"),
            Node("Repository, MIT"),       Node("Ten application sets"),
            Node("Four legislative drafts"),
        ]
    with Cluster("Licensed, none exclusive"):
        licensed = [
            Node("Base model weights"),
            Node("Robotic platform, site"),
            Node("Daraxonrasib, developer"),
        ]
    with Cluster("Contracted"):
        contracted = []          # zero nodes, by fact
    with Cluster("Absent, none obtainable alone"):
        absent = [
            Node("Site agreement"), Node("IRB approval"),
            Node("Drug supply"),    Node("Letter of authorization"),
        ]
    owned >> Edge(label="supplies") >> absent
    licensed >> Edge(label="depends on", style="dashed") >> absent
```

## Glyph assignment

Every tile carries one vector pictogram from `capstyle.sty`. None is a raster.

| Tile | Glyph macro | Tile style | Zone |
|:--|:--|:--|:--|
| Phase 1 protocol | `\glyphdoc` | `dgtiled` | Owned |
| Phase 2 protocol | `\glyphdoc` | `dgtile` | Owned |
| IND package, drafted | `\glyphflask` | `dgtiled` | Owned |
| QSP and VVUQ suite | `\glyphcpu` | `dgtilem` | Owned |
| Repository, MIT | `\glyphdb` | `dgtile` | Owned |
| Ten application sets | `\glyphbank` | `dgtile` | Owned |
| Four legislative drafts | `\glyphdoc` | `dgtile` | Owned |
| Base model weights | `\glyphai` | `dgtileg` | Licensed |
| Robotic platform, site | `\glyphrobot` | `dgtileg` | Licensed |
| Daraxonrasib, developer | `\glyphpill` | `dgtileg` | Licensed |
| Site agreement | `\glyphhand` | `dgtilek` | Absent |
| IRB approval | `\glyphshield` | `dgtilek` | Absent |
| Drug supply | `\glyphpill` | `dgtilek` | Absent |
| Letter of authorization | `\glyphlink` | `dgtilek` | Absent |

Fourteen tiles across three occupied zones. `\glyphpill` appears twice, once in
Licensed and once in Absent, and that repetition is the point: the same object
is licensed to somebody else and absent from this company.

## TikZ construction notes

Canvas 14.6 by 9.4 cm. Four clusters, two per band, at a horizontal tile pitch
of 27 mm and a vertical row pitch of 22 mm, both above the stage floor.

| Element | Style token | Placement |
|:--|:--|:--|
| Owned tiles, row 1 | `\dgnodew` with `dgtiled`, `\dgnode` with `dgtile` | y = 0, x = 0, 2.70, 5.40, 8.10 |
| Owned tiles, row 2 | Same | y = -2.20, x = 0, 2.70, 5.40 |
| Owned cluster | `dgcluster`, `fit` all seven tiles and all seven labels | `inner sep=7pt` |
| Owned title | `dgctitle` | Anchored south west, 1.2 mm above the cluster |
| Licensed tiles | `\dgnodeg` with `dgtileg` | y = -4.90, x = 0, 2.70, 5.40 |
| Licensed cluster | `dgcluster2`, `fit` three tiles and three labels | `inner sep=7pt` |
| Contracted cluster | `dgcluster2` with `pagrayl` fill, no `fit` target | Fixed rectangle, x = 8.40 to 11.30, y = -4.30 to -6.20 |
| Contracted note | `umlnote`, `text width=24mm` | Centred inside the empty cluster |
| Absent tiles | `\dgnodeg` with `dgtilek` | y = -7.60, x = 0, 2.70, 5.40, 8.10 |
| Absent cluster | `dgcluster2`, `fit` four tiles and four labels | `inner sep=7pt` |
| Supplies edge | `dgedgeb` | Owned cluster south to absent cluster north, `bend right=14` |
| Depends edge | `dgedged` | Licensed to absent, straight, vertical |
| Zone counts | `d2cellk`, `minimum width=12mm` | Anchored north east on each cluster, carrying 7, 3, 0 and 4 |
| In-figure note | `pnote`, `text width=132mm` | x = 0, y = -8.85 |

Label discipline: every tile is placed with `\dgnode`, `\dgnodew` or `\dgnodeg`,
each of which sets the label 5.4 mm beneath the tile centre, outside the 9 mm
tile. No label sits inside a tile, and every cluster `fit` names both the tile
node and its label node, so no dashed cluster border cuts a label.

The contracted cluster is a fixed rectangle rather than a `fit`, because a `fit`
over zero nodes has no extent. It is drawn at 29 by 19 mm, which is the size a
three-tile zone would occupy, so its emptiness reads as absence at scale rather
than as a small box.

## Repository sources

- `funding/supplementary/Physical AI Oncology Trial Founding Documents.md` - the owned zone
- `funding/supplementary/source-files/Physical-AI-Oncology-Trial-Competition-Proposal.zip` - the January 13, 2026 baseline, on which the contracted zone was also empty
- `trial-protocol/`, `trial-phase-2/`, `trial-ind/` - four of the seven owned tiles
- `funding/pdac-funding-applications/` - the ten application sets tile
- `funding/potential-partners/UC-San-Diego/` - two of the four absent tiles
- `LICENSE` at the repository root - the MIT terms on the repository tile
