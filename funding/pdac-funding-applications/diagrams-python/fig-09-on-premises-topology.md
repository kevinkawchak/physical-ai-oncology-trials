# Figure 9 - What sits inside the hospital boundary, and what never crosses it

**Type.** diagrams (python)-type, clustered infrastructure. **Section.** §6,
Physical AI Governance. **Perspective.** *Deployment across a trust boundary.*
Application 04's Figure 1 gives the same boundary for one recipient; this is the
full topology including the two components no application figure shows, the
identity broker and the offline model registry.

**No Python file is generated.** The specification below is machine-readable and
the figure is reproduced natively in TikZ, because the master prompt forbids
raster output and the repository's lint job must stay green.

**Caption (three balanced lines, 63 to 67 characters each).**

```
Nine components, one trust boundary, and one direction of travel.
Records leave for the public archive; nothing returns, and no path
from the model reaches a motion controller at any point.
```

## diagrams (Python) declaration

```python
# Specification only. Not executed, not committed as a .py file.
with Diagram("On-premises advisory topology", direction="LR"):
    with Cluster("Inside the hospital trust boundary"):
        with Cluster("Operative"):
            platform = Node("Eight-arm robotic platform")
            controller = Node("Motion controller")
            display = Node("Advisory display")
        with Cluster("Advisory"):
            model = Node("On-premises model, pinned commit")
            registry = Node("Offline model registry")
            log = Node("Local audit log")
        with Cluster("Control"):
            identity = Node("Identity broker and access control")
            monitor = Node("Site monitoring endpoint")
    with Cluster("Outside"):
        archive = Node("Public archive, DOI minted")
    model >> display >> controller >> platform  # via a human at the display
    registry >> model
    identity >> model
    model >> log
    log >> monitor
    log >> archive  # the only crossing
```

## TikZ construction notes

| Element | Style token | Placement |
|:--|:--|:--|
| Operative cluster | `dgcluster` | Upper left, tiles at x = 0, 2.4, 4.8, y = 0 |
| Advisory cluster | `dgcluster` | Lower left, same x pitch, y = -2.6 |
| Control cluster | `dgcluster2` | Lower left, y = -5.2, two tiles |
| Outside cluster | `dgcluster2` | Right of the boundary, one tile at x = 9.6 |
| Trust boundary | `protoblack` solid 1pt rule | x = 7.6, full height, rotated label at 90 degrees |
| Human gate | `\pnote` on the display-to-controller edge | The word "human" on a white ground, mid-edge |
| Absent path | `\pxmark` between the model tile and the controller tile | Placed at the midpoint of the path that does not exist |
| Glyphs | `\glyphrobot`, `\glyphcpu`, `\glyphmon`, `\glyphai`, `\glyphdb`, `\glyphlock`, `\glyphsignal`, `\glyphcloud` | One vector pictogram per tile; no raster anywhere |

Cluster titles sit above their cluster, never inside the node field. The three
inside clusters are stacked rather than placed side by side, so the single
boundary rule can run the full height without crossing a cluster.

## Repository sources

- `funding/pdac-funding-applications/applications/app-04-doe-genesis-mission/sections/sec-02-mechanism-fit.tex` - the boundary and the struck inbound path
- `funding/pdac-funding-applications/applications/app-01-nih-pioneer-award/sections/sec-04-operation-governance.tex` - the three boundaries and the pinned commit
- `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed/sections/sec-04-operation-governance.tex` - the logging schema and audit replay
