# Figure 2 - The Eleven Roles by Location and Fit

**Platform.** Diagrams (mingrammer, Python). **Native construct.** Clustered
glyph tiles with the label under each tile.

## Perspective no other figure in this day gives

Table 3 lists the eleven roles as rows. A row cannot show that the roles live in
different places and answer to different institutions. This figure groups them
by where each sits and colors them by who can fill them, so a reader sees at a
glance that six roles, 3.10 of the 3.95 full-time equivalents, are open to an
unsolicited applicant and five are not.

## Native source

```python
from diagrams import Cluster, Diagram
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL
from diagrams.generic.device import Mobile
from diagrams.generic.network import Firewall
from diagrams.onprem.compute import Server
from diagrams.onprem.monitoring import Grafana

with Diagram("Eleven roles by location", show=False, direction="TB"):
    with Cluster("Site operations: open to an applicant, 1.85 FTE"):
        ops = [Server("Director of clinical operations, 0.40"),
               Rack("Lead clinical research coordinator, 1.00"),
               Firewall("Regulatory and quality manager, 0.45")]
    with Cluster("Systems and data: open to an applicant, 1.25 FTE"):
        sys = [Server("Systems engineer, site safety officer, 0.55"),
               Grafana("Model governance lead, 0.40"),
               SQL("Data manager and biostatistician, 0.30")]
    with Cluster("Host institution: California license required, 0.65 FTE"):
        host = [Mobile("Site principal investigator, 0.10"),
                Mobile("Sub-investigator, medical oncology, 0.10"),
                Mobile("Investigational drug pharmacist, 0.20"),
                Mobile("Research nurse and navigator, 0.25")]
    with Cluster("Sponsor, 0.20 FTE"):
        ceo = Server("Chief executive, sponsor representative, 0.20")
```

The icons above stand in for the vector pictograms in the TikZ version; the
package ships no clinical icon set, and a stick figure is never used.

## TikZ construction

| Element | Style | Geometry |
|:--|:--|:--|
| Six open roles | `dgtiled` accent tile, white pictogram (`\dgnodew`) | Upper row, pitch 2.45 cm, at y = 0 |
| Four licensed roles | `dgtileg` gray tile, gray pictogram (`\dgnodeg`) | Lower row, pitch 2.45 cm, at y = -3.05 |
| Sponsor role | `dgtilek` darker gray tile | `(11.55,-3.05)` |
| Two open clusters | `dgcluster`, dashed accent | Fit to tiles and labels |
| Two closed clusters | `dgcluster2`, dashed gray on light gray | Fit to tiles and labels |
| Cluster titles | `dgctitle`, `dgctitle2` | Anchored above each cluster's north west corner |
| Legend | `\legkey` twice | y = -5.05 |

Pictograms: gear, document, shield, robot arm, AI, chart for the open roles;
scalpel, pill, flask, hand and bank for the other five. No node name contains a
decimal point.

## Caption, exactly as set

> Figure 2. The eleven roles grouped by where each sits and who can fill it, with the
> six open roles in the accent color and the five licensed or sponsor roles in gray.

The two lines are 83 and 82 characters.

## Where it is used

`../packet/sections/sec-03-the-roster.tex`, after Table 3.
