# Figure 14 - One Working Day as a Topology

**Platform.** Diagrams (Python). **Native construct.** Clustered topology with
glyph tiles and one loop.

## Perspective no other figure in this day gives

The cadence brief gives the week as a table of themes and times. It does not show
that every day is the same set of places, and that a letter cannot reach a
window without passing the gate. The topology does: four clusters, in, the gate,
out and the record, and one dashed edge from the record back to the next
morning's read.

## Native source

```python
from diagrams import Cluster, Diagram, Edge
from diagrams.generic.device import Mobile
from diagrams.generic.network import Firewall
from diagrams.generic.storage import Storage
from diagrams.onprem.client import Client, User
from diagrams.onprem.compute import Server
from diagrams.onprem.inmemory import Redis
from diagrams.programming.flowchart import Document, Decision

with Diagram("One working day", direction="LR"):
    with Cluster("In"):
        inbox = Client("Email inbox")
        linkedin = User("LinkedIn threads")
        portals = Mobile("Portals and forms")
    with Cluster("The gate"):
        read = Server("5:00 read, replies first")
        rule = Firewall("Follow-up rule, three open days")
        approve = Decision("5:30 approval, one decision")
    with Cluster("Out"):
        federal = Server("Federal window, 6:00 to 8:00")
        west = Server("West Coast window, 11:00")
        federal2 = Server("Second federal window, 12:30")
    with Cluster("The record"):
        build = Redis("Build and commit, 2:00")
        record = Document("Record, 3:30")
        ledger = Storage("Friday ledger and follow-ups")

    [inbox, linkedin, portals] >> read >> rule >> approve
    approve >> [federal, west, federal2] >> build >> record >> ledger
    build >> Edge(style="dashed", label="read first, next morning") >> read
```

## TikZ construction

| Element | Style | Geometry |
|:--|:--|:--|
| Tiles | `\dgnode` for inputs, `\dgnodew` for the gate and windows, `\dgnodeg` for the record | Columns at x = 0.00, 3.60, 7.20, 10.80; rows at y = 0.00, -2.35, -4.70 |
| Clusters | `dgcluster2` for in and the record; `dgcluster` for the gate and out | Fitted to their tiles and labels |
| Buses | Orthogonal lines at x = 1.80, 5.40 and 9.00 | So that no edge crosses a label |
| The loop | `dgedged` from the build tile up to y = 1.45 and back down into the read tile | Labeled once, above the top segment |
| Legend | `\legkey` twice | y = -6.60 |

## Caption, exactly as set

> Figure 14. One working day as a topology of four places, with every letter passing
> the follow-up rule and the 5:30 approval, and the record read first next morning.

The two lines are 82 and 81 characters.

## Where it is used

`../packet/sections/sec-02-action-register.tex`, after Table 22.
