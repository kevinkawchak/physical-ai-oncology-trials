# Figure 12 - Where Compute, Data and Review Would Sit

**Platform.** Diagrams (Python). **Native construct.** Clustered infrastructure,
three clusters and one nested cluster, with a glyph tile per component.

## Perspective no other figure in this day gives

Table 20 says the resources are adequate and that a NAIRR request is pending. It
cannot show where anything runs. The cluster drawing does: the model and its
verifiers on premises at the site, the robot control network sealed inside the
site with no route from the model, and only code and synthetic cases crossing to
shared resources. Letter 3 points the NAIRR team to this figure.

## Native source

```python
from diagrams import Cluster, Diagram, Edge
from diagrams.generic.database import SQL
from diagrams.generic.compute import Rack
from diagrams.generic.device import Tablet
from diagrams.generic.network import Firewall
from diagrams.generic.storage import Storage
from diagrams.onprem.client import Client
from diagrams.onprem.compute import Server
from diagrams.onprem.vcs import Github
from diagrams.programming.flowchart import Document

with Diagram("Where compute, data and review would sit", direction="LR"):
    with Cluster("Clinical site, on premises: no patient data leaves it"):
        edc = SQL("Data capture; the model cannot write")
        model = Server("Advisory language model, on premises")
        verifiers = Server("Two verifier models, other developers")
        seg = Firewall("Segmentation: no route to control")
        audit = Storage("Audit trail, replayable")
        console = Client("Surgeon's console, advice as text")
        with Cluster("Isolated robot control network"):
            platform = Rack("Eight-arm platform")
            stops = Rack("Stops at 3 ms and 500 ms")
            force = Rack("Force limits, 3 N and 18 N")
    with Cluster("NAIRR Pilot shared resources"):
        compute = Rack("Allocated compute, if granted")
        cases = Storage("Synthetic cases from published values")
        rates = Tablet("Measured disagreement rates")
    with Cluster("Public record"):
        repo = Github("Repository and identifiers")
        report = Document("Report of rates and limits")

    edc >> Edge(label="read", style="dashed") >> model
    model >> Edge(label="output") >> verifiers
    verifiers >> console
    verifiers >> audit
    model - Edge(style="dashed") - seg - Edge(style="dashed") - platform
    verifiers >> Edge(label="code only", style="dashed") >> compute
    cases >> compute >> rates >> report >> repo
```

## TikZ construction

| Element | Style | Geometry |
|:--|:--|:--|
| Site tiles, top row | `\dgnodew` with `dgtiled` or `dgtilem` | x = 0.00, 2.55, 5.10 at y = 0.00 |
| Site tiles, middle row | `\dgnode` with `dgtile` | Same x at y = -2.25 |
| Robot network tiles | `\dgnodew` with `dgtilem`, inside a nested `dgcluster2` | Same x at y = -4.55 |
| Shared and public tiles | `\dgnodeg` with `dgtileg` | x = 9.90 and 12.45 |
| Clusters | `dgcluster` for the site; `dgcluster2` for the other three | Fitted to their tiles; the site cluster is drawn before the nested one so both show |
| Edges | `dgedgeb` for advice, `dgedge` for data flow, `dgedged` for read, code, and the sealed link | Labels in white boxes |
| Legend | `\legkey` twice | y = -6.55 |

## Caption, exactly as set

> Figure 12. Where compute, data and review would sit, with the model and its verifiers
> on premises, the robot network sealed off, and only code and synthetic cases shared.

The two lines are 85 and 84 characters.

## Where it is used

`../packet/sections/sec-05-merit-review.tex`, after Table 20 and the subsection on
where compute, data and review would sit.
