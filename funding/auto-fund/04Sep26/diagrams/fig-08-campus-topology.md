# Figure 8 - Where Each Trial Function Would Physically Sit

**Platform.** Diagrams, the `mingrammer/diagrams` idiom. **Native construct.**
Dashed titled clusters of icon tiles, each tile carrying a glyph with its label
set beneath it.

## Perspective no other figure in this day gives

Figure 7 says who is responsible. This one says where the work happens, which is
a different question and the one a site operations lead actually asks first. A
function that is legally the sponsor's but physically on a hospital floor needs
space, badge access, and a network drop, and none of that appears in an
obligation diagram.

## Native source

```python
from diagrams import Diagram, Cluster
from diagrams.generic.compute import Rack
from diagrams.generic.storage import Storage
from diagrams.generic.blank import Blank

with Diagram("Trial functions by location", direction="LR"):
    with Cluster("Clinical campus, candidate site"):
        orsuite = Rack("Robotic procedure suite")
        pharmacy = Storage("Investigational pharmacy")
        clinic = Blank("Consent and follow-up clinic")
        path = Storage("Pathology and specimens")
    with Cluster("Sponsor premises, ChemicalQDevice"):
        model = Rack("On-premises model host")
        verify = Rack("Verification harness")
        regbind = Storage("Regulatory binder")
    with Cluster("Neither, and deliberately so"):
        control = Blank("Robot control network")
        edc = Storage("Electronic data capture")
    model >> verify
    orsuite >> path
```

## TikZ construction

Three dashed clusters on a 4.55 cm horizontal pitch. Tiles sit on a 1.75 cm
vertical pitch, and each is drawn with `\dgnode`, which places the tile, its
pictogram and its label as three related nodes so that a `fit` over the tile and
its label encloses both.

| Element | Style and glyph | Geometry |
|:--|:--|:--|
| Clinical cluster, four tiles | `dgtile` with `\glyphrobot`, `\glyphpill`, `\glyphuser`, `\glyphflask` | `(0,0)`, `(0,-1.75)`, `(0,-3.50)`, `(0,-5.25)` |
| Sponsor cluster, three tiles | `dgtilem` with `\glyphai`, `\glyphgear`, `\glyphdoc` | `(4.55,0)`, `(4.55,-1.75)`, `(4.55,-3.50)` |
| Excluded cluster, two tiles | `dgtileg` with `\glyphnet`, `\glyphdb` | `(9.10,0)`, `(9.10,-1.75)` |
| Cluster frames | `dgcluster`, `dgcluster2` on the third | Braced `fit` over each tile and its label node |
| Cluster titles | `dgctitle`, `dgctitle2` on the third | Anchored north west |
| Edges | `dgedgeb` inside the sponsor cluster, `dgedge` inside the clinical cluster | Two only |
| Prohibition marks | `\pxmark` | Two, one on each excluded tile's inbound path |

Edge routing: the only two edges are within clusters, so no edge crosses a
cluster boundary. The two prohibition marks sit on the paths that would exist if
the model host could reach the robot control network or the data capture system,
and drawing the path and then striking it is more legible than omitting it, which
would leave a reader to infer an absence.

## The third cluster is the point of the figure

The third cluster contains the two systems the model process is architecturally
prevented from reaching. The model process holds no write credential to the
electronic data capture system and no route to the robot control network, and
that is a property of the wiring rather than of a policy document: an auditor
given the network diagram and no cooperation from the operator can verify it.

A figure that showed only what exists would not communicate this. The third
cluster exists to show what is deliberately absent.

## Value provenance

| Value in the figure | Source |
|:--|:--|
| The four clinical functions | `funding/move-in/final-move-in/sections/sec-14-staffing-and-roles.tex` |
| The three sponsor functions | `funding/auto-fund/02Sep26/briefs/brief-02-sbir-phase-i-readiness.md`, milestones 1 and 2 |
| The two excluded systems and the prohibition marks | `funding/auto-fund/02Sep26/briefs/brief-01-approval-delta.md`, the advisory boundary paragraph |
| The cluster title wording | `../README.md`: candidate site, feasibility stage only |

## Caption, exactly as printed

```
Figure 8. Where each trial function would physically sit, and the two systems
the advisory model process is architecturally prevented from ever reaching.
```

Line 1 is 76 characters, line 2 is 75 characters.

## Sources read

- `funding/move-in/final-move-in/sections/sec-14-staffing-and-roles.tex`
- `funding/auto-fund/02Sep26/briefs/brief-01-approval-delta.md`
- `funding/capitalization-plan/final-capital/capstyle.sty`, for the `dg*` styles and the vector glyphs
