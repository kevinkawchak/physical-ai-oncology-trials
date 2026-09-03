# Figure 15 - The Thirty-Day Pipeline as Layers

**Platform.** D2. **Native construct.** A layered stack with a count on each
layer.

## Perspective no other figure in this day gives

The pipeline brief is a set of tables and a table cannot show shape. A layered
stack can: it shows that the pipeline is wide at the top, where questions are
cheap to ask, and narrow at the bottom, where decisions are expensive to obtain.
A reader sees the funnel without being told about it.

## Native source

```d2
pipeline: {
  layer_1: "Asked and waiting" {
    count: 17
    note: "Every item waits on somebody other than this company"
  }
  layer_2: "Answerable inside 30 days" {
    count: 9
    note: "Federal, capital and institutional items with a stated interval"
  }
  layer_3: "Unblocks something else" {
    count: 4
    note: "FDA center, sponsor-investigator, instrument, site routing"
  }
  layer_4: "Decisions expected" {
    count: 5
    note: "If nothing goes wrong"
  }
  layer_5: "Cannot be chased at all" {
    count: 4
    note: "Classification, foundation cycle, developer window, IRB calendar"
  }
  layer_1 -> layer_2 -> layer_3 -> layer_4
}
```

## TikZ construction

Four stacked layers of decreasing width, centered on a common vertical axis, with
a fifth layer drawn to one side because it is not part of the funnel. Layer pitch
is 1.15 cm.

| Element | Style | Geometry |
|:--|:--|:--|
| Layer 1 | `d2step`, 96 mm wide | `(0,0)` |
| Layer 2 | `d2step`, 76 mm | `(0,-1.15)` |
| Layer 3 | `d2step`, 56 mm | `(0,-2.30)` |
| Layer 4 | `d2key`, 40 mm | `(0,-3.45)` |
| Layer 5, to the side | `d2ghost`, 46 mm | `(4.70,-1.72)` |
| Count badges | `d2cellk`, 12 mm | Right edge of each layer |
| Layer notes | `pnote`, `text width=40mm` | Right of layers 1 to 4 |
| Funnel edges | `d2edgeb` | Three, vertical, between adjacent layers |
| Side note for layer 5 | `pnote` | Below the side layer |

Edge routing: three vertical edges on the common axis, and no edge to or from the
side layer. The absence of an edge is the point: the four unchaseable items are
not a stage of the funnel, they are outside it, and drawing a connector would
suggest they can be moved along.

## Why layer 5 sits to the side

The four items that cannot be chased at all are not a pipeline stage. They arrive
on somebody else's calendar and no action by this company advances them. Drawing
them inside the funnel would invite exactly the behavior the pipeline brief warns
against: a founder with time and anxiety chasing the one thing that must not be
chased.

## The counts, and what each is

| Layer | Count | What it counts |
|:--|:--|:--|
| Asked and waiting | 17 | Every open item across federal, commercial, institutional, state and regional |
| Answerable inside thirty days | 9 | Those with a stated interval short enough to resolve in the window |
| Unblocks something else | 4 | The four diligence questions with no answer today |
| Decisions expected | 5 | One classification, one determination, one instrument, one site routing, and between zero and three federal replies |
| Cannot be chased | 4 | Classification, foundation cycle, developer window, review board calendar |

## Value provenance

| Value in the figure | Source |
|:--|:--|
| All five counts | `../briefs/brief-01-thirty-day-pipeline.md`, its five tables |
| The four unchaseable items | The same file, its own section |
| The five expected decisions | The same file, the closing paragraph |

## Caption, exactly as printed

```
Figure 15. The thirty-day pipeline as four narrowing layers, with the four
items that cannot be chased drawn outside the funnel rather than inside it.
```

Line 1 is 72 characters, line 2 is 74 characters.

## Sources read

- `funding/auto-fund/08Sep26/briefs/brief-01-thirty-day-pipeline.md`
- `funding/capitalization-plan/final-capital/sections/sec-05-twelve-milestones.tex`
- `funding/capitalization-plan/final-capital/capstyle.sty`, for the `d2*` styles
