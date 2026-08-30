## Stage 4 sub-prompt - diagrams (python)-type figures

[![Stage](https://img.shields.io/badge/Stage-4%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-4-diagrams-python)
[![Platform](https://img.shields.io/badge/Platform-diagrams%20(python)-A32A3C.svg)](https://diagrams.mingrammer.com)
[![Figures](https://img.shields.io/badge/Figures-4-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/diagrams-python)
[![Output](https://img.shields.io/badge/Output-new--trial--system%2Fdiagrams--python-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/diagrams-python)

### Instruction

Produce four diagrams (python)-type figure specifications in
[new-trial-system/diagrams-python](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/diagrams-python),
one file per figure, one commit per file, committed and pushed the moment each
file is written.

The `mingrammer/diagrams` idiom is chosen wherever the paper's claim is about
**a system's physical or logical parts, grouped into clusters, where each part
is best named by what kind of thing it is**. Its native rendering, an icon glyph
with the label set beneath it inside a dashed titled cluster, carries kind and
grouping at once, which neither a grid nor a flowchart does. Four of the paper's
twenty-five figures are of this type.

The Python source in each specification is real `diagrams` code and would run,
but it is never executed here: raster output is forbidden by the master prompt,
so the paper redraws each figure in TikZ using the `dg*` vocabulary, where every
glyph is a vector pictogram.

| Figure | Section | Perspective no other figure takes |
|:--|:--|:--|
| 6 | §3 IND | The IND as an assembled object: which clusters of source material fed which module, and the three modules that had no prior-system counterpart to copy |
| 13 | §4 Trial Protocol | The on-premises site stack a participant is operated within: compute, robot, data, and oversight, with the boundary that keeps inference inside the building |
| 20 | §6 Funding Proposals | The production pipeline that turned one evidence store into fourteen funding artifacts, and where each artifact type leaves the pipeline |
| 25 | §8 Limitations | The single-vendor dependency and the watermark provenance chain, drawn with the mitigation attached to each exposure |

### Required contents of each file

1. An H1 naming the figure number and its one-line perspective.
2. A **Type**, **Section**, **Perspective** paragraph stating what no other
   figure in the paper shows.
3. A caption block of exactly two lines within a four-character spread, opening
   with `Figure N. ` exactly as printed.
4. Valid Python `diagrams` source in a fenced `python` block, using `Diagram`,
   `Cluster`, `Node`, and edge operators only.
5. A TikZ construction table using the `dg*` vocabulary: `dgnode`, `dgnodew`,
   `dgnodeg`, `dgtile`, `dgtiled`, `dgtilem`, `dgtileg`, `dgcluster`,
   `dgcluster2`, `dgctitle`, `dgedge`, `dgedgeb`, `dgedged`, `dgbi`, with
   absolute coordinates and a stated cluster inner separation.
6. A glyph table naming the vector pictogram each tile carries and why that
   pictogram and not another.
7. A repository-sources list naming exact files.

### Palette

Burgundy `#800020` for filled tiles with a white pictogram, lighter burgundy 1
`#A32A3C` for mid tiles, lighter burgundy 2 `#E2D6D9` for pale tiles with a
burgundy pictogram, Mist Gray `#C9C9C9` and its tints for neutral tiles, Slate
Gray `#6B6B6B` for cluster strokes and neutral edges, Charcoal `#2E2E2E` for
text and rules, white for ground. **No black fill.**

### Anti-defect requirements

- **Cluster discipline.** Every leaf tile belongs to exactly one cluster. A tile
  outside every cluster is a defect unless it is the figure's single subject
  node, which must then be visually distinguished.
- **Label geometry.** Every tile's label sits 5.4 mm beneath the tile center,
  is at most 23 mm wide, and is at most two lines. A three-line label means the
  node is really two nodes.
- **Over-density.** No cluster may hold more than five tiles, and no figure more
  than four clusters plus one subject node.
- **Edge overlap.** An edge that leaves a cluster does so through a stated
  waypoint on the cluster boundary. An edge may never pass through a tile or its
  label box.
- **Layout instability.** Absolute coordinates only, and cluster frames are
  `fit` over named tiles rather than drawn by hand, so a tile added later grows
  its cluster and moves nothing outside it.

### Prohibitions

Do not copy the infrastructure figures from
[funding/capitalization-plan](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan)
or
[funding/pdac-funding-applications](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications).
Figure 13 is a site stack with an inference boundary, which neither of those
works draws; Figure 20 is a production pipeline with typed exits, not a team
chart.
