## Stage 5 sub-prompt - graphviz-type figures

[![Stage](https://img.shields.io/badge/Stage-5%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-5-graphviz)
[![Platform](https://img.shields.io/badge/Platform-Graphviz-A32A3C.svg)](https://graphviz.org)
[![Figures](https://img.shields.io/badge/Figures-5-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/graphviz)
[![Output](https://img.shields.io/badge/Output-new--trial--system%2Fgraphviz-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/graphviz)

### Instruction

Produce five graphviz-type figure specifications in
[new-trial-system/graphviz](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/graphviz),
one file per figure, one commit per file, committed and pushed the moment each
file is written.

Graphviz is chosen wherever the paper's claim is about **structure without an
implied clock**: a record with fields, a lineage that branches and merges, a
fault tree where a top event needs a specific combination of basal events, or a
decision tree read from a root. Five of the paper's twenty-five figures are
graphviz-type. Where mermaid would impose a sequence and d2 would impose a
grid, the dot idiom leaves the reader to follow the structure itself.

| Figure | Section | Construct | Perspective no other figure takes |
|:--|:--|:--|:--|
| 5 | §2 Methods | dot record | What one machine-readable figure specification stores, byte for byte, against what a raster figure stores, and what only the first can be re-read as input |
| 9 | §3 IND | fault tree | The specific combination of basal failures that would produce a clinical hold, with the gate type on each junction |
| 15 | §5 Legislation | cluster and record | The five bill versions and four companion documents, with what each version added that its predecessor did not carry |
| 19 | §6 Funding Proposals | record with ports | Where each dollar of the award lands across four work packages, and the two packages the award does not reach |
| 24 | §7 AI Peer Review | decision tree | What happens when the three model reviewers disagree, read from a single root to five terminal dispositions |

### Required contents of each file

1. An H1 naming the figure number and its one-line perspective.
2. A **Type**, **Section**, **Perspective** paragraph stating what no other
   figure in the paper shows.
3. A caption block of exactly two lines within a four-character spread, opening
   with `Figure N. ` exactly as printed.
4. Valid DOT source in a ` ```dot ` fence, using `digraph`, `subgraph cluster_`,
   `shape=record`, `rankdir`, and `label` only. No HTML-like labels, because the
   paper redraws them in TikZ and an HTML label states nothing the record
   syntax does not.
5. A TikZ construction table using the `gv*` vocabulary: `gvnode`, `gvkey`,
   `gvmid`, `gvsoft`, `gvgray`, `gvbox`, `gvcell`, `gvcellh`, `gvcircle`,
   `gvcluster`, `gvctitle`, `gvedge`, `gvedgeb`, `gvedged`, `gvundir`, plus
   `\umlgateand` and `\umlgateor` for the fault tree, with absolute coordinates.
6. A structure table: fields for a record, basal events for a fault tree,
   version deltas for a lineage, ports for a port record, dispositions for a
   decision tree.
7. A repository-sources list naming exact files.

### Palette

Burgundy `#800020` for key nodes with white text, lighter burgundy 1 `#A32A3C`
for mid nodes, lighter burgundy 2 `#E2D6D9` for soft nodes, Mist Gray `#C9C9C9`
and its tints for neutral nodes, Slate Gray `#6B6B6B` for cluster strokes,
Charcoal `#2E2E2E` for node strokes, edges, and text, white for ground.
Graphviz's own default look is thin black strokes on white, and this stage keeps
that character: strokes are Charcoal at 0.55 pt and most nodes are unfilled.
**No black fill.**

### Anti-defect requirements

- **Rank discipline.** Every node is assigned to a stated rank, and the rank
  separation is given in centimeters. No node floats between ranks.
- **Record field alignment.** In a record node, every field is the same height
  and the field separator is a single 0.4 pt rule. A record whose fields are
  sized by their content is a defect.
- **Fault tree correctness.** Every gate has at least two inputs, the gate type
  is drawn as a glyph rather than written as a word, and no basal event feeds
  two gates unless that sharing is the point and is stated.
- **Edge overlap.** A tree drawn top down may not have an edge that crosses a
  sibling subtree. Where an edge must, the notes state the waypoint and the
  clearance.
- **Layout instability.** Absolute coordinates only. Adding a leaf extends its
  parent's fan and moves no other subtree.

### Prohibitions

Do not copy the fault tree from
[funding/capitalization-plan](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan).
Figure 9 is a clinical hold tree whose basal events are regulatory and device
failures, not program failures, and its top event is an agency action rather
than a company one.
