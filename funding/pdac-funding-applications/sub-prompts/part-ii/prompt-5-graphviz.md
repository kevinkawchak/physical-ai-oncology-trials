## prompt-5-graphviz

**Stage.** PART II, Stage 5 of 8. **Output.** `funding/pdac-funding-applications/graphviz/`.

### Objective

Specify every **graphviz-type** figure. Graphviz is the right vocabulary when
the subject is a **graph proper**: a dependency order, a fault tree, a record
node with ruled fields, or a cluster subgraph. It is used where the paper must
show what depends on what, or how a failure propagates.

### Allocation

Four figures.

| File | Construct | Perspective (must be unique) |
|:--|:--|:--|
| `fig-06-funding-dependency-dag.gv.md` | directed acyclic graph | Which of the ten awards unblocks which trial activity, and which activities have no single point of failure |
| `fig-10-stop-authority-fault-tree.gv.md` | fault tree with AND / OR gates | Every path to an unsafe state and the gate that has to fail first |
| `fig-16-prior-work-citation-graph.gv.md` | citation graph | The fourteen founding documents and how each one feeds the next |
| `fig-20-verification-record-nodes.gv.md` | record nodes | The verification artifacts, field by field, with the reviewer question each field answers |

### Rules

1. Same palette rule. No black fill. Graphviz-type figures keep thin black
   strokes and serif labels, which is the notation's own idiom.
2. A fault tree must have exactly one top event, gates drawn as gate glyphs, and
   no edge crossing a gate.
3. Record nodes are ruled boxes with field separators, not tables with borders.
4. Each file states the figure number, the balanced caption, the DOT source, the
   TikZ `gv*` tokens, and the repository sources.

### Commits

One commit per figure file, then one for the directory README.
