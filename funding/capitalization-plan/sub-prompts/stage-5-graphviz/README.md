## Stage 5 sub-prompt - graphviz-type figures

[![Stage](https://img.shields.io/badge/Stage-5%20of%208-00417A.svg)](.)
[![Platform](https://img.shields.io/badge/Platform-Graphviz-3C7DB2.svg)](https://graphviz.org)
[![Figures](https://img.shields.io/badge/Figures-4-6C757D.svg)](../../graphviz)
[![Output](https://img.shields.io/badge/Output-..%2Fgraphviz-9AA1A8.svg)](../../graphviz)

### Instruction

Produce four graphviz-type figure specifications in
`funding/capitalization-plan/graphviz/`, one file per figure, one commit per
file, committed the moment each file is written.

Graphviz is chosen wherever the claim is about **dependency or propagation**:
what must exist before what, and how one failure reaches everything downstream.
Four such claims exist in this paper.

| Figure | Section | Construct | Perspective no other figure takes |
|:--|:--|:--|:--|
| 3 | §1 The Novel-Performer Case | record nodes, side by side | The same $1,396,000 of direct work decomposed under a university rate and under this company's, to the dollar |
| 9 | §3 The $1.6M Gate | directed acyclic graph, three clusters | Which work packages can be reached with Phase I money, which need Phase II, and which are unreachable at $1.606M |
| 14 | §5 Twelve Milestones | fault tree with AND and OR gates | What has to fail, and in what combination, for the programme to stop at each of four halt points |
| 17 | §6 The Clinical Evidence | record chain | Published quantity to IND section to protocol section to milestone, with the link type on every edge |

### Required contents of each file

Identical to stage 1, with the source fence marked ```dot and the graph
declared `digraph` or `graph` as appropriate.

### Anti-defect requirements

- Set `rankdir` explicitly and set `ranksep` and `nodesep` numerically. The
  default 0.25 in nodesep is too tight once a record node carries three fields.
- Use `cluster_` prefixes for subgraphs; a subgraph without that prefix is not
  drawn as a box and the intended grouping silently disappears.
- Record labels use `|` and `{}` only. Any literal brace, angle bracket, or
  vertical bar inside a field must be escaped, or the record parser fails and
  the node renders as one undivided box.
- A fault tree is strictly layered. Basic events on one rank, gates on the rank
  above, the top event alone on the top rank. No edge may skip more than one
  rank; if a basic event feeds the top event directly, insert a single-input
  gate rather than drawing the long edge across the other gates.
- Neither gate glyph is filled black. `\umlgateand` uses `pagraym`,
  `\umlgateor` uses `pagrayl`, both inherited from the parent style.
- Figure 3 carries real arithmetic. Both records must sum to a stated total and
  the two totals must differ by the stated premium; if they do not, the figure
  is wrong, not merely ugly.
- Do not copy `funding/pdac-funding-applications/graphviz/fig-10-stop-authority-fault-tree`.
  That tree is about who may stop an operation. Figure 15 is about what stops a
  programme, and its basic events are milestones and money, not people.
