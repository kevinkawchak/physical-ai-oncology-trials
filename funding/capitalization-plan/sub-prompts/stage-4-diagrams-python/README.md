## Stage 4 sub-prompt - diagrams (python)-type figures

[![Stage](https://img.shields.io/badge/Stage-4%20of%208-00417A.svg)](.)
[![Platform](https://img.shields.io/badge/Platform-diagrams%20(python)-3C7DB2.svg)](https://diagrams.mingrammer.com)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../../diagrams-python)
[![Output](https://img.shields.io/badge/Output-..%2Fdiagrams--python-9AA1A8.svg)](../../diagrams-python)
[![Emits](https://img.shields.io/badge/Emits-specification%2C%20not%20.py-9AA1A8.svg)](../../diagrams-python)

### Instruction

Produce three diagrams (python)-type figure specifications in
`funding/capitalization-plan/diagrams-python/`, one file per figure, one commit
per file, committed the moment each file is written.

This vocabulary is chosen wherever the claim is about **where something runs and
which boundary it crosses**. Three such claims exist in this paper.

| Figure | Section | Construct | Perspective no other figure takes |
|:--|:--|:--|:--|
| 4 | §2 The Entity and the Asset | four clusters, glyph tiles | Owned, licensed, contracted and absent drawn as four physical zones with the empty zone left visibly empty |
| 18 | §7 Operating Plan | three clusters across two trust boundaries | Where the 2.6 FTE, the compute, and the contributed site functions actually sit |
| 20 | §10 Build Method | custody clusters | Which custodian holds which artifact if the programme stops, and what a third party can still reproduce |

### Emission rule

**No `.py` file is written.** The stage emits a machine-readable specification
in Markdown: the node graph, the cluster membership, the glyph assignment, and
the TikZ placement. A `.py` file in this repository is linted by three
`lint-and-format` jobs and would have to satisfy `ruff check` and
`ruff format --check`; a specification carries the same information and cannot
break the build. The rule is inherited from
`funding/pdac-funding-applications/diagrams-python/README.md`.

### Required contents of each file

Identical to stage 1, with the source block given as a `diagrams`-library node
and cluster listing in a ```python fence that is illustrative of the graph and
is never executed, plus an explicit glyph assignment table naming the
`\glyph*` macro each tile carries.

### Anti-defect requirements

- Every tile is a `\dgnode`, `\dgnodew` or `\dgnodeg`, which places the label
  **beneath** the tile at a 5.4 mm offset. A label placed inside a 9 mm tile is
  the single commonest overlap in this vocabulary.
- A cluster fit must name both the tile node and its label node, `(n1)(n1l)`,
  or the dashed cluster border will cut through the label.
- Horizontal tile pitch is at least 26 mm; vertical pitch between tile rows is
  at least 20 mm. State both in the notes.
- Figure 4's fourth cluster is empty by design. Draw the empty cluster, label
  it, and place a single italic note inside it. Do not silently drop it, and do
  not fill it with placeholders.
- Do not copy
  `funding/pdac-funding-applications/diagrams-python/fig-09-on-premises-topology`.
  Figure 18 shares that work's subject only at the level of the word topology:
  it maps employment and contract status, not network segments.
