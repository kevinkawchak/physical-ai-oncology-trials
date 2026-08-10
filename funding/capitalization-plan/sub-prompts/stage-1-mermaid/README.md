## Stage 1 sub-prompt - mermaid-type figures

[![Stage](https://img.shields.io/badge/Stage-1%20of%208-00417A.svg)](.)
[![Platform](https://img.shields.io/badge/Platform-Mermaid-3C7DB2.svg)](https://mermaid.js.org)
[![Figures](https://img.shields.io/badge/Figures-5-6C757D.svg)](../../mermaid)
[![Output](https://img.shields.io/badge/Output-..%2Fmermaid-9AA1A8.svg)](../../mermaid)

### Instruction

Produce five mermaid-type figure specifications in
`funding/capitalization-plan/mermaid/`, one file per figure, one commit per
file, committed the moment each file is written.

Mermaid is chosen wherever the paper's claim is about **order in time or a
decision taken at a point in time**. Use its four native constructs and no
others: `flowchart`, `sequenceDiagram`, `stateDiagram-v2`, `gantt`.

| Figure | Section | Construct | Perspective no other figure takes |
|:--|:--|:--|:--|
| 1 | §1 The Novel-Performer Case | flowchart LR | Which clause of the report survives an eligibility filter, and why the two that fail, fail |
| 7 | §3 The $1.6M Gate | stateDiagram-v2 | The Phase I to Phase II award state machine and the four guards on the gate transition |
| 12 | §4 Capital Bridge | sequenceDiagram | Who signs what, in what order, relative to the trial clock, during a financing event |
| 13 | §5 Twelve Milestones | gantt | Thirty-three months, twelve milestones, the evidence artifact date on each |
| 19 | §8 San Diego Traction | flowchart LR | What each of the four July and August 2026 contacts unlocks, and what it does not |

### Required contents of each file

1. An H1 naming the figure number and its one-line perspective.
2. **Type**, **Section**, **Perspective** paragraph. The perspective must state
   what no other figure in the paper shows.
3. A caption block of exactly three lines, each within a four-character spread
   of the others, no line shorter than 58 or longer than 68 characters.
4. Valid Mermaid source, in a ```mermaid fence, using only the palette below.
5. A TikZ construction-notes table: element, style token, placement with an
   explicit pitch or coordinate.
6. A repository-sources list naming exact files.

### Palette

Corporate Blue `#00417A`, lighter `#3C7DB2`, pale `#DCE8F1`, Professional Gray
`#6C757D`, grays `#E9ECEF`, `#CED4DA`, `#9AA1A8`, white `#FFFFFF`. Black is a
stroke and a text colour only. **No black fill.**

### Anti-defect requirements

- Declare direction explicitly. Use `LR` for anything sequential or wide;
  `TB` only where the claim is genuinely a hierarchy.
- Group with `subgraph` rather than flattening. No rank may carry more than
  five nodes.
- No node label may contain an unescaped quote, bracket, ampersand, or percent.
  Write `and` for `&`.
- No PlantUML or Graphviz keyword may appear in a Mermaid fence.
- State the node pitch in the TikZ notes so the LaTeX figure is reproducible
  from the specification and does not re-scramble when a node is added.
- Do not copy any figure from `funding/pdac-funding-applications`. Figure 13 is
  a gantt, as that work's Figure 17 is, and it must share nothing else with it:
  different axis, different rows, different unit, different claim.
