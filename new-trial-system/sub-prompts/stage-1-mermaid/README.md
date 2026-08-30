## Stage 1 sub-prompt - mermaid-type figures

[![Stage](https://img.shields.io/badge/Stage-1%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-1-mermaid)
[![Platform](https://img.shields.io/badge/Platform-Mermaid-A32A3C.svg)](https://mermaid.js.org)
[![Figures](https://img.shields.io/badge/Figures-6-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/mermaid)
[![Output](https://img.shields.io/badge/Output-new--trial--system%2Fmermaid-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/mermaid)

### Instruction

Produce six mermaid-type figure specifications in
[new-trial-system/mermaid](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/mermaid),
one file per figure, one commit per file, committed and pushed the moment each
file is written.

Mermaid is chosen wherever the paper's claim is about **order in time, or a
decision taken at a point in time**. Use its four native constructs and no
others: `flowchart`, `sequenceDiagram`, `stateDiagram-v2`, `gantt`. Six of the
paper's twenty-five figures are mermaid-type because six of its claims are
chronological: the policy chain that created the demand, the shape of one
generation turn, the IND clock, the escalation ladder, the 2026 artifact
calendar, and the two peer-review timelines laid against one another.

| Figure | Section | Construct | Perspective no other figure takes |
|:--|:--|:--|:--|
| 1 | §1 Introduction | flowchart LR | How eleven Federal AI and cancer actions between January 2025 and July 2026 converge on one unmet capability, and which single capability none of them supplies |
| 4 | §2 Methods | sequenceDiagram | What one generation turn looks like end to end: author, master prompt, Claude Code Opus 5, repository, and the author's real-time monitor |
| 7 | §3 IND | gantt | The IND assembly clock in hours against the prior system's same work in months, drawn on one axis |
| 11 | §4 Trial Protocol | flowchart LR | The dose and autonomy escalation ladder from the Phase 1 3+3 to Phase 2 randomization, and the gates between rungs |
| 17 | §6 Funding Proposals | gantt | The 2026 funding artifact calendar: ten applications, two RFA versions, and the capitalization plan, with the deposit date on each |
| 21 | §7 AI Peer Review | sequenceDiagram | The same manuscript through prior-system human review and new-system AI review, drawn as two lanes on one clock |

### Required contents of each file

1. An H1 naming the figure number and its one-line perspective.
2. A **Type**, **Section**, **Perspective** paragraph. The perspective must
   state what no other figure in this paper shows.
3. A caption block of exactly two lines, each within a four-character spread of
   the other, no line shorter than 62 or longer than 78 characters, opening with
   `Figure N. ` exactly as printed in the paper.
4. Valid Mermaid source in a fenced `mermaid` block, using only the palette below.
5. A TikZ construction-notes table: element, style token, placement with an
   explicit pitch or coordinate, so the LaTeX figure is reproducible from the
   specification and does not re-scramble when a node is added.
6. An edge-routing paragraph naming every edge that could cross a node and the
   bend or waypoint that clears it.
7. A repository-sources list naming exact files.

### Palette

Burgundy `#800020`, lighter burgundy 1 `#A32A3C`, lighter burgundy 2 `#E2D6D9`,
Charcoal `#2E2E2E`, Slate Gray `#6B6B6B`, Mist Gray `#C9C9C9`, white `#FFFFFF`.
Charcoal is a stroke and a text color only. **No black fill, and no near-black
fill.** Mist Gray may be lightened to a tint for a pale neutral fill; no eighth
token is introduced.

### Anti-defect requirements

These are the five defects named in the master prompt's ADDRESS DIAGRAM ISSUES
section, restated as rules this stage must satisfy.

- **Edge overlap.** Declare every edge's routing in the construction notes. No
  edge may pass through a node. Where two edges must share a horizontal band,
  one carries an explicit bend and the notes state the clearance in centimeters.
- **Over-density.** Group with `subgraph` rather than flattening. No rank may
  carry more than five nodes, and no figure more than twenty-two nodes.
- **Syntax hallucination.** No PlantUML or Graphviz keyword may appear in a
  mermaid fence. No node label may contain an unescaped quote, bracket,
  ampersand, or percent; write `and` for `&`. Only `classDef` styling is used.
- **Spatial directionality.** `LR` for anything sequential or wide, `TB` only
  where the claim is genuinely a hierarchy. Both gantt figures are horizontal by
  construction; both sequence diagrams read top to bottom by construction.
- **Layout instability.** Every TikZ placement is an absolute coordinate, never
  a relative `below of=`, so adding an element in a later stage moves nothing
  already placed.

### Prohibitions

Do not copy any figure from
[funding/capitalization-plan](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan),
[funding/pdac-funding-applications](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications),
[trial-ind](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind),
or
[trial-protocol](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-protocol).
Figure 7 and Figure 17 are gantt charts, as those works contain gantt charts,
and they must share nothing else with them: different axis, different rows,
different unit, and a different claim. Figure 11 is an escalation ladder, not
the Phase 1 schema reproduced in `trial-protocol/final-protocol`.
