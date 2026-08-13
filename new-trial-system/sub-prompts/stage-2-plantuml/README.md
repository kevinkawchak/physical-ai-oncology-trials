## Stage 2 sub-prompt - plantuml-type figures

[![Stage](https://img.shields.io/badge/Stage-2%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-2-plantuml)
[![Platform](https://img.shields.io/badge/Platform-PlantUML-A32A3C.svg)](https://plantuml.com)
[![Figures](https://img.shields.io/badge/Figures-4-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/plantuml)
[![Output](https://img.shields.io/badge/Output-new--trial--system%2Fplantuml-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/plantuml)

### Instruction

Produce four plantuml-type figure specifications in
[new-trial-system/plantuml](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/plantuml),
one file per figure, one commit per file, committed and pushed the moment each
file is written.

PlantUML is chosen wherever the paper's claim needs **formal notation**: named
actors with duties, a state that only changes when a stated guard evaluates
true, or two or more things that genuinely happen at once and must join before
the next step. Four of the paper's twenty-five figures are plantuml-type
because four of its claims cannot be drawn honestly without a fork, a guard, or
an actor boundary.

| Figure | Section | Construct | Perspective no other figure takes |
|:--|:--|:--|:--|
| 3 | §2 Methods | activity with fork and join | One master prompt forking into five diagram stages that run against one specification, then joining into three paper stages |
| 10 | §4 Trial Protocol | state with guards | One participant's state machine with every transition guard written as a testable quantity, including the two that route to a hold |
| 14 | §5 Legislation | use case | The actors a Physical AI oncology trial statute creates duties for, and which duty each actor owns that no other actor can discharge |
| 23 | §7 AI Peer Review | activity with fork and join | Three model manufacturers reviewing one artifact concurrently and joining at a single human approval that no model can bypass |

### Required contents of each file

1. An H1 naming the figure number and its one-line perspective.
2. A **Type**, **Section**, **Perspective** paragraph stating what no other
   figure in the paper shows.
3. A caption block of exactly two lines within a four-character spread, opening
   with `Figure N. ` exactly as printed.
4. Valid PlantUML source in a ` ```plantuml ` fence, using `@startuml` and
   `@enduml`, `skinparam` for color only, and no Mermaid or Graphviz keyword.
5. A TikZ construction table using the `uml*` vocabulary: `umlbox`, `umlctrl`,
   `umlkey`, `umlusecase`, `umlstate`, `umlinit`, `umlfinal`, `umlbar`,
   `umlguard`, `umlactor`, with absolute coordinates.
6. A guard table for the state figure, an actor-duty table for the use case
   figure, and a concurrency table for the two activity figures.
7. A repository-sources list naming exact files.

### Palette

Burgundy `#800020`, lighter burgundy 1 `#A32A3C`, lighter burgundy 2 `#E2D6D9`,
Charcoal `#2E2E2E`, Slate Gray `#6B6B6B`, Mist Gray `#C9C9C9`, white `#FFFFFF`.
Charcoal is a stroke and a text color only. **No black fill.**

### Anti-defect requirements

- **Fork and join discipline.** A fork bar must have at least two outgoing
  branches, and every branch must reach the matching join bar. A branch that
  terminates without joining is a defect, not a shortcut.
- **Guard completeness.** Every transition out of a state carries a guard in
  square brackets, and no two guards out of one state can be true at once. The
  guard table must show that the guards partition the space.
- **Actor boundary.** In the use case figure, no actor connects to a use case
  that a different actor owns. Association lines never cross a use case ellipse.
- **Edge overlap.** Fork and join bars are horizontal, branches are vertical,
  and any branch that must change column does so on a dedicated waypoint row
  stated in the construction notes.
- **Layout instability.** Absolute coordinates only. No `below of=` and no
  automatic layout, so a node added in stage 8 moves nothing placed in stage 7.

### Prohibitions

Do not copy the use case figure from
[funding/capitalization-plan](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan)
or the state machine from
[trial-protocol](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-protocol).
Figure 10 is a participant state machine with guards, which the Phase 1
protocol's schema is not; the schema is a flowchart of the study, this is the
state of one person in it.
