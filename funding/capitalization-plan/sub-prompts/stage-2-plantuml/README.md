## Stage 2 sub-prompt - plantuml-type figures

[![Stage](https://img.shields.io/badge/Stage-2%20of%208-00417A.svg)](.)
[![Platform](https://img.shields.io/badge/Platform-PlantUML-3C7DB2.svg)](https://plantuml.com)
[![Figures](https://img.shields.io/badge/Figures-3-6C757D.svg)](../../plantuml)
[![Output](https://img.shields.io/badge/Output-..%2Fplantuml-9AA1A8.svg)](../../plantuml)

### Instruction

Produce three plantuml-type figure specifications in
`funding/capitalization-plan/plantuml/`, one file per figure, one commit per
file, committed the moment each file is written.

PlantUML is chosen wherever the claim is about **permission**: who may act, what
guard must hold before an action is allowed, and what proceeds concurrently.
This paper makes exactly three such claims, so the count is three.

| Figure | Section | Construct | Perspective no other figure takes |
|:--|:--|:--|:--|
| 6 | §2 The Entity and the Asset | use case with two system boundaries | What ChemicalQDevice may do alone, what only the site may do, and the three cases neither may do yet |
| 10 | §4 Capital Bridge | state with guards | The 21 CFR part 54 capital firewall as five states with the guard on every transition |
| 15 | §5 Twelve Milestones | activity with fork and join | Evidence production and program-officer audit running concurrently against one milestone |

### Required contents of each file

Identical to stage 1, with the source fence marked ```plantuml and delimited by
`@startuml` and `@enduml`.

### Anti-defect requirements

- A guard is written `[condition]` on the transition, never inside the state
  label. A state label that carries its own guard is the commonest way these
  diagrams become unreadable.
- Every `fork` has a matching `fork again` and `end fork`. Every `if` has an
  `endif`. An unclosed block is a parse failure, not a layout problem.
- `skinparam` may set colour and font only. Do not emit CSS, hex gradients, or
  a font the renderer will not have.
- In Figure 11 the five states must be laid out on one horizontal spine so no
  transition arrow crosses a state box. Guards sit above their transition, at a
  stated vertical offset, never on the line itself.
- Figure 6's two boundaries must not overlap. Reserve at least 8 mm of clear
  space between the sponsor boundary and the site boundary, and place any use
  case shared by both in the corridor between them rather than inside either.
- Do not copy `funding/pdac-funding-applications/plantuml/fig-05-actor-authority`
  or `fig-13-advisory-state-guards`. Figure 6 is about corporate and contractual
  authority, not clinical authority. Figure 11 is about money, not advice.
