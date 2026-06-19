## Figure 10. Daraxonrasib 3+3 dose-escalation scheme

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    DL1["DL1 starting dose<br/>enroll 3"]:::light
    Q1{"DLTs in first 3?<br/>(28-day window)"}:::warn
    DL1E["Expand DL1 to 6"]:::mid
    Q1B{"DLTs in 6?"}:::warn
    DL2["DL2 escalate<br/>enroll 3"]:::mid
    MTD["Declare MTD / RP2D<br/>at prior dose level"]:::goal
    DL3["DL3 target dose<br/>enroll 3-6"]:::goal
    DL1 --> Q1
    Q1 -->|0 of 3| DL2
    Q1 -->|1 of 3| DL1E --> Q1B
    Q1 -->|>=2 of 3| MTD
    Q1B -->|1 of 6| DL2
    Q1B -->|>=2 of 6| MTD
    DL2 --> DL3
    DL3 -->|tolerated| MTD
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** The standard 3+3 escalation for the daraxonrasib drug arm: cohorts
of three advance when no dose-limiting toxicity (DLT) is seen in the 28-day
window, expand to six on a single DLT, and define the maximum tolerated dose
(MTD) and recommended Phase 2 dose (RP2D) at the highest tolerated level.

**Role in the protocol.** Realizes the &sect;6.1.2 dose-escalation/de-escalation
scheme and the &sect;3 primary dose-finding objective.

**Source files.** `nih-protocol/04_study_intervention.md` (dose escalation in
exact doses, DLT rules); `inputs/2030-pdac-1min-final-paper/sections/introduction.tex`
(daraxonrasib eligibility, RASolute 302 anchor).
