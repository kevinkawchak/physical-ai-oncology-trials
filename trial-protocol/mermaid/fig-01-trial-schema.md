## Figure 1. Overall trial schema

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    SCR["Screening (up to 28 days)<br/>KRAS G12 PDAC, ECOG 0-1<br/>resectable / borderline-resectable"]:::light
    ELIG{"Eligible?<br/>&sect;5.1 inclusion / &sect;5.2 exclusion"}:::warn
    SF["Screen failure<br/>(minimal data set, &sect;5.4)"]:::light
    ENR["Enrollment + informed consent<br/>incl. Physical AI opt-out (&sect;312.60(f))"]:::mid
    P0["Phase 0 simulation validation<br/>&ge;1000 sim procedures, &ge;2 frameworks<br/>USL &ge; 7.0 (surgical)"]:::mid
    DL["Daraxonrasib dose-cohort assignment<br/>3+3 escalation, DL1-DL3"]:::mid
    SURG["Robotic Whipple, Class II collaborative<br/>8 arms, continuous human oversight"]:::goal
    PERI["Perioperative daraxonrasib advisory<br/>pause then restart T+7 / T+14 / T+21"]:::goal
    FU["Follow-up<br/>acute, 30-day SAE window, then q12wk"]:::mid
    EOS["End of study<br/>last visit / 24-month OS"]:::goal
    SCR --> ELIG
    ELIG -->|no| SF
    ELIG -->|yes| ENR --> P0 --> DL --> SURG --> PERI --> FU --> EOS
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** End-to-end schema for the Phase 1 combined IND/IDE study, from
28-day screening through enrollment with the Physical AI consent opt-out, Phase 0
simulation validation, daraxonrasib 3+3 dose assignment, the eight-arm robotic
Whipple, the perioperative pause-and-restart advisory, and follow-up to the
24-month overall-survival endpoint.

**Role in the protocol.** Renders as the &sect;1.2 Schema and orients the Synopsis;
becomes a TikZ `mermaidfig` in the draft, full, and final stages.

**Source files.** `inputs/2030-pdac-1min-final-paper/sections/{introduction,methods}.tex`
(clinical subject, daraxonrasib advisory windows); `inputs/21cfr312_adapt/02_ind_content_phases.tex`
(Phase 0, USL &ge; 7.0); `research/ChatGPT-5.5-Thinking-Extended-19Jun26.md` (IDE feasibility framing).
