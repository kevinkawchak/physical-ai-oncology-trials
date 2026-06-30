## Figure 21. General Investigational Plan: objectives mapped to endpoints

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart LR
    PR["PRIMARY (co-primary)<br/>safety + feasibility"]:::goal
    P1["Device / procedure SAE<br/>incidence through 30 days"]:::step
    P2["MTD / RP2D of daraxonrasib<br/>(3+3, 28-day DLT)"]:::step
    P3["Assigned tasks completed<br/>without unsafe conversion"]:::step
    SE["SECONDARY<br/>oncologic-surgical"]:::accent
    S1["R0 rate; ISGPS B/C fistula"]:::input
    S2["Clavien-Dindo III+;<br/>30 / 90-day mortality"]:::input
    S3["Conversion rate; operative<br/>parameters (vs Dutch 2025)"]:::input
    EX["EXPLORATORY<br/>non-confirmatory"]:::ctx
    E1["PFS / OS to 24 mo (RECIST);<br/>LLM concordance; sim-to-real gap"]:::input
    PR --> P1
    PR --> P2
    PR --> P3
    PR -.-> SE
    SE --> S1
    SE --> S2
    SE --> S3
    SE -.-> EX
    EX --> E1
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef accent fill:#6C757D,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef step fill:#9AA0A6,stroke:#000000,stroke-width:1px,color:#000000
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef ctx fill:#F5F5F5,stroke:#6C757D,stroke-width:1px,color:#000000
```

**Caption.** The objective-to-endpoint hierarchy of the general investigational
plan. Three co-primary objectives (device or procedure serious-adverse-event
incidence through 30 days, the maximum tolerated dose and recommended Phase 2 dose
of daraxonrasib by 3+3 over the 28-day window, and completion of assigned operative
tasks without unsafe conversion) sit above secondary oncologic-surgical quality
endpoints (R0 rate, ISGPS grade B/C fistula, Clavien-Dindo grade III or higher, 30
and 90-day mortality, conversion rate, and operative parameters benchmarked against
the Dutch 2025 robotic cohort) and exploratory, non-confirmatory endpoints
(progression-free and overall survival to 24 months by RECIST, LLM advisory
concordance, and the sim-to-real gap).

**Role in the IND.** Renders in §4.3 (General Approach for Evaluation of Treatment)
and §4.4 (Description of First Year Trials).

**Source files.**
`trial-protocol/final-protocol/publication/sections/sec-03-objectives.tex` (the
objective-to-endpoint hierarchy);
`trial-protocol/final-protocol/publication/sections/sec-08-assessments.tex` (the
endpoint definitions and Dutch 2025 comparators).
