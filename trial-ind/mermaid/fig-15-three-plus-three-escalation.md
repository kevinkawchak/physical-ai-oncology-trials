## Figure 15. The 3+3 dose-escalation decision automaton

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    DL1["DL1: 160 mg QD<br/>enroll 3"]:::input
    G1{"DLT in<br/>28-day window?"}:::dec
    EXP1["Expand to 6<br/>at this level"]:::step
    DL2["DL2: 220 mg QD<br/>enroll 3"]:::input
    G2{"DLT?"}:::dec
    EXP2["Expand to 6"]:::step
    DL3["DL3: 300 mg QD<br/>enroll 3"]:::input
    G3{"DLT?"}:::dec
    EXP3["Expand to 6"]:::step
    MTD["MTD = prior level<br/>(>= 2 of 6 DLT)"]:::dark
    RP2D["RP2D at or below MTD<br/>integrating PK + safety"]:::goal
    DL1 --> G1
    G1 -->|"0 of 3"| DL2
    G1 -->|"1 of 3"| EXP1
    EXP1 -->|"1 of 6"| DL2
    EXP1 -->|">= 2 of 6"| MTD
    DL2 --> G2
    G2 -->|"0 of 3"| DL3
    G2 -->|"1 of 3"| EXP2
    EXP2 -->|"1 of 6"| DL3
    EXP2 -->|">= 2 of 6"| MTD
    DL3 --> G3
    G3 -->|"0 of 3 / 1 of 6"| RP2D
    G3 -->|">= 2 of 6"| MTD
    MTD --> RP2D
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef step fill:#9AA0A6,stroke:#000000,stroke-width:1px,color:#000000
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef dec fill:#D9D9D9,stroke:#000000,stroke-width:1px,color:#000000
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.2px,color:#FFFFFF
```

**Caption.** The standard 3+3 dose-escalation automaton across the three
daraxonrasib dose levels (DL1 160 mg, DL2 220 mg, DL3 300 mg once daily). Three
participants enroll at each level; with no dose-limiting toxicity in the 28-day
window the cohort escalates, with one of three it expands to six, and two or more
of six fixes the maximum tolerated dose at the prior level. The recommended Phase 2
dose is selected at or below the maximum tolerated dose, integrating pharmacokinetic
and safety data. The design maximum is three levels times six, up to 18 treated.

**Role in the IND.** Renders in §4.3 (General Approach for Evaluation of Treatment)
and §4.4 (Description of First Year Trials).

**Source files.**
`trial-protocol/final-protocol/publication/sections/sec-04-design.tex` (the 3+3
rule and dose levels, presented here as a new decision automaton);
`trial-protocol/final-protocol/publication/sections/sec-09-statistics.tex` (MTD /
RP2D definitions).
