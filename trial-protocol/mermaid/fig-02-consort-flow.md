## Figure 2. CONSORT-style participant flow

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    ASSESS["Assessed for eligibility<br/>(n approx. 36 screened)"]:::light
    EXCL["Excluded<br/>not meeting &sect;5.1 / meeting &sect;5.2<br/>declined / unresectable on staging"]:::warn
    ENR["Enrolled and dose-assigned<br/>(n = 18 treated, 3+3 design)"]:::mid
    DL1["DL1 cohort<br/>n = 3 (expand to 6 on 1 DLT)"]:::mid
    DL2["DL2 cohort<br/>n = 3-6"]:::mid
    DL3["DL3 cohort (target dose)<br/>n = 3-6"]:::mid
    TREAT["Received robotic Whipple +<br/>perioperative daraxonrasib"]:::goal
    DISC["Discontinued intervention<br/>(conversion, withdrawal, &sect;7)"]:::warn
    ANA["Analyzed<br/>Safety (all dosed), DLT-evaluable,<br/>per-protocol, mITT"]:::goal
    ASSESS -->|exclude| EXCL
    ASSESS --> ENR
    ENR --> DL1 --> DL2 --> DL3 --> TREAT
    TREAT -.->|protocol-defined| DISC
    TREAT --> ANA
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** Participant flow from screening through the 3+3 dose-escalation
cohorts (DL1-DL3, up to n = 18 treated) to the analysis populations, with the
exclusion and discontinuation branches that feed the per-protocol and safety
sets.

**Role in the protocol.** Supports &sect;5 Study Population and &sect;9.3
Populations for Analyses; the n and cohort sizes anchor &sect;9.2 sample size.

**Source files.** `nih-protocol/03_study_design_and_study_population.md` (CONSORT,
screen-failure set); `nih-protocol/07_statistical_considerations.md`
(analysis populations); `research/ChatGPT-5.5-Thinking-Extended-19Jun26.md`
(device-feasibility cohort sizing).
