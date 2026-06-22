## Figure 2. CONSORT randomized participant flow

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    ASSESS["Assessed for eligibility<br/>(n &approx; 245 screened)"]:::light
    EXCL["Excluded<br/>not meeting &sect;5.1 / meeting &sect;5.2<br/>declined / unresectable on staging"]:::warn
    RAND["Randomized 1:1<br/>n = 220 (up to 245 enrolled,<br/>&le;10% non-evaluability)"]:::mid
    ARMA["Allocated to Arm A<br/>n &approx; 110<br/>daraxonrasib RP2D + LLM Whipple"]:::goal
    ARMB["Allocated to Arm B<br/>n &approx; 110<br/>mFOLFIRINOX + standard Whipple"]:::goal
    DISCA["Discontinued intervention<br/>(conversion, withdrawal, &sect;7)"]:::warn
    DISCB["Discontinued intervention<br/>(toxicity, withdrawal, &sect;7)"]:::warn
    POPS["Analysis populations<br/>ITT (all randomized, primary)<br/>mITT, per-protocol, safety"]:::dark
    ANA["Analyzed for PFS / OS<br/>&ge;140 PFS events target<br/>BICR sensitivity analysis"]:::goal
    ASSESS -->|exclude| EXCL
    ASSESS --> RAND
    RAND --> ARMA
    RAND --> ARMB
    ARMA -.->|protocol-defined| DISCA
    ARMB -.->|protocol-defined| DISCB
    ARMA --> POPS
    ARMB --> POPS
    POPS --> ANA
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** CONSORT-style participant flow for the parallel-group randomized design: approximately 245 participants are screened, approximately 220 are randomized 1:1 (up to 245 enrolled to allow up to 10 percent non-evaluability), allocating roughly 110 per arm to Arm A and Arm B, with protocol-defined discontinuation branches feeding into the intention-to-treat, modified intention-to-treat, per-protocol, and safety populations and the event-driven PFS and OS analysis.

**Role in the protocol.** Supports &sect;5 Study Population, &sect;9.2 Sample Size, and &sect;9.3 Populations for Analyses; the screened, randomized, and per-arm counts anchor the design.

**Source files.** `sections/sec-09-statistics.tex` (sample size, populations, evaluability); `sections/sec-04-design.tex` (randomized allocation, eight-site conduct).
