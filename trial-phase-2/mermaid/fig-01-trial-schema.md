## Figure 1. Overall randomized multicenter trial schema

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    SCR["Screening (up to 28 days)<br/>KRAS G12 PDAC, ECOG 0-1<br/>resectable / borderline-resectable"]:::light
    ELIG{"Eligible?<br/>&sect;5.1 inclusion / &sect;5.2 exclusion"}:::warn
    SF["Screen failure<br/>(minimal data set, &sect;5.4)"]:::light
    RAND["Central randomization 1:1<br/>permuted-block, stratified<br/>consent + Physical AI opt-out (&sect;312.60(f))"]:::mid
    ARMA["Arm A (experimental)<br/>daraxonrasib RP2D 300 mg PO daily<br/>LLM-directed robotic Whipple (PancreSpeed II)"]:::goal
    ARMB["Arm B (control)<br/>modified FOLFIRINOX<br/>standard high-volume Whipple"]:::goal
    SITES["8 academic HPB sites<br/>harmonized fleet, single IRB"]:::dark
    FU["Follow-up q8-12wk<br/>BICR RECIST 1.1 + central masked pathology<br/>ctDNA baseline + week 12"]:::mid
    EOS["Endpoints<br/>PFS (primary), OS to 24 months<br/>&ge;140 PFS events, database lock"]:::goal
    SCR --> ELIG
    ELIG -->|no| SF
    ELIG -->|yes| RAND
    RAND --> ARMA
    RAND --> ARMB
    ARMA --> SITES
    ARMB --> SITES
    SITES --> FU --> EOS
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** End-to-end schema for the Phase 2 multicenter randomized study, from 28-day screening through eligibility determination, central 1:1 stratified randomization with the Physical AI consent opt-out, the two parallel arms (Arm A daraxonrasib at the RP2D plus the LLM-directed robotic Whipple; Arm B modified FOLFIRINOX plus standard Whipple), conduct across eight harmonized HPB sites, follow-up with blinded independent central review, and the progression-free-survival and 24-month overall-survival endpoints, including the screen-failure branch.

**Role in the protocol.** Renders as the &sect;1.2 Schema and orients the Synopsis; becomes a TikZ `mermaidfig` in the draft, full, and final stages.

**Source files.** `sections/sec-01-summary.tex` (synopsis, schema, arms, sample size); `sections/sec-04-design.tex` (randomization, eight-site design, BICR).
