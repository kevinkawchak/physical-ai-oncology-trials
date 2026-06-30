## Figure 5. Composition of the initial IND and IRB package

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    subgraph LEFT[" "]
      direction TB
      P["Clinical protocol<br/>(Phase 1, 3+3, n=18)"]:::input
      IB["Investigator's Brochure"]:::input
      NC["Nonclinical, CMC,<br/>stability"]:::input
    end
    subgraph RIGHT[" "]
      direction TB
      ICF["ICF + recruitment<br/>(Physical AI opt-out)"]:::input
      FORM["Investigator forms<br/>1571 / 1572 / 3674"]:::input
      SAFE["Safety information"]:::input
    end
    PKG["Initial IND + IRB package<br/>(highest schedule value)"]:::goal
    CLK["30-day FDA clock<br/>starts at submission"]:::proc
    ACC["Faster assembly starts<br/>the clock and all sites sooner"]:::accent
    LIM["Cannot shorten 30 days<br/>or create missing<br/>toxicology / stability data"]:::dec
    P --> PKG
    IB --> PKG
    NC --> CLK
    ICF --> PKG
    FORM --> CLK
    SAFE --> CLK
    PKG --> CLK
    CLK --> ACC
    CLK --> LIM
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef accent fill:#6C757D,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef dec fill:#D9D9D9,stroke:#000000,stroke-width:1px,color:#000000
```

**Caption.** Composition of the initial IND and IRB package. The clinical
protocol, Investigator's Brochure, nonclinical / CMC / stability data, informed
consent and recruitment materials, investigator forms (1571, 1572, 3674), and
safety information are assembled into one internally consistent package. Faster,
internally consistent assembly starts the 30-day FDA clock and all sites sooner
but cannot shorten the fixed period or generate missing toxicology or stability
data.

**Role in the IND.** Renders in §6 (Proposed Clinical Research) and the
Introduction, itemizing the package the submission comprises.

**Source files.**
`trial-documents/final-paper/publication/sections/sec-04-results.tex` (Figure 17,
adapted in context to this IND);
`trial-protocol/final-protocol/publication/sections/sec-01-summary.tex` (the
protocol, IB, and consent components).
