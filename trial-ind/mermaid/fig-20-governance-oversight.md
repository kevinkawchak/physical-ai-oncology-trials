## Figure 20. Study governance and three-tier safety oversight

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    SI["Sponsor-Investigator<br/>ChemicalQDevice (K. Kawchak)<br/>COI disclosed, 21 CFR part 54"]:::goal
    IRB["Reviewing IRB<br/>(human-subjects protection)"]:::proc
    DSMB["DSMB<br/>cohort-boundary + triggered;<br/>halt authority"]:::proc
    ISM["Independent Safety Monitor<br/>per-procedure, real-time;<br/>stop authority"]:::proc
    PAISRC["Physical AI Safety Review<br/>Committee (<= 90-day cadence)"]:::proc
    HALT["Binding halt rules:<br/>device SAE >= 1 of 3 in a cohort<br/>OR 2 device-related deaths"]:::dark
    DEC{"Continue, modify,<br/>pause, or halt"}:::dec
    SI --> IRB
    SI --> DSMB
    SI --> ISM
    SI --> PAISRC
    DSMB --> HALT --> DEC
    ISM --> DEC
    PAISRC --> DEC
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef dec fill:#D9D9D9,stroke:#000000,stroke-width:1px,color:#000000
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.2px,color:#FFFFFF
```

**Caption.** Study governance and the three-tier safety oversight. The
sponsor-investigator (ChemicalQDevice, Kevin Kawchak, with the conflict of
interest disclosed under 21 CFR part 54) operates under a reviewing IRB and three
independent bodies: the Data Safety Monitoring Board (cohort-boundary and triggered
reviews, with halt authority), the Independent Safety Monitor (per-procedure,
real-time, with stop authority), and the Physical AI Safety Review Committee (a
cadence of 90 days or less). The binding halt rules, a device-related serious
adverse event in at least one of three within a cohort or two device-related deaths
at any point, drive the continue, modify, pause, or halt decision.

**Role in the IND.** Renders in §6.3 (Investigator and Facilities Data) and §10.4
(Other Information).

**Source files.**
`trial-protocol/final-protocol/publication/sections/sec-10-oversight.tex` (the
three-tier oversight and halt rules);
`trial-protocol/final-protocol/publication/sections/sec-09-statistics.tex` (the
binding stopping rules).
