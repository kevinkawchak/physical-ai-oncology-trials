## Figure 15. Adverse-event and Physical AI AE reporting workflow

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    DET["Event detected<br/>clinical AE or Physical AI AE<br/>(malfunction, e-stop, sim divergence)"]:::light
    CLASS{"Classify<br/>severity / causality / expectedness"}:::warn
    NS["Non-serious<br/>record in CRF (CTCAE v5)"]:::mid
    S15["Serious + unexpected<br/>report <=15 calendar days"]:::goal
    S7["Fatal / life-threatening<br/>report <=7 calendar days"]:::goal
    PAI["Physical AI triggers<br/>system-drug interaction; cybersecurity;<br/>model degradation; sim-to-real >2x"]:::mid
    AUD["Preserve hash-chained audit trail<br/>-24 h to +72 h around event"]:::dark
    COMM["Notify FDA, IRB,<br/>DSMB, Safety Review Committee"]:::goal
    DET --> CLASS
    CLASS -->|non-serious| NS
    CLASS -->|serious| S15
    CLASS -->|fatal/LT| S7
    DET --> PAI --> AUD
    S15 --> COMM
    S7 --> COMM
    PAI --> COMM
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
```

**Caption.** Parallel pharmacovigilance and Physical AI safety reporting: clinical
events are graded by CTCAE v5 and reported on the 15-day (serious, unexpected) or
7-day (fatal, life-threatening) IND/IDE timelines, while Physical AI triggers
(system-drug interaction, cybersecurity, model degradation, sim-to-real divergence
beyond twice tolerance) add their own reports, each preserving the hash-chained
audit trail from 24 hours before to 72 hours after the event.

**Role in the protocol.** Operationalizes &sect;8.3 AE/SAE and the &sect;312.32(g)
Physical AI reporting machinery.

**Source files.** `inputs/21cfr312_adapt/03_protocol_amendments_reporting.tex`
(7/15-day timelines, 6 Physical AI triggers, -24h/+72h audit window);
`nih-protocol/06_study_assessments_and_procedures.md` (AE/SAE classification).
