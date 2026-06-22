## Figure 15. Adverse-event and Physical AI AE reporting workflow

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    DET["Event detected<br/>clinical AE or Physical AI AE"]:::light
    CLS{"Classify severity /<br/>causality / expectedness"}:::warn
    NS["Non-serious<br/>record in CRF (CTCAE v5)"]:::mid
    S15["Serious + unexpected<br/>report &le;15 days"]:::goal
    S7["Fatal / life-threatening<br/>report &le;7 days"]:::goal
    PAI["6 Physical AI triggers (&sect;312.32(g))<br/>serious PAI AE; system-drug;<br/>cyber; degradation &ge;3 cases / 24 h;<br/>sim-to-real &gt;2x; digital-twin"]:::mid
    AUD["Preserve audit trail<br/>-24 h to +72 h<br/>federated, all 8 sites"]:::dark
    COMM["Notify FDA, IRB, DSMB,<br/>Physical AI Safety Review Committee"]:::goal
    DET --> CLS
    CLS -->|non-serious| NS
    CLS -->|serious| S15
    CLS -->|fatal / life-threat| S7
    DET --> PAI
    PAI --> AUD --> COMM
    S15 --> COMM
    S7 --> COMM
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Adverse-event and Physical AI AE reporting workflow. Clinical events in both arms are graded by CTCAE v5 and reported on the 15-day (serious, unexpected) or 7-day (fatal, life-threatening) IND/IDE timelines, while six Physical AI triggers under &sect;312.32(g) add their own reports (serious Physical AI AE; system-drug interaction; cybersecurity incident; sustained model degradation over three or more procedures or 24 hours; sim-to-real divergence beyond twice tolerance; digital-twin discrepancy), each preserving the hash-chained audit trail from 24 hours before to 72 hours after the event, federated across the eight sites, with notification to the FDA, the IRB, the DSMB, and the Physical AI Safety Review Committee.

**Role in the protocol.** Renders the &sect;8.3 AE/SAE machinery and the &sect;8.3.6 Physical AI reporting triggers.

**Source files.** `sections/sec-08-assessments.tex` (AE/Physical AI AE streams, 7/15-day timelines, six triggers, -24h to +72h audit preservation).
