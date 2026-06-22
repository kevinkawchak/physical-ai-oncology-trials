## Figure 16. Multicenter governance and safety oversight

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    SI["Sponsor-Investigator<br/>ChemicalQDevice"]:::goal
    CC["Coordinating Center<br/>(8-site network)"]:::mid
    IRB["Single IRB (sIRB)<br/>45 CFR 46.114"]:::mid
    DSMB["DSMB<br/>group-sequential interim"]:::mid
    PAIS["Physical AI Safety<br/>Review Committee<br/>&le;90-day cadence"]:::mid
    CRO["CRO / clinical monitor<br/>(robotics + AI competent)"]:::light
    ISM["Independent Safety Monitor<br/>(per procedure)"]:::mid
    SITE["Site PIs + teams<br/>surgeon, operator, backup operator"]:::light
    SI --> CC
    CC --> IRB
    CC --> DSMB
    CC --> PAIS
    CC --> CRO
    CRO --> SITE
    DSMB -.->|stop / continue| SITE
    PAIS -.->|USL changes| ISM
    ISM -.-> SITE
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Multicenter governance and safety oversight. The Sponsor-Investigator directs the trial through a Coordinating Center across the eight-site network and answers to a single IRB and an independent DSMB with a group-sequential interim plan, convenes a Physical AI Safety Review Committee on a &le;90-day cadence, designates an Independent Safety Monitor for each procedure, and delegates monitoring to a robotics- and AI-competent CRO over the site Principal Investigators and their teams (surgeon, operator, and backup operator).

**Role in the protocol.** Renders the &sect;10 governance structure and &sect;10.9 safety oversight; defines the three independent oversight tiers.

**Source files.** `sections/sec-10-oversight.tex` (Coordinating Center, sIRB, DSMB, Physical AI Safety Review Committee, ISM, CRO, site PIs).
