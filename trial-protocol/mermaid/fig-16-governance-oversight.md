## Figure 16. Study governance and safety oversight

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    SI["Sponsor-Investigator<br/>ChemicalQDevice"]:::goal
    IRB["Reviewing IRB<br/>(SR device + IND)"]:::mid
    DSMB["Data and Safety<br/>Monitoring Board"]:::mid
    ISM["Independent Safety<br/>Monitor (per procedure)"]:::mid
    PAISRC["Physical AI Safety<br/>Review Committee<br/>(&le;90-day cadence)"]:::mid
    CRO["CRO / clinical monitor<br/>(robotics + AI competent)"]:::light
    SITE["Site team<br/>surgeon, operator, backup operator"]:::light
    SI --> IRB
    SI --> DSMB
    SI --> PAISRC
    SI --> CRO --> SITE
    DSMB -. interim stop / continue .-> SI
    PAISRC -. USL / oversight changes .-> SI
    ISM -. independent monitor .-> SITE
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
```

**Caption.** The oversight architecture: the Sponsor-Investigator answers to the
reviewing IRB and a Data and Safety Monitoring Board, convenes a Physical AI
Safety Review Committee on a &le;90-day cadence, designates an Independent Safety
Monitor for each procedure, and delegates monitoring to a robotics- and
AI-competent CRO over the site team (surgeon, operator, and a qualified backup
operator).

**Role in the protocol.** Structures &sect;10.1.5 governance and &sect;10.1.6 safety
oversight.

**Source files.** `inputs/21cfr312_adapt/{03_protocol_amendments_reporting,04_annual_reports_withdrawal}.tex`
(Safety Review Committee, sponsor/CRO/investigator duties);
`nih-protocol/{08_supporting_documentation_regulatory_oversight,10_references_and_footnotes}.md` (DSMB/ISM definitions).
