## Figure 22. Capital firewall governance

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    FUND["Funders / PACIF<br/>(patient-aligned<br/>private + philanthropic)"]:::light
    FW{"Capital firewall<br/>no influence"}:::warn
    ADMIN["Independent fund administrator<br/>(disburse-only)"]:::mid
    LEV["Permitted operational levers<br/>8 sites, Phase 0 compute, fleet,<br/>access fund, central review core"]:::mid
    OVR["Independent oversight<br/>sIRB, DSMB, Physical AI<br/>Safety Review Committee"]:::goal
    SCI["Randomization, endpoints,<br/>adjudication, analysis,<br/>data access, publication"]:::dark
    FUND --> FW --> ADMIN
    ADMIN --> LEV
    ADMIN --> OVR
    FW -.->|blocked| SCI
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Capital firewall governance. Patient-aligned funders contribute through the PACIF into an independent fund administrator behind a capital firewall that disburses only to the permitted operational levers (the eight-site network, Phase 0 compute, the robot fleet, the Patient Access and Equity Fund, and the central review and biomarker core) and to independent oversight; the dashed blocked path shows that funders cannot reach randomization, endpoint definition, outcome adjudication, statistical analysis, data access, or publication. Financial interests are disclosed under 21 CFR part 54 and governed by the H.R. 9510 VVUQ financial-data standard.

**Role in the protocol.** Renders the &sect;10.3 Funding, Co-Investment Governance, and the Capital Firewall.

**Source files.** `sections/sec-10-oversight.tex` (PACIF, capital firewall, fund administrator, permitted levers, independent oversight, blocked science); `sections/sec-00-compliance.tex` (part 54, H.R. 9510).
