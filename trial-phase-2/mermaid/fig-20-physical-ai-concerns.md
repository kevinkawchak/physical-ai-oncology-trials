## Figure 20. Nine Physical AI concerns and their mitigations

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart LR
    C1["Physical AI limitations"]:::light
    C2["Patient safety is paramount"]:::light
    C3["Loss of human workers"]:::light
    C4["Single-source software"]:::light
    C5["Proprietary-model dependency"]:::light
    C6["Open-source LLMs not domestic"]:::light
    C7["Overly complex AI workflows"]:::light
    C8["LLMs are black boxes"]:::light
    C9["Financial influence,<br/>inequitable access"]:::dark
    M1["Phase 0 sim validation;<br/>narrow locked device"]:::goal
    M2["10 kHz heartbeat, &le;3 ms<br/>e-stop, force caps"]:::goal
    M3["Human-in-the-loop;<br/>surgeon + backup + ISM"]:::goal
    M4["Cross-framework<br/>(&ge;3 frameworks)"]:::goal
    M5["On-premises LLM,<br/>open weights, isolated"]:::goal
    M6["Domestic deployment;<br/>data residency preserved"]:::goal
    M7["Deny-by-default,<br/>simplified MCP gate"]:::goal
    M8["Hash-chained audit trail;<br/>21 CFR part 11"]:::goal
    M9["Capital firewall, equity fund,<br/>part 54, H.R. 9510, CC BY"]:::goal
    C1 --> M1
    C2 --> M2
    C3 --> M3
    C4 --> M4
    C5 --> M5
    C6 --> M6
    C7 --> M7
    C8 --> M8
    C9 --> M9
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** The mapping of all nine Physical AI concerns to protocol mitigations. The original eight device concerns are paired with validated device behavior over open-endedness, layered hard safety limits, retained human roles, multi-framework non-single-source validation, on-premises open-weight isolation, domestic deployment, a deny-by-default simplified gate surface, and a hash-chained audit trail under 21 CFR part 11; the ninth, financial influence and inequitable access, new to this Phase 2 design, is answered by the capital firewall, the Patient Access and Equity Fund, 21 CFR part 54 disclosure, the H.R. 9510 VVUQ financial standard, and open CC BY deposition.

**Role in the protocol.** Renders the &sect;2.2 Physical AI concerns mapping and the companion table; the ninth concern is unique to Phase 2.

**Source files.** `sections/sec-02-introduction.tex` (nine concerns and mitigations table, financial-influence concern, capital firewall and equity fund).
