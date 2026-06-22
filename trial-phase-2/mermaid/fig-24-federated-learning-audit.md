## Figure 24. Federated learning and hash-chained audit across sites

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    subgraph SITES["8 academic HPB sites (federated network)"]
        S1["Site 1<br/>local telemetry +<br/>identifiable records (local)"]:::light
        S2["Site 2<br/>local telemetry +<br/>identifiable records (local)"]:::light
        SN["Sites 3-8<br/>local telemetry +<br/>identifiable records (local)"]:::light
    end
    FED["Federated learning<br/>model + validation evidence exchanged<br/>raw data stays local"]:::mid
    HARM["Fleet harmonization<br/>every device instance<br/>behaves identically"]:::mid
    LEDGER["Federated hash-chained audit<br/>each entry incorporates prior hash<br/>tamper-evident across fleet"]:::goal
    PART11["21 CFR part 11<br/>validated systems, time-stamped<br/>audit, deny-by-default, e-signatures"]:::dark
    S1 --> FED
    S2 --> FED
    SN --> FED
    FED --> HARM
    FED --> LEDGER
    HARM --> LEDGER
    LEDGER --> PART11
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Federated learning and federated hash-chained audit across the eight sites. The academic HPB centers operate as a federated network: model and validation evidence is exchanged across sites while raw telemetry and identifiable records remain local, fleet harmonization ensures every device instance behaves identically, and the audit trail is hash-chained across the fleet so that each entry cryptographically incorporates the hash of the prior entry and any alteration is detectable. The whole record is maintained under 21 CFR part 11 with validated systems, secure time-stamped audit trails, a deny-by-default authorization model, and electronic signatures bound to the records they authenticate.

**Role in the protocol.** Renders the &sect;11.1 federated network and the &sect;10.6 confidentiality and &sect;10.10 data-handling controls.

**Source files.** `sections/sec-11-additional.tex` (federated network, model exchange, fleet harmonization, hash-chained audit); `sections/sec-10-oversight.tex` (21 CFR part 11, deny-by-default, tamper-evident chain).
