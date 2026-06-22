## Figure 13. Analysis populations and group-sequential interim

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    RAND["Randomized 1:1<br/>n = 220 (110 per arm)"]:::light
    subgraph POPS["Analysis populations"]
        ITT["ITT population<br/>all randomized (primary)"]:::mid
        MITT["modified ITT<br/>dosed + &ge;1 post-baseline"]:::mid
        PP["Per-protocol<br/>assigned intervention,<br/>no major deviation"]:::mid
        SAF["Safety population<br/>intervention as received"]:::mid
    end
    INTM{"Interim at &approx;60% events<br/>(&approx;84 of &approx;140):<br/>DSMB review"}:::warn
    BOUND["O'Brien-Fleming<br/>efficacy / futility<br/>boundary crossed"]:::goal
    CONT["Continue to<br/>final analysis"]:::goal
    HALT["Binding device-safety halt<br/>SAE-rate or hard-limit breach<br/>mandatory DSMB review"]:::dark
    RAND --> ITT
    RAND --> MITT
    RAND --> PP
    RAND --> SAF
    ITT --> INTM
    INTM -->|crossed| BOUND
    INTM -->|within| CONT
    SAF -.-> HALT
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** The four analysis populations (intention-to-treat, modified intention-to-treat, per-protocol, safety) and the single group-sequential interim analysis at approximately 60 percent of the targeted progression-free-survival events (approximately 84 of about 140), at which the DSMB evaluates the Lan-DeMets O'Brien-Fleming efficacy and futility boundaries. Binding device-safety halt rules operate in parallel: a device-related serious-adverse-event-rate breach or a hard-limit breach triggers a mandatory DSMB review and possible halt.

**Role in the protocol.** Renders the &sect;9.3 populations and &sect;9.4.6 interim analysis; defines the confirmatory stopping framework and the safety halt overlay.

**Source files.** `sections/sec-09-statistics.tex` (four populations, interim at 60%, O'Brien-Fleming, device-safety halt rules).
