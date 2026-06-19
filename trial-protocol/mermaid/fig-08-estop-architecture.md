## Figure 8. Heartbeat, watchdog, and E-stop safety architecture

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    HB["10 kHz heartbeat bus<br/>64-byte frame, 100 us deadline"]:::goal
    WD{"Watchdog window<br/>100 us per arm"}:::warn
    OK["Frame on time<br/>continue command state"]:::light
    MISS["Frame missed<br/>or out-of-parameter"]:::mid
    PARK["Emergency arm park<br/>within 50 us"]:::mid
    EST["Cross-arm ESTOP<br/>full platform halt <=3 ms<br/>force clamp 0.0 N"]:::goal
    HW["Hardware e-stop<br/>independent of software<br/>(&sect;312.404, <=500 ms)"]:::dark
    HB --> WD
    WD -->|met| OK
    WD -->|missed| MISS --> PARK --> EST
    EST --- HW
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
```

**Caption.** The layered halt chain: a 10 kHz heartbeat bus with a 100 us
per-arm watchdog; a missed frame parks the arm within 50 us and escalates to a
cross-arm E-stop that halts the platform in &le;3 ms at a 0.0 N force clamp,
backed by a software-independent hardware E-stop meeting the &sect;312.404
&le;500 ms requirement.

**Role in the protocol.** Operationalizes the &sect;6.3 / &sect;8.2 device safety
systems and the &sect;312.60 continuous-oversight requirement.

**Source files.** `inputs/2030-pdac-1min-final-paper/sections/methods.tex`
(heartbeat 10 kHz, watchdog 100 us, park 50 us, E-stop 3 ms);
`inputs/21cfr312_adapt/05_clinical_holds_appendices_closing.tex` (&sect;312.404 oversight, e-stop timing).
