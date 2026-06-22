## Figure 8. Heartbeat, watchdog, and E-stop architecture

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    HB["10 kHz heartbeat bus<br/>64-byte frame, 100 us deadline"]:::goal
    WD{"Watchdog window<br/>100 us per arm"}:::warn
    OK["Frame on time<br/>continue command state"]:::light
    MISS["Frame missed or<br/>out-of-parameter"]:::mid
    PARK["Emergency arm park<br/>within 50 us"]:::mid
    EST["Cross-arm ESTOP<br/>halt &le;3 ms,<br/>force clamp 0.0 N"]:::goal
    HW["Hardware E-stop<br/>independent of software<br/>(&le;500 ms)"]:::dark
    HB --> WD
    WD -->|met| OK
    WD -->|missed| MISS
    MISS --> PARK --> EST --> HW
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Heartbeat, watchdog, and E-stop architecture for Arm A. A 10 kHz heartbeat bus with a 100 us per-arm watchdog continues the command state on an on-time frame; a missed or out-of-parameter frame parks the affected arm within 50 us and escalates to a cross-arm E-stop that halts the platform in &le;3 ms at a 0.0 N force clamp, backed by a software-independent hardware E-stop meeting the &le;500 ms requirement.

**Role in the protocol.** Renders the &sect;8.2 halt-chain architecture; defines the layered cyber-physical stop budget that bounds every arm.

**Source files.** `sections/sec-08-assessments.tex` (heartbeat, watchdog, park, cross-arm E-stop, hardware E-stop); `sections/sec-06-intervention.tex` (50 us park, &le;3 ms / &le;500 ms latencies).
