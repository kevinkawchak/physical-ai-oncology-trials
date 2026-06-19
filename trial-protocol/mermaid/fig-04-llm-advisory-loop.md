## Figure 4. On-premises LLM advisory control loop (second-opinion oracle isolation)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart LR
    SENS["640 sensor channels<br/>100 kHz force / 10 kHz other"]:::light
    MAP["Sensor to x,y,z mapping<br/>Cartesian command frame"]:::mid
    LLM["On-premises repository LLM<br/>advisory commands, hash-pinned<br/>to a commit + deterministic seed"]:::goal
    GATE["Safety gate<br/>vessel zones + force caps<br/>10 kHz heartbeat, 100 us watchdog"]:::mid
    VEND["Robot vendor kinematic stack<br/>(isolated from LLM)"]:::light
    ACT["8-arm actuators<br/>3 N/arm, 18 N cumulative cap"]:::goal
    AUD["Hash-chained audit trail<br/>21 CFR part 11"]:::dark
    SENS --> MAP --> LLM --> GATE --> VEND --> ACT
    ACT -- telemetry --> SENS
    GATE -- "ESTOP <=3 ms" --> ACT
    LLM -. logged .-> AUD
    GATE -. logged .-> AUD
    ACT -. logged .-> AUD
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
```

**Caption.** The on-premises LLM sits outside the robot vendor kinematic stack
(second-opinion oracle isolation): sensor data is mapped to x,y,z, the LLM issues
hash-pinned advisory commands, and a safety gate enforces vessel zones, force
caps, the 10 kHz heartbeat, and the &le;3 ms E-stop before any actuator moves.
Every step is logged to a 21 CFR part 11 hash-chained audit trail.

**Role in the protocol.** Core of &sect;6 Study Intervention and the patient-safety
argument; this loop is what minimizes single-robot error.

**Source files.** `inputs/2030-pdac-1min-final-paper/sections/{introduction,methods,discussion}.tex`
(on-premises LLM thesis, heartbeat/watchdog/E-stop, force caps);
`inputs/21cfr312_adapt/01_preamble_scope_definitions.tex` (hash-chained audit, deny-by-default).
