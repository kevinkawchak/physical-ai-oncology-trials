## Figure 5. On-premises LLM advisory control loop

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    SENS["640 sensor channels<br/>100 kHz force / 10 kHz other<br/>80 channels per arm"]:::light
    MAP["Map to x,y,z<br/>Cartesian command frame"]:::mid
    LLM["On-premises LLM advisory<br/>second-opinion oracle<br/>hash-pinned to commit + seed"]:::goal
    GATE{"Safety gate<br/>vessel zones + force caps<br/>10 kHz beat, 100 us watchdog"}:::warn
    VEND["Robot vendor kinematic stack<br/>(isolated from LLM)"]:::light
    ACT["8-arm actuators<br/>&le;3 N per arm,<br/>&le;18 N cumulative"]:::goal
    AUD["Hash-chained federated audit<br/>21 CFR part 11, all 8 sites"]:::dark
    SENS --> MAP --> LLM --> GATE
    GATE -->|verdict| VEND --> ACT
    ACT -->|telemetry feedback| SENS
    ACT --> AUD
    GATE -.->|block / escalate| ACT
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** On-premises LLM advisory control loop. Sensor data from the 640-channel stack is mapped to a Cartesian command frame; the LLM issues hash-pinned advisory commands from outside the vendor kinematic stack (second-opinion oracle isolation); and a safety gate enforces the vessel zones, the force caps, the 10 kHz heartbeat, and the 100 us watchdog before any actuator moves. Actuator telemetry returns to the sensor stack, and every step is written to a hash-chained federated audit trail under 21 CFR part 11 across all eight sites.

**Role in the protocol.** Renders the &sect;6.1.1 advisory control loop; establishes the bounded, isolated, auditable LLM design that answers the black-box concern.

**Source files.** `sections/sec-06-intervention.tex` (sensor stack, advisory loop, vendor isolation, audit); `sections/sec-11-additional.tex` (federated audit across sites).
