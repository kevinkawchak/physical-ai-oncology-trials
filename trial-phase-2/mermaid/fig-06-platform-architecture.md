## Figure 6. Eight-arm platform architecture (PancreSpeed II)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    HUB["10 kHz heartbeat coordination bus<br/>64-byte frame, 100 us deadline"]:::goal
    SPEC["56 DOF (8 x 7)<br/>640 channels (80/arm)<br/>0.05 mm RMS at 1,200 mm/s"]:::light
    A12["Arm 1 / Arm 2<br/>hybrid u-w-p dissection<br/>(P1-P8)"]:::mid
    A34["Arm 3 bipolar + retractor<br/>Arm 4 NIR + ICG imaging<br/>(P1-P8)"]:::mid
    A56["Arm 5 anastomosis (P5-P7)<br/>Arm 6 coag (P2, P5-P8)"]:::mid
    A7["Arm 7 suction +<br/>irrigation (P1-P8)"]:::mid
    A8["Arm 8 imaging +<br/>drain (P1, P4, P8)"]:::mid
    HUB --> A12
    HUB --> A34
    HUB --> A56
    A56 --> A7
    A34 --> A8
    HUB -.->|specification| SPEC
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Eight-arm platform architecture for PancreSpeed II. Eight cooperating arms (56 mechanical degrees of freedom as 8 by 7 per-arm, 640 sensor channels at 80 per arm, 0.05 mm root mean square positioning at a 1,200 mm/s peak tip velocity) are coordinated by a 10 kHz heartbeat bus broadcasting a 64-byte frame on a 100 us deadline; each arm's tool assignment and operative-phase coverage (P1 through P8) drive the Schedule of Activities.

**Role in the protocol.** Renders the &sect;6.1.1 device description and Table of per-arm tool assignment; defines the cyber-physical platform under test in Arm A.

**Source files.** `sections/sec-06-intervention.tex` (56 DOF, 640 channels, heartbeat bus, per-arm tools, phase coverage).
