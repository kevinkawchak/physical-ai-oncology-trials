## Figure 5. Eight-arm PancreSpeed platform architecture

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    HUB["10 kHz heartbeat coordination bus<br/>64-byte frame, 100 us deadline"]:::goal
    A1["Arm 1 - hybrid u-w-p<br/>active dissection (P1-P8)"]:::mid
    A2["Arm 2 - hybrid u-w-p<br/>active dissection (P1-P8)"]:::mid
    A3["Arm 3 - bipolar + retractor<br/>vessel control (P1-P8)"]:::mid
    A4["Arm 4 - NIR + ICG probe<br/>imaging (P1-P8)"]:::mid
    A5["Arm 5 - anastomosis probe<br/>PJ/HJ/GJ (P5-P7)"]:::mid
    A6["Arm 6 - bipolar coag + cautery<br/>(P2,P5-P8)"]:::mid
    A7["Arm 7 - suction + irrigation<br/>(P1-P8)"]:::mid
    A8["Arm 8 - imaging + drain<br/>(P1,P4,P8)"]:::mid
    SPEC["56 DOF total (8 x 7)<br/>640 sensor channels (80/arm)<br/>0.05 mm RMS at 1200 mm/s"]:::light
    HUB --- A1 & A2 & A3 & A4
    HUB --- A5 & A6 & A7 & A8
    A1 & A8 -. specs .- SPEC
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.4px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
```

**Caption.** The eight cooperating arms (56 degrees of freedom, 640 sensor
channels, 0.05 mm RMS positioning at 1,200 mm/s) coordinated by a 10 kHz
heartbeat bus, with each arm's tool assignment and phase coverage.

**Role in the protocol.** Describes the investigational device in &sect;6.1.1 and
&sect;6.2; the per-arm tool/phase mapping drives the Schedule of Activities.

**Source files.** `inputs/2030-pdac-1min-final-paper/sections/methods.tex`
(per-arm tool assignment, sensor inventory, DOF, positioning);
`inputs/21cfr312_adapt/02_ind_content_phases.tex` (&sect;312.23(g) hardware specification).
