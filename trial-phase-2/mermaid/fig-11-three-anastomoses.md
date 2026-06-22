## Figure 11. Three reconstructive anastomoses and ring-tension bands

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    REM["Specimen removed (end P4)<br/>reconstruction begins"]:::light
    PJ["Pancreaticojejunostomy (PJ)<br/>duct-to-mucosa, P5<br/>0.30 to 0.60 N<br/>dominant fistula determinant"]:::goal
    HJ["Hepaticojejunostomy (HJ)<br/>end-to-side, P6<br/>0.20 to 0.50 N"]:::goal
    GJ["Gastrojejunostomy (GJ)<br/>antecolic, P7<br/>0.40 to 0.80 N"]:::goal
    MON["Ring-tension monitor (Arm 5)<br/>soft-warn outside band"]:::mid
    P8["P8: final imaging,<br/>hemostasis, drain placement"]:::light
    REM --> PJ --> HJ --> GJ --> P8
    MON -.->|guards| PJ
    MON -.->|guards| HJ
    MON -.->|guards| GJ
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Three reconstructive anastomoses and their controller ring-tension bands, constructed in sequence under closed-loop Arm 5 control: the duct-to-mucosa pancreaticojejunostomy in P5 (0.30 to 0.60 N), the end-to-side hepaticojejunostomy in P6 (0.20 to 0.50 N), and the antecolic gastrojejunostomy in P7 (0.40 to 0.80 N), each guarded by the Arm 5 ring-tension monitor issuing a soft warning outside the band, followed by final imaging and drain placement in P8. The pancreaticojejunostomy is the dominant fistula determinant and supplies the fistula-grade input to the drug advisory.

**Role in the protocol.** Renders the &sect;6.1.2 reconstruction sequence and the &sect;8.1 anastomosis-quality assessment; ties device telemetry to the fistula endpoint.

**Source files.** `sections/sec-06-intervention.tex` (PJ/HJ/GJ sequence, ring-tension bands, Arm 5 soft-warn); `sections/sec-08-assessments.tex` (anastomosis-quality assessment).
