## Figure 11. The three anastomoses and ring-tension targets

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart LR
    REM["Specimen removed (P4)<br/>reconstruction begins"]:::light
    PJ["Pancreaticojejunostomy (PJ)<br/>duct-to-mucosa, P5<br/>ring tension 0.30-0.60 N"]:::goal
    HJ["Hepaticojejunostomy (HJ)<br/>end-to-side, P6<br/>ring tension 0.20-0.50 N"]:::goal
    GJ["Gastrojejunostomy (GJ)<br/>antecolic, P7<br/>ring tension 0.40-0.80 N"]:::goal
    MON["Ring-tension monitor (Arm 5)<br/>soft-warn outside band"]:::mid
    REM --> PJ --> HJ --> GJ
    MON -. guards .- PJ
    MON -. guards .- HJ
    MON -. guards .- GJ
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
```

**Caption.** The three reconstructive anastomoses with their controller
ring-tension bands: the duct-to-mucosa pancreaticojejunostomy (0.30-0.60 N), the
end-to-side hepaticojejunostomy (0.20-0.50 N), and the antecolic
gastrojejunostomy (0.40-0.80 N), each guarded by the Arm 5 ring-tension monitor
that issues a soft warning outside the band. The pancreaticojejunostomy is the
dominant determinant of postoperative pancreatic-fistula risk.

**Role in the protocol.** Anchors the &sect;8.1 efficacy/feasibility anastomosis
endpoints and the ISGPS fistula-grading safety assessment.

**Source files.** `inputs/2030-pdac-1min-final-paper/sections/methods.tex`
(PJ/HJ/GJ ring-tension targets, anastomosis controllers, ISGPS grading).
