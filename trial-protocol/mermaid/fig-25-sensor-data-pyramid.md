## Figure 25. Sensor-data pyramid and provenance retention

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    L0["L0 raw - 412 MB/iteration<br/>~410M records/run, Zenodo only,<br/>never committed to Git"]:::dark
    L1["L1 - 50 ms aggregates"]:::mid
    L2["L2 - 1 s aggregates"]:::mid
    L3["L3 - phase aggregates (P1-P8)"]:::mid
    L4["L4 - anastomosis events + event log"]:::goal
    COMMIT["Committed budget<br/>980 KB/iteration; 33.4 MB total"]:::goal
    L0 --> L1 --> L2 --> L3 --> L4 --> COMMIT
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
```

**Caption.** The provenance pyramid: raw L0 telemetry (412 MB per iteration,
roughly 410 million records per run) stays on Zenodo and is never committed,
while progressively coarser L1-L4 aggregates (50 ms, 1 s, phase, and anastomosis
events) fit a 980 KB-per-iteration committed budget (33.4 MB total). Raw data is
decompressible to human-readable form on FDA request per &sect;312.57.

**Role in the protocol.** Supports &sect;10.1.9 data handling/record keeping and the
&sect;312.57 telemetry-retention requirement.

**Source files.** `inputs/2030-pdac-1min-final-paper/sections/{results,limitations_future}.tex`
(L0-L4 pyramid, file-size budget, ~410M records);
`inputs/21cfr312_adapt/04_annual_reports_withdrawal.tex` (&sect;312.57 telemetry retention).
