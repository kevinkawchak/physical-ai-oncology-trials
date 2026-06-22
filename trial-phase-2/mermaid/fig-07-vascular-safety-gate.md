## Figure 7. Five-vessel vascular safety-zone gate

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    TIP["Instrument-tip proximity<br/>sampled at 10 kHz"]:::light
    V["Vessels: SMV, PV, hepatic artery,<br/>celiac axis, SMA<br/>(5 vessels x 8 phases)"]:::dark
    Z{"Zone test<br/>per vessel"}:::warn
    CLR["Clear<br/>continue command"]:::light
    SOFT["soft_warning<br/>LLM advisory<br/>all vessels 3.0 mm"]:::mid
    NOFLY["no_fly: block command<br/>SMV/PV 1.0 mm<br/>HA/CA/SMA 1.5 mm"]:::mid
    HARD["hard_stop: ESTOP<br/>all vessels 5.0 mm<br/>halt &le;3 ms"]:::goal
    TIP --> Z
    V -.-> Z
    Z -->|clear| CLR
    Z -->|soft| SOFT
    Z -->|no_fly| NOFLY
    Z -->|hard| HARD
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Five-vessel vascular safety-zone gate for Arm A. Three concentric proximity zones around each named vessel drive the response: a soft-warning zone (3.0 mm) raises an LLM advisory, a no-fly zone (1.0 mm at the superior mesenteric and portal veins; 1.5 mm at the hepatic artery, celiac axis, and superior mesenteric artery) blocks the command, and a hard-stop zone (5.0 mm) forces a &le;3 ms cross-arm E-stop. The gate is sampled at 10 kHz across all eight phases of the operation, and every verdict is written to the audit trail.

**Role in the protocol.** Renders the &sect;8.2 vascular safety-zone gate; defines the deterministic vessel-exclusion limits backing the significant-risk determination.

**Source files.** `sections/sec-08-assessments.tex` (five-vessel gate, three zones, 1.0/1.5/3.0/5.0 mm thresholds); `sections/sec-06-intervention.tex` (no-fly standoff distances).
