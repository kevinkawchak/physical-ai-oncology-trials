## Figure 7. Five-vessel vascular safety-zone gate

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    TIP["Instrument tip proximity<br/>sampled at 10 kHz"]:::light
    Z{"Zone test<br/>per vessel"}:::warn
    CLR["Clear<br/>(83 of 100 ticks)"]:::light
    SOFT["soft_warning -> LLM advisory<br/>SMV/PV 3.0 mm; HA/CA/SMA 3.0 mm"]:::mid
    NOFLY["no_fly -> block command<br/>SMV/PV 1.0 mm; HA/CA/SMA 1.5 mm"]:::mid
    HARD["hard_stop -> ESTOP<br/>all vessels 5.0 mm, halt <=3 ms"]:::goal
    V["Vessels guarded:<br/>SMV, PV, hepatic artery,<br/>celiac axis, SMA<br/>5 vessels x 8 phases = 40 cells"]:::dark
    TIP --> Z
    Z -->|clear| CLR
    Z -->|soft| SOFT
    Z -->|no_fly| NOFLY
    Z -->|hard| HARD
    Z -.-> V
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
```

**Caption.** Three concentric proximity zones around each of five named vessels
(superior mesenteric vein, portal vein, hepatic artery, celiac axis, superior
mesenteric artery): a soft-warning zone triggers an LLM advisory, a no-fly zone
blocks the command, and a hard-stop zone forces a &le;3 ms E-stop. The gate is
sampled at 10 kHz across all 8 phases (40 vessel-by-phase cells).

**Role in the protocol.** A central &sect;6 / &sect;8 safety control and the
mechanism behind counterfactual Scenario B (vascular injury averted).

**Source files.** `inputs/2030-pdac-1min-final-paper/sections/methods.tex`
(vessel thresholds 1.0/1.5/3.0/5.0 mm, gate verdict counts);
`inputs/21cfr312_adapt/05_clinical_holds_appendices_closing.tex` (&sect;312.404 e-stop).
