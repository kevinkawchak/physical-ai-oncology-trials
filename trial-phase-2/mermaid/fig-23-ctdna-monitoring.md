## Figure 23. ctDNA KRAS clearance monitoring timeline

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart LR
    BASE["Baseline (randomization)<br/>KRAS ctDNA drawn<br/>central laboratory assay"]:::light
    SURG["Surgery Day 0<br/>curative-intent resection<br/>(both arms)"]:::mid
    WK12["Week 12 KRAS ctDNA<br/>detectable to undetectable<br/>= clearance (key secondary)"]:::goal
    DET{"Cleared by<br/>week 12?"}:::warn
    YES["Molecular clearance<br/>(favorable)"]:::mid
    NO["Persistent ctDNA<br/>(residual disease signal)"]:::dark
    DYN["Longitudinal ctDNA dynamics<br/>beyond week 12<br/>(exploratory)"]:::dark
    BASE --> SURG --> WK12 --> DET
    DET -->|yes| YES
    DET -->|no| NO
    WK12 -.-> DYN
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Circulating tumor DNA KRAS clearance monitoring timeline. KRAS ctDNA bearing the participant's mutation is measured by a central laboratory at baseline (randomization) in both arms; the conversion from a detectable to an undetectable KRAS ctDNA level by week 12 is the week-12 molecular-clearance key secondary endpoint, tested in the confirmatory hierarchy. The longitudinal dynamics of KRAS ctDNA beyond the week-12 clearance endpoint are an exploratory, non-confirmatory analysis.

**Role in the protocol.** Renders the &sect;8.1 ctDNA assessment, the &sect;9.1 key secondary endpoint, and the &sect;9.4.10 exploratory dynamics.

**Source files.** `sections/sec-08-assessments.tex` (ctDNA baseline + week-12 clearance); `sections/sec-09-statistics.tex` (key secondary clearance, exploratory ctDNA dynamics); `sections/sec-01-summary.tex` (ctDNA draw timing).
