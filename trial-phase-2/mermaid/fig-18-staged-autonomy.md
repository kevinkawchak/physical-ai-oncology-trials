## Figure 18. Phase-graduated staged-autonomy model

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart LR
    ST1["Stage 1<br/>clinician-controlled teleoperation /<br/>supervised; full autonomy<br/>prohibited &sect;312.21(e)"]:::light
    R1{"Independent safety review<br/>+ USL &ge;8.0"}:::warn
    ST2["Stage 2 (this phase)<br/>supervised semiautonomous;<br/>Class II, continuous oversight,<br/>immediate manual override"]:::goal
    R2{"Expanded subtask:<br/>independent review +<br/>USL &ge;8.0"}:::warn
    ST3["Stage 3<br/>higher autonomy<br/>(reserved, prohibited this phase)"]:::dark
    ST1 --> R1 --> ST2 --> R2 --> ST3
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Phase-graduated staged-autonomy model for Phase 2. The operative system runs at Stage 2 supervised semiautonomy (Class II, continuous oversight with immediate manual override). Movement from clinician-controlled Stage 1 to Stage 2, and any expanded supervised-autonomous subtask, requires an independent safety review and a Unified Safety Level of at least 8.0; full autonomy (Stage 3) remains reserved and prohibited at this phase under 21 CFR &sect;312.21(e).

**Role in the protocol.** Renders the &sect;4.3 staged-autonomy model; defines the autonomy ceiling and the USL gate for any escalation.

**Source files.** `sections/sec-04-design.tex` (Stage 1 to Stage 2 to Stage 3, USL &ge;8.0, full autonomy prohibited &sect;312.21(e)); `sections/sec-00-compliance.tex` (Class II, supervised semiautonomy).
