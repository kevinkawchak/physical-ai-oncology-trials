## Figure 18. Staged autonomy model (phase-graduated)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart LR
    ST1["Stage 1 - Clinician-controlled<br/>teleoperation / supervised<br/>full autonomy prohibited (&sect;312.21(e))"]:::light
    REV1{"Independent safety<br/>review + USL reassessment"}:::warn
    ST2["Stage 2 - Supervised semiautonomous<br/>Class II, continuous oversight,<br/>immediate manual override"]:::mid
    REV2{"Accumulated safety data<br/>+ DSMB concurrence"}:::warn
    ST3["Stage 3 - Higher autonomy<br/>only after review<br/>(reserved, later phase)"]:::goal
    ST1 --> REV1 --> ST2 --> REV2 --> ST3
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** Autonomy increases only through gated review: this Phase 1 protocol
operates at clinician-controlled and supervised semiautonomous levels (Class II,
continuous oversight with immediate manual override; full autonomy prohibited per
21 CFR §312.21(e)), and any move to higher autonomy requires an independent
safety review, a USL reassessment, and DSMB concurrence, reserved for a later
phase.

**Role in the protocol.** Frames the &sect;4.1 design and the &sect;6.3
autonomy-level controls.

**Source files.** `inputs/21cfr312_adapt/02_ind_content_phases.tex`
(phase-graduated autonomy, &sect;312.21(e)); `research/ChatGPT-5.5-Thinking-Extended-19Jun26.md`
(staged-autonomy cohorts).
