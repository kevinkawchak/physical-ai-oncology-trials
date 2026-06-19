## Figure 12. VVUQ ten-gate assurance pipeline

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart LR
    GEN["Proposed robot-patient<br/>interaction code / command"]:::light
    GATES["10-Gate Assurance Suite<br/>14 external standards + 2 clinical baselines<br/>IEC 80601-2-77, ASME V&V 40,<br/>NASA-STD-7009A, IEEE 7009-2024"]:::mid
    UQ["Uncertainty quantification<br/>epistemic + aleatory bounds<br/>coefficient-of-variation"]:::mid
    V{"Gate surface verdict"}:::warn
    ACC["ACCEPT<br/>verified before generation"]:::goal
    BLK["BLOCK (5 of 7 surface)<br/>command refused"]:::dark
    ESC["ESCALATE<br/>hand back to human"]:::dark
    GEN --> GATES --> UQ --> V
    V -->|accept| ACC
    V -->|block| BLK
    V -->|escalate| ESC
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
```

**Caption.** Verification before generation: every proposed robot-patient
command passes a ten-gate assurance suite aligned to 14 external safety and
robotics standards plus 2 clinical baselines, with epistemic and aleatory
uncertainty bounds, before the gate surface returns ACCEPT, BLOCK, or ESCALATE
(hand back to human). Prior author suites cleared 51 of 51 and 172 of 172 tests.

**Role in the protocol.** Establishes the &sect;6 / &sect;10 assurance basis and the
H. R. 9510 VVUQ legislative alignment.

**Source files.** `research/Gemini-3.1-Pro-19Jun26.md` (ten-gate suite, 1 ACCEPT
/ 5 BLOCK / 1 ESCALATE, standards list, test counts); `inputs/auto-bill-02`
(VVUQ financial and legislative framing).
