## Figure 12. Verification-before-generation ten-gate assurance

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    GEN["Proposed robot-patient<br/>interaction code / command"]:::light
    GATES["10-gate assurance suite<br/>14 external standards<br/>+ 2 clinical baselines"]:::mid
    UQ["Uncertainty quantification<br/>epistemic + aleatory bounds"]:::mid
    V{"Gate-surface verdict"}:::warn
    ACC["ACCEPT<br/>verified before generation"]:::goal
    BLK["BLOCK<br/>command refused"]:::dark
    ESC["ESCALATE<br/>hand back to human"]:::dark
    GEN --> GATES --> UQ --> V
    V -->|accept| ACC
    V -->|block| BLK
    V -->|escalate| ESC
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Verification before generation. Every proposed robot-patient command passes a ten-gate assurance suite aligned to fourteen external safety and robotics standards plus two clinical baselines, with epistemic and aleatory uncertainty bounds, before the gate surface returns one of ACCEPT (verified before generation), BLOCK (command refused), or ESCALATE (hand back to human). This operationalizes the technical half of the H.R. 9510 VVUQ standard.

**Role in the protocol.** Renders the &sect;11.1 ten-gate pipeline; the deny-by-default verification surface that constrains the LLM action space.

**Source files.** `sections/sec-11-additional.tex` (ten-gate suite, 14 standards + 2 baselines, ACCEPT/BLOCK/ESCALATE, VVUQ).
