## Figure 22. Statistical analysis populations and interim stopping logic

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    ENR["Enrolled and dose-assigned (n = 18)"]:::light
    SAF["Safety population<br/>all who received any intervention"]:::mid
    DLT["DLT-evaluable population<br/>completed 28-day window or had a DLT"]:::mid
    PP["Per-protocol population<br/>received Whipple + advisory per protocol"]:::mid
    MITT["modified ITT<br/>all dosed with >=1 post-baseline assessment"]:::mid
    STOP{"Interim safety review<br/>after each cohort + DSMB"}:::warn
    HALT["Halt rule<br/>>=33% device-related serious AE,<br/>or 2 device-related deaths"]:::goal
    CONT["Continue escalation"]:::goal
    ENR --> SAF & DLT & PP & MITT
    SAF --> STOP
    STOP -->|threshold met| HALT
    STOP -->|below threshold| CONT
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** The four analysis populations (safety, dose-limiting-toxicity
evaluable, per-protocol, modified intention-to-treat) and the cohort-level interim
safety review with prespecified DSMB halt rules (for example, a device-related
serious-AE rate at or above one third, or two device-related deaths).

**Role in the protocol.** Defines &sect;9.3 populations and &sect;9.4.6 interim
analyses / stopping logic.

**Source files.** `nih-protocol/07_statistical_considerations.md` (populations,
interim analysis); `research/ChatGPT-5.5-Thinking-Extended-19Jun26.md`
(30-day serious-AE primary safety endpoint).
