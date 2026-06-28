## Figure 24. Complete clinical-hold response and the 30-day review clock (acceleration target 4)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart LR
    HOLD["FDA clinical hold<br/>Step 8 absolute hard gate<br/>held trial cannot resume"]:::warn

    subgraph COMP["Complete response components"]
        direction TB
        C1["Revised protocol"]:::input
        C2["New nonclinical analyses"]:::input
        C3["CMC information"]:::input
        C4["Updated risk assessment"]:::input
        C5["Revised consent language"]:::input
    end

    ASSEMBLE["Assemble complete response<br/>address EVERY deficiency<br/>internally consistent package"]:::proc
    SUBMIT{"Response<br/>complete?"}:::warn
    REVIEW["FDA review<br/>30 calendar days<br/>once complete response received"]:::proc
    NOTIFY["FDA notifies sponsor<br/>hold lifted"]:::proc
    RESUME["Enrollment resumes<br/>dosing restarts<br/>patient outcome"]:::goal

    ACCEL["Faster assembly moves<br/>restart date forward<br/>every day saved counts"]:::accent
    CYCLE["Incomplete response<br/>does not start useful clock<br/>risks another cycle"]:::ctx

    HOLD --> COMP
    C1 --> ASSEMBLE
    C2 --> ASSEMBLE
    C3 --> ASSEMBLE
    C4 --> ASSEMBLE
    C5 --> ASSEMBLE
    ASSEMBLE --> SUBMIT
    SUBMIT -->|"all deficiencies addressed"| REVIEW
    SUBMIT -.->|"deficiency remains"| CYCLE
    CYCLE -.->|"re-author and resubmit"| ASSEMBLE
    REVIEW -->|"day 30 or sooner"| NOTIFY
    NOTIFY --> RESUME
    ACCEL -.->|"shorten pre-submission days"| ASSEMBLE
    ACCEL -.->|"earlier restart"| RESUME

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** This flow traces the response to an FDA clinical hold (Step 8), an absolute hard gate under which a held trial cannot resume until FDA notifies the sponsor. A complete response must address every deficiency, assembling a revised protocol, new nonclinical analyses, CMC information, an updated risk assessment, and revised consent language into one internally consistent package. Once FDA receives a complete response, the agency reviews it within 30 calendar days, after which the hold is lifted and enrollment resumes. The looping caveat path shows that an incomplete response does not start the useful review clock and risks another cycle, while faster pre-submission assembly moves the potential restart date forward.

**Role in the paper.** It appears in the Results/Discussion as the fourth acceleration target, illustrating where faster LLM document generation has the greatest schedule value, and it becomes a TikZ mermaidfig in the draft, full, and final LaTeX stages.

**Source files.** research/document-types/ChatGPT-5-5-Thinking-Extended-DocTypes-2026-06-26.md (step 8 clinical hold)
