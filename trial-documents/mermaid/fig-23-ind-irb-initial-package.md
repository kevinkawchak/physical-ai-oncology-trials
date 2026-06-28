## Figure 23. Composition of the initial IND and IRB package (acceleration target 1)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    GOAL["Initial IND + IRB package<br/>highest schedule value<br/>acceleration target 1"]:::goal

    subgraph IND["IND dossier (hard regulatory gate, step 1)"]
        direction TB
        PROTO["Clinical protocol<br/>Phase 1 design, DLT window<br/>dose-escalation algorithm"]:::input
        IB["Investigator's Brochure<br/>known + potential risks<br/>reference safety information"]:::input
        TOX["Nonclinical pharm/tox<br/>summaries and reports"]:::input
        CMC["CMC and stability info<br/>process, specs, batches"]:::input
        REGFORMS["Investigator info<br/>and regulatory forms<br/>administrative + regional"]:::input
    end

    subgraph IRBSET["IRB-specific items (hard ethical/site gate, step 2)"]
        direction TB
        ICF["Informed consent form<br/>built from protocol + IB"]:::input
        RECRUIT["Recruitment materials<br/>ads, prescreening scripts"]:::input
        SAFEINFO["Safety information<br/>and trial procedures"]:::input
    end

    CLOCK["FDA review clock starts<br/>at submission<br/>30 calendar day wait"]:::proc
    IRBREV["IRB review of protocol<br/>ICF and IB<br/>approve, change or disapprove"]:::proc

    ACCEL["Faster, internally consistent<br/>assembly starts the clock<br/>and all sites sooner"]:::accent

    WARN["Caveat: cannot shorten<br/>30-day period or generate<br/>missing tox/stability data"]:::warn

    PROTO --> GOAL
    IB --> GOAL
    TOX --> GOAL
    CMC --> GOAL
    REGFORMS --> GOAL
    ICF --> GOAL
    RECRUIT --> GOAL
    SAFEINFO --> GOAL

    GOAL -->|"submit to FDA"| CLOCK
    GOAL -->|"submit to all sites"| IRBREV
    ACCEL -.->|"earlier complete package"| GOAL
    ACCEL -->|"high impact"| CLOCK
    CLOCK -.->|"30 days fixed, no shortcut"| WARN
    WARN -.->|"data gaps reopen authoring cycle"| GOAL

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** This figure decomposes the initial IND and IRB package, the single highest-schedule-value document set, into its feeding inputs: the clinical protocol, Investigator's Brochure, nonclinical pharmacology/toxicology summaries, CMC and stability information, and investigator information with regulatory forms, plus the IRB-specific informed consent form, recruitment materials and safety information. Submission triggers two parallel gates: the 30 calendar day FDA review clock and IRB review of the protocol, ICF and IB. Faster, internally consistent assembly is high impact because it starts the FDA clock earlier and can be submitted to all sites sooner, while the looping caveat edge marks the hard limits, namely that authoring speed cannot shorten the fixed 30-day period or generate missing toxicology or stability data, and that data gaps reopen the cycle.

**Role in the paper.** It appears in the Methods/Results discussion of where faster document production has the greatest schedule value (acceleration target 1) and becomes a TikZ mermaidfig in the draft, full and final LaTeX stages.

**Source files.** research/document-types/ChatGPT-5-5-Thinking-Extended-DocTypes-2026-06-26.md (steps 1-2 and the schedule-value ranking); research/industry-workflow/ (ChatGPT-5-5-Thinking-Extended-Workflow-2026-06-26.md initial IND/CTA dossier and core clinical document sections, Gemini-3-1-Pro-Workflow-2026-06-26.md, prompt-workflow.md).
