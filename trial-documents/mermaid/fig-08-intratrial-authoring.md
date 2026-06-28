## Figure 8. During-trial document authoring driven by safety data

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    subgraph TRIG[Trigger inputs during the trial]
        T1[SAE or SUSAR<br/>safety event reported]:::input
        T2[Emerging PK / PD data<br/>unexpected exposure]:::input
        T3[Dose-cohort completion<br/>DLT window closed]:::input
    end

    subgraph DOCS[Safety-driven documents authored]
        D1[Clinical safety narratives<br/>7-day fatal or<br/>life-threatening window]:::proc
        D2[Clinical safety narratives<br/>15-day other<br/>SUSAR window]:::proc
        D3[DSUR annual<br/>cumulative safety report<br/>ICH E2F]:::proc
        D4[Protocol amendment<br/>what changes and why<br/>before implementation]:::proc
        D5[Dose-escalation minutes<br/>and briefing books<br/>DLT listings, AE tables]:::proc
    end

    LLM[LLM synchronized authoring<br/>protocol + consent +<br/>database + site docs together]:::accent

    subgraph PI[Principal Investigator role]
        P1{PI assesses causality<br/>drug-related?}:::warn
        P2[PI reviews narratives<br/>for medical accuracy]:::goal
        P3[PI justifies amendment<br/>scientifically]:::goal
        P4{PI approves cohort<br/>escalation?}:::warn
    end

    G1[Submit to regulators<br/>IRB and FDA / EMA<br/>within window]:::goal
    G2[Next dose cohort opens<br/>or stopping rule applied]:::goal

    T1 --> D1
    T1 --> D2
    T1 -.-> D3
    T2 -- revise dosing --> D4
    T2 -.-> D3
    T3 --> D5

    D1 --> P2
    D2 --> P2
    P2 --> P1
    P1 -- causality confirmed --> G1
    P1 -. not related, document .-> D3

    D4 --> P3
    P3 --> LLM
    LLM -- one coordinated package --> G1
    LLM -. re-consent participants .-> P2

    D5 --> P4
    P4 -- no new DLT, escalate --> G2
    P4 -. DLT exceeds threshold .-> D4

    D3 --> G1
    G1 -. emerging signal reopens .-> T1
    G2 -. new cohort data accrues .-> T2

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** This flowchart maps during-trial document authoring that is driven by safety data in a Phase 1 pancreatic cancer trial. Trigger inputs (SAE or SUSAR events, emerging PK/PD data, and dose-cohort completion) initiate clinical safety narratives submitted within 7-day fatal or life-threatening and 15-day SUSAR windows, the annual cumulative DSUR, protocol amendments detailing what changes and why, and dose-escalation minutes with briefing books. The Principal Investigator reviews narratives for medical accuracy, assesses causality, justifies amendments scientifically, and approves cohort escalation, while the terracotta node shows the LLM authoring protocol, consent, database, and site documents together rather than sequentially. Looping edges capture how submissions and new cohort data can reopen the safety cycle.

**Role in the paper.** It appears in the Methods/Results discussion of intratrial document generation, illustrating where coordinated LLM authoring shortens enrollment pauses; in the draft, full, and final LaTeX stages it becomes a TikZ mermaidfig.

**Source files.**
- research/industry-workflow/Gemini-3-1-Pro-Workflow-2026-06-26.md (section C: safety narratives and SUSARs, DSUR, protocol amendments, dose-escalation meeting minutes)
- research/document-types/ChatGPT-5-5-Thinking-Extended-DocTypes-2026-06-26.md (protocol amendments and synchronized consent/site updates, responding to a new safety signal)
