## Figure 3. The large Phase 1 oncology document landscape

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    HUB["Repository-based LLM<br/>document generation<br/>(controlled source reuse,<br/>TLF-to-narrative drafting)"]:::accent

    subgraph BEFORE["Before trial (regulatory and ethics start-up)"]
        direction TB
        IND["Initial IND dossier<br/>nonclinical, CMC, clinical<br/>30-day FDA review clock"]:::input
        IB["Investigator's Brochure (IB)<br/>all preclinical + clinical data<br/>read by all site investigators"]:::input
        PROT["Clinical Trial Protocol<br/>3+3 / Bayesian escalation,<br/>RECIST, DLT stopping rules"]:::input
        ICF["Informed Consent Form (ICF)<br/>20+ pages, severe toxicities,<br/>6th-8th grade reading level"]:::input
        IRB["IRB/IEC package<br/>per-site application,<br/>recruitment + safety info"]:::input
    end

    subgraph DURING["During trial (safety-driven, on critical path)"]
        direction TB
        SAE["SAE/SUSAR safety narratives<br/>PI causality assessment,<br/>7-day / 15-day windows"]:::proc
        DSUR["DSUR annual report<br/>cumulative safety per ICH E2F,<br/>PI risk-benefit assessment"]:::proc
        AMEND["Protocol amendments<br/>dosing/schedule changes,<br/>synced consent + database"]:::proc
        COHORT["Dose-escalation minutes /<br/>cohort-review packages<br/>DLT listings + briefing books"]:::proc
    end

    subgraph AFTER["After trial (reporting and dissemination)"]
        direction TB
        TLF["Tables, Listings, Figures (TLFs)<br/>biostatistics from locked EDC,<br/>ORR + safety outputs"]:::goal
        CSR["Clinical Study Report (CSR)<br/>per ICH E3, thousands of pages,<br/>establishes RP2D"]:::goal
        PUB["Manuscripts / abstracts<br/>JCO, ASCO, ESMO<br/>per GPP guidelines"]:::goal
        LAY["Lay summaries<br/>plain-language results<br/>returned to participants"]:::goal
    end

    HUB ==>|"start-up authoring"| BEFORE
    HUB ==>|"on-study safety drafting"| DURING
    HUB ==>|"post-lock reporting"| AFTER

    BEFORE -->|"DLT observation, dosing"| DURING
    DURING -->|"database lock"| AFTER
    AFTER -.->|"amendment feedback loop"| DURING

    TLF --> CSR
    CSR --> PUB
    CSR --> LAY

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** This map groups the large Phase 1 oncology documents into three temporal lanes: start-up documents created before the trial (initial IND dossier, Investigator's Brochure, Clinical Trial Protocol, a 20+ page Informed Consent Form at a 6th-8th grade reading level, and the IRB/IEC package), safety-driven documents produced during the trial (SAE/SUSAR narratives on 7-day and 15-day windows, the annual DSUR per ICH E2F, protocol amendments, and dose-escalation minutes and cohort-review packages), and reporting documents generated after database lock (TLFs, the CSR per ICH E3 that establishes the recommended Phase 2 dose, manuscripts and abstracts, and participant lay summaries). A central terracotta hub shows repository-based LLM document generation feeding all three lanes through controlled source reuse and table-to-narrative drafting. Lanes flow forward through DLT observation and database lock, with a curved feedback loop where post-trial findings can trigger in-trial amendments.

**Role in the paper.** It appears in the Methods/Background as the document-landscape overview that motivates which artifacts the repository-based LLM workflow targets, and it becomes a TikZ mermaidfig in the draft, full, and final LaTeX stages.

**Source files.** research/document-types/* (ChatGPT-5-5-Thinking-Extended-DocTypes-2026-06-26.md, Gemini-3-1-Pro-DocTypes-2026-06-26.md, prompt-types.md); research/industry-workflow/* (ChatGPT-5-5-Thinking-Extended-Workflow-2026-06-26.md, Gemini-3-1-Pro-Workflow-2026-06-26.md, prompt-workflow.md).
