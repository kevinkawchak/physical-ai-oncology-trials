## Figure 17. Patient time-saved cascade extends PFS and OS

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart LR
    A["Faster validated<br/>document authoring<br/>(draft to full to final)<br/>1-4 day iterations"]:::accent
    B["Earlier IND and IRB<br/>submission<br/>plus earlier<br/>site activation"]:::proc
    C["Earlier first dose<br/>and faster<br/>cohort transitions<br/>to RP2D"]:::proc
    D["Shorter administrative<br/>white space<br/>between phases<br/>(CSR and protocol)"]:::proc
    E["Earlier access<br/>to effective therapy<br/>for enrolled patients<br/>(Daraxonrasib)"]:::goal
    F["Extended PFS<br/>and overall survival<br/>(OS) for<br/>enrolled patients"]:::goal
    G["Caveat: does not change<br/>tumor biology nor<br/>fixed regulatory<br/>review clocks"]:::ctx
    H["Compresses<br/>administrative and<br/>prep time only<br/>(1 of 3 buckets)"]:::accent

    A -- "compress prep time" --> B
    B -- "recruit sooner" --> C
    C -- "mature safety data" --> D
    D -- "shrink dead time" --> E
    E -- "probable benefit<br/>over probable risk" --> F
    A -. "shaves months cumulatively" .-> H
    H -. "feeds savings" .-> D
    G -. "bounds claim" .-> E
    G -. "fixed 30-day clock" .-> B
    F -- "OUTLINE themes 1 and 3" --> A

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** This causal cascade traces how faster validated document authoring (1-4 day draft-to-full-to-final iterations) propagates into earlier IND/IRB submission and site activation, earlier first dose with faster cohort transitions to the recommended Phase 2 dose, and shorter administrative white space between phases. These compressions deliver earlier patient access to effective therapy (Daraxonrasib) and thereby extend progression-free survival (PFS) and overall survival (OS) for enrolled patients. The dashed branch isolates the mechanism to the administrative and prep-time bucket, which can cumulatively shave months from the timeline, while the near-white caveat node bounds the claim: acceleration does not alter tumor biology or the fixed 30-day regulatory review clock. The looping edge back to authoring marks this as OUTLINE themes 1 and 3, where probable benefit exceeds probable risk.

**Role in the paper.** It appears in Discussion as the mechanistic argument that compressed administrative time yields patient survival benefit, and it becomes a TikZ mermaidfig in the draft, full, and final LaTeX stages.

**Source files.**
- prompts/prompt-paper.md (OUTLINE 1, 3)
- research/document-types/Gemini-3-1-Pro-DocTypes-2026-06-26.md (white space)
