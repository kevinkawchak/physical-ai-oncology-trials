## Figure 10. New 1-4 day document iteration cadence versus the traditional baseline

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    SRC[/"Source inputs<br/>prompt-paper.md OUTLINE<br/>llm-adoption main.tex"/]:::input

    SRC --> TRAD_LANE
    SRC --> NEW_LANE

    subgraph TRAD_LANE["Traditional baseline: weeks to months per iteration"]
        direction TB
        T1["Sequential manual drafting<br/>large document by hand"]:::ctx
        T2["Manual tables and figures<br/>hand built per section"]:::ctx
        T3["Multiple review rounds<br/>edit and recirculate"]:::ctx
        TGATE{"Iteration<br/>complete?"}:::warn
        T1 --> T2 --> T3 --> TGATE
        TGATE -->|"No: revise again"| T1
        TGATE -->|"Yes"| TOUT["One large document<br/>weeks to months"]:::warn
    end

    subgraph NEW_LANE["New cadence: 1-4 days per iteration"]
        direction TB
        P0(["Single master prompt<br/>creates and runs sub-prompts"]):::proc
        D1["Draft document<br/>day 1: bracketed instructions<br/>plus repo file pointers"]:::accent
        D2["Full document<br/>days 2-3: source files<br/>resolved into full version"]:::accent
        D3["Final document<br/>day 4: max quality<br/>formatting and proofing"]:::accent
        NGATE{"Quality<br/>verified?"}:::warn
        P0 --> D1 --> D2 --> D3 --> NGATE
        NGATE -->|"No: refine in place"| P0
        NGATE -->|"Yes"| NOUT["One new document iteration<br/>1-4 days"]:::accent
    end

    TOUT -. "weeks to months elapsed" .-> COMPARE{{"Cadence<br/>comparison"}}:::warn
    NOUT -. "1-4 days elapsed" .-> COMPARE

    NOUT ==> GOAL["Administrative and prep time compressed<br/>faster paperwork, faster treatment"]:::goal
    COMPARE -->|"weeks-to-months down to 1-4 days"| GOAL

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** Two-lane comparison of document iteration cadence for the Phase 1 pancreatic cancer trial. The traditional baseline lane runs sequential manual drafting, hand built tables and figures, and repeated review rounds, requiring weeks to months per large-document iteration. The new lane executes a single master prompt that produces a draft (day 1), a full version (days 2-3), and a final version (day 4), completing one new document iteration in 1-4 days. The compressed cadence feeds the patient-facing goal of reduced administrative and preparation time, supporting faster paperwork and faster treatment.

**Role in the paper.** This figure appears in Results and Discussion as the quantitative contrast motivating the single-prompt mermaid-draft-full-final workflow, and it becomes a TikZ mermaidfig in the draft, full, and final LaTeX stages.

**Source files.** 
- prompts/prompt-paper.md (OUTLINE: New Document Iterations 1-4 Days; Single Prompt: Draft to Full to Final; Mechanisms Compresses Administrative/Prep Time)
- inputs/llm-adoption/main.tex (single-prompt draft, full, final document generation with sub-prompts)
