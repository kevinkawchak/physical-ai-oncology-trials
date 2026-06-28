## Figure 12. Repository-based LLM document-generation architecture

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart LR
    subgraph IN["Inputs: trial-documents/inputs"]
        direction TB
        TMPL["llm-adoption/main.tex<br/>template + sample.bib"]:::input
        BIB["references.bib<br/>51 ORCID entries 2024-2026"]:::input
        SRC["pre-chunked sources<br/>+ section READMEs"]:::input
    end

    subgraph CTX["Context layer: 1M-token LLM"]
        direction TB
        ING["Ingest right-sized<br/>high-quality inputs"]:::proc
        LIM["Per-file limits apply<br/>total input under 5MB"]:::warn
        RDM["READMEs under 25K tokens<br/>avoid LLM errors"]:::warn
    end

    subgraph GEN["Generation: sub-prompts 1-4"]
        direction TB
        SECT["Sectioned outputs<br/>one .tex per section"]:::proc
        BUILD["Reusable downstream chunks<br/>direct future access"]:::proc
    end

    subgraph OUT["Outputs"]
        direction TB
        M["mermaid<br/>24 figures + README"]:::goal
        D["draft-paper<br/>sections/ + Overleaf zip"]:::goal
        F["full-paper<br/>sections/ + Overleaf zip"]:::goal
        FN["final-paper<br/>sections/ + Overleaf zip"]:::goal
    end

    ACC["Section-wise outputs become<br/>right-sized input chunks"]:::accent

    TMPL --> ING
    BIB --> ING
    SRC -->|"chunk + README"| ING
    ING --> LIM
    ING --> RDM
    LIM -->|"validated context"| SECT
    RDM -->|"validated context"| SECT
    SECT --> BUILD
    BUILD -->|"one .tex per section"| M
    BUILD --> D
    BUILD --> F
    BUILD --> FN
    M -.->|"feeds stage 2"| ACC
    D -.->|"feeds stage 3"| ACC
    F -.->|"feeds stage 4"| ACC
    ACC -.->|"reused as inputs"| ING

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** This architecture shows how a single repository directory, trial-documents/inputs, supplies the llm-adoption template, references.bib (51 ORCID entries), and pre-chunked sources with section READMEs to a 1M-token context LLM. The context layer ingests appropriately sized, high-quality inputs under the practical bounds of total input below 5MB and READMEs below 25K tokens, then sub-prompts produce sectioned outputs of one .tex file per section for downstream reuse. Generation yields four output stages (mermaid, draft-paper, full-paper, final-paper), each emitting a sections/ directory and an Overleaf zip bundle. The looping accent edge captures how section-wise outputs serve as right-sized chunks that re-enter the context layer for future builds.

**Role in the paper.** It appears in the Methods section as the architectural overview of the repository-based generation pipeline, and it becomes a TikZ mermaidfig in the draft, full, and final LaTeX stages.

**Source files.**
- inputs/llm-adoption/main.tex (&sect; Repository Setup; &sect; LLM Limitations)
- inputs/references.bib
