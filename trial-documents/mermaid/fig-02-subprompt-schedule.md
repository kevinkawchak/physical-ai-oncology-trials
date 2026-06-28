## Figure 2. Process A generation and Process B execution under one master prompt

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    MP["Master prompt<br/>prompts/prompt-paper.md<br/>single prompt, generate then execute"]:::goal

    subgraph PA["Process A - generate"]
        direction TB
        SP1["prompt-1-mermaid.md<br/>Stage 1 sub-prompt"]:::input
        SP2["prompt-2-draft-paper.md<br/>Stage 2 sub-prompt"]:::input
        SP3["prompt-3-full-paper.md<br/>Stage 3 sub-prompt"]:::input
        SP4["prompt-4-final-paper.md<br/>Stage 4 sub-prompt"]:::input
    end

    subgraph PB["Process B - execute sequentially"]
        direction TB
        ST1["Stage 1 mermaid<br/>24 colored figures<br/>24 commits"]:::proc
        ST2["Stage 2 draft<br/>scaffold, 8 sections<br/>bracketed instructions, 10+ commits"]:::proc
        ST3["Stage 3 full<br/>prose, TikZ, tables<br/>10+ commits"]:::proc
        ST4["Stage 4 final<br/>polished, clearpage, zip<br/>10+ commits"]:::goal
    end

    ACC["Project-scale context<br/>outputs accumulate<br/>across stages"]:::accent

    MP ==>|"feeds Process A"| PA
    SP1 -.->|"executes as"| ST1
    SP2 -.->|"executes as"| ST2
    SP3 -.->|"executes as"| ST3
    SP4 -.->|"executes as"| ST4
    ST1 -->|"next"| ST2
    ST2 -->|"next"| ST3
    ST3 -->|"next"| ST4
    ST1 -.->|"adds context"| ACC
    ST2 -.->|"adds context"| ACC
    ST3 -.->|"adds context"| ACC
    ACC -.->|"informs next stage"| ST4

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** A single master prompt (prompts/prompt-paper.md) first drives Process A, which generates the four stage sub-prompt files, and then drives Process B, which executes those sub-prompts sequentially. Each sub-prompt maps to one build stage: Stage 1 mermaid (24 colored figures, 24 commits), Stage 2 draft (a scaffold with bracketed drafting instructions, 10+ commits), Stage 3 full (full prose, TikZ figures, and tables, 10+ commits), and Stage 4 final (the polished paper and Overleaf zip, 10+ commits). The dashed loop shows project-scale outputs accumulating as context that informs each subsequent stage, so the final paper is produced under one prompt rather than four independent runs.

**Role in the paper.** Appears in Methods to define the sub-prompt schedule that produces the paper, and it becomes a TikZ mermaidfig in the draft, full, and final LaTeX stages.

**Source files.**
- prompts/prompt-paper.md (the master prompt and SUB-PROMPT SCHEDULE)
- sub-prompts/ (prompt-1-mermaid.md, prompt-2-draft-paper.md, prompt-3-full-paper.md, prompt-4-final-paper.md, README.md)
- inputs/llm-adoption/main.tex (Prompt Proficiency section)
