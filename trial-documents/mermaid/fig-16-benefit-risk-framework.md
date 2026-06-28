## Figure 16. Probable benefit greater than probable risk for faster administration

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    subgraph RISKS[Probable Risk - LLM Limitations]
        direction TB
        R1[Possible LLM error<br/>404 / server errors<br/>on oversized input]:::input
        R2[Formatting artifacts<br/>e.g. =0mu plus 3mu<br/>ToC overlap]:::input
        R3[Hallucinated context<br/>unverified URLs<br/>rare sentence errors]:::input
    end

    subgraph MIT[Mitigations - Process Controls]
        direction TB
        M1[Human-in-the-loop<br/>review per commit]:::proc
        M2[Mermaid grounding<br/>20+ figures match context]:::proc
        M3[Name-matching<br/>paper vs repo files]:::proc
        M4[Real-time GitHub<br/>monitorability]:::proc
        M5[Author proofreading<br/>final edits]:::proc
    end

    subgraph BEN[Probable Benefit - Speed Gains]
        direction TB
        B1[Faster paperwork<br/>1 to 4 day iterations]:::accent
        B2[Faster treatment<br/>single prompt to final]:::accent
        B3[Compressed admin<br/>and prep time]:::accent
        B4[More figures<br/>for oversight]:::accent
    end

    D{Probable benefit<br/>vs probable risk}:::warn

    R1 -.->|controlled by| M1
    R2 -.->|controlled by| M2
    R2 -.->|controlled by| M3
    R3 -.->|controlled by| M4
    R3 -.->|controlled by| M5

    M1 ==>|residual risk low| D
    M2 ==>|residual risk low| D
    M3 ==>|residual risk low| D
    M4 ==>|residual risk low| D
    M5 ==>|residual risk low| D

    B1 ==>|weight benefit| D
    B2 ==>|weight benefit| D
    B3 ==>|weight benefit| D
    B4 ==>|weight benefit| D

    D ==>|benefit outweighs risk| G[Benefit greater than Risk<br/>for enrolled patients<br/>who cannot wait]:::goal
    G -.->|extends patient lives| BEN

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** This balance figure weighs the probable risks of repository-based LLM document generation (possible LLM or server errors, formatting artifacts, and hallucinated context) against the probable benefits of faster administration (faster paperwork at 1 to 4 day iterations, faster treatment via single-prompt draft-to-full-to-final outputs, compressed administrative and preparation time, and more figures for oversight). Each risk is reduced to low residual risk by a process control: human-in-the-loop review, mermaid grounding against paper context, paper-to-repository name-matching verification, real-time GitHub monitorability, and author proofreading. The central decision diamond integrates mitigated risk and weighted benefit, and the looping edge back to the benefits subgraph reinforces that realized speed gains extend patient lives. The conclusion is that probable benefit outweighs probable risk for enrolled patients who cannot wait.

**Role in the paper.** It appears in the Discussion as the concluding argument for OUTLINE theme 3 (Probable Benefit greater than Probable Risk), and it becomes a TikZ mermaidfig across the draft, full, and final LaTeX stages.

**Source files.**
- prompts/prompt-paper.md (OUTLINE 3: Probable Benefit greater than Probable Risk)
- inputs/llm-adoption/main.tex (LLM Limitations)
