## Figure 13. Real-time auto-commit and auto-PR monitorability loop

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart LR
    GEN["LLM generates one file<br/>main.tex, .sty, .bib,<br/>README, or section .tex"]:::proc
    COMMIT["git commit<br/>one commit per file<br/>do not hold commits"]:::proc
    PUSH["Push to feature branch<br/>real-time, no user<br/>intervention"]:::proc
    PR["Single continuously<br/>updated pull request<br/>auto-PR, always current"]:::accent
    REVIEW["Author reviews files<br/>on GitHub in real time<br/>human-in-the-loop"]:::goal
    GATE{"Errors or<br/>gaps found?"}:::warn
    FINAL["Final commit<br/>CHANGELOG, releases,<br/>v4.2.0 updates"]:::proc

    NOTE["One commit per file<br/>per Rule 6, 7, 8<br/>10+ to 20+ commits"]:::ctx
    OBS["Branch progress observable<br/>throughout generation<br/>monitorability"]:::ctx

    GEN -->|"file emitted"| COMMIT
    COMMIT -->|"the moment generated"| PUSH
    PUSH -->|"updates open PR"| PR
    PR -->|"diff visible live"| REVIEW
    REVIEW --> GATE
    GATE -->|"yes: feedback"| GEN
    GATE -->|"no: next file"| GEN
    REVIEW -.->|"curved feedback loop"| GEN
    GEN -->|"after all files"| FINAL
    FINAL -->|"last commit"| PR

    NOTE -.-> COMMIT
    OBS -.-> REVIEW

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** This loop shows the real-time auto-commit and auto-PR process that keeps the human author in the loop throughout an extensive LLM generation. As each file is generated (main.tex, .sty, .bib, README, or a per-section .tex), the LLM issues exactly one commit, pushes it to the feature branch the moment it is generated, and updates a single continuously open pull request rather than holding commits from GitHub. The author reviews the live diff on GitHub and feeds corrections back into generation (curved feedback edge), and a single final commit delivers the CHANGELOG, releases, and v4.2.0 repository updates per Rule 6, Rule 7, and Rule 8.

**Role in the paper.** It appears in the Methods/Discussion sections describing human-in-the-loop monitorability and observability, and it becomes a TikZ mermaidfig in the draft, full, and final LaTeX stages.

**Source files.**
- prompts/prompt-paper.md (auto-commit / auto-PR process; do not hold commits; one commit per file per Rules 6, 7, 8)
- inputs/llm-adoption/main.tex (human-in-the-loop monitorability and observability via continual push to GitHub)
