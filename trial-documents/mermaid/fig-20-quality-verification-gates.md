## Figure 20. Document quality gates that prevent rework cycles

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart LR
    DRAFT[LLM draft document<br/>generated in 1-4 days]:::input

    G1{Gate 1<br/>Internally consistent?}:::warn
    G2{Gate 2<br/>Supported by<br/>validated data?}:::warn
    G3{Gate 3<br/>Tables match<br/>body text width?}:::warn
    G4{Gate 4<br/>References clickable<br/>DOI plus URL?}:::warn
    G5{Gate 5<br/>Single dashes,<br/>correct symbols?}:::warn
    G6{Gate 6<br/>No single-word<br/>orphan lines?}:::warn

    REVISE[Revise<br/>targeted fix only]:::accent
    PASS[Submission-ready<br/>no extra cycle]:::goal

    NOTE[Caveat: poorly supported docs<br/>spawn amendments, regulator<br/>questions, IRB revisions,<br/>an extra review cycle = slower]:::ctx

    DRAFT --> G1
    G1 -->|pass| G2
    G2 -->|pass| G3
    G3 -->|pass| G4
    G4 -->|pass| G5
    G5 -->|pass| G6
    G6 -->|all 6 pass| PASS

    G1 -.->|fail| REVISE
    G2 -.->|fail| REVISE
    G3 -.->|fail| REVISE
    G4 -.->|fail| REVISE
    G5 -.->|fail| REVISE
    G6 -.->|fail| REVISE
    REVISE ==>|re-enter at Gate 1| G1

    REVISE -.->|skip gates = risk| NOTE
    NOTE -.->|avoided when<br/>gates enforced| PASS

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** An LLM-generated draft (produced in 1 to 4 days) passes left to right through six verification gates: internal consistency, support by validated data, tables matching body-text width, references clickable with both DOI and URL, single dashes with correct symbols (for example &sect;), and no single-word orphan lines. Passing all six gates routes the document to the maroon goal node "Submission-ready, no extra cycle," while any failure routes it through the terracotta "Revise" node and loops it back to re-enter at Gate 1 with a targeted fix. The near-white context note records the quality caveat that poorly supported or inconsistent documents make the program slower by generating amendments, regulator questions, IRB revisions, and an additional review cycle.

**Role in the paper.** This figure appears in the Methods section as the document quality-control workflow and later becomes a TikZ mermaidfig in the draft, full, and final LaTeX paper stages.

**Source files.**
- trial-documents/research/document-types/ChatGPT-5-5-Thinking-Extended-DocTypes-2026-06-26.md (quality caveat: poorly supported documents generate amendments, regulator questions, IRB revisions, and an extra review cycle)
- trial-documents/prompts/prompt-paper.md (formatting rules: clickable DOI plus URL references, body-width tables, single dashes, symbol correction such as &sect;, no single-word or two-word orphan lines)
