## Figure 1. The single-prompt IND build pipeline

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    MP["Master prompt<br/>prompts/prompt-ind.md"]:::goal
    PA["Process A<br/>write sub-prompts 1-4"]:::proc
    subgraph PB["Process B - execute in order"]
      direction TB
      S1["Stage 1 mermaid<br/>22 grayscale figures"]:::input
      S2["Stage 2 draft-ind<br/>12 sec-*.tex scaffold<br/>+ bracketed instructions"]:::input
      S3["Stage 3 full-ind<br/>full prose + 22 TikZ figures<br/>+ full-width tables"]:::accent
      S4["Stage 4 final-ind<br/>senior-author polish<br/>+ Overleaf zip"]:::goal
    end
    REL["Release v4.3.0<br/>README + CHANGELOG<br/>+ releases + output-ind"]:::dark
    GH["GitHub branch<br/>real-time auto-commit / auto-PR"]:::ctx
    MP --> PA --> S1 --> S2 --> S3 --> S4 --> REL
    S1 -.commit.-> GH
    S2 -.commit.-> GH
    S3 -.commit.-> GH
    S4 -.commit.-> GH
    GH -.author review.-> S4
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef accent fill:#6C757D,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef ctx fill:#F5F5F5,stroke:#6C757D,stroke-width:1px,color:#000000
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.2px,color:#FFFFFF
```

**Caption.** The single-prompt mermaid to draft to full to final build pipeline for
the Phase 1 PDAC IND. One master prompt drives Process A (write the four
sub-prompts) and then Process B (execute them as Stages 1 to 4). Dashed edges show
that every generated file is committed and pushed to one continuously updated pull
request in real time, and the curved feedback edge shows author review returning
into the final stage, all without manual intervention.

**Role in the IND.** Renders as the opening process figure in the Introduction
(§3.1), establishing how the IND package is assembled and monitored.

**Source files.** `trial-ind/prompts/prompt-ind.md` (master prompt);
`trial-ind/sub-prompts/prompt-1-mermaid.md .. prompt-4-final-ind.md` (the four
stages); `trial-documents/final-paper/publication/sections/sec-03-methods.tex`
(the single-prompt method adapted in context, not copied).
