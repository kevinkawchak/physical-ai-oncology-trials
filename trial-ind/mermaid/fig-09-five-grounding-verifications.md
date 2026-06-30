## Figure 9. Five name-matching and grounding verifications for the IND

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart LR
    K1{"1 Grounding"}:::dec
    K2{"2 Figures match context"}:::dec
    K3{"3 GitHub real-time"}:::dec
    K4{"4 Repo / dir names match"}:::dec
    K5{"5 File names match"}:::dec
    E1["22 grayscale Mermaid<br/>figures, each sourced"]:::input
    E2["each figure cited where<br/>its §section discusses it"]:::input
    E3["one commit per file,<br/>auto-pushed to one PR"]:::input
    E4["trial-ind/ matches the<br/>repository tree"]:::input
    E5["sec-06-cmc.tex matches<br/>the CMC section"]:::input
    G["Grounded, monitorable IND"]:::goal
    K1 --> E1 --> G
    K2 --> E2 --> G
    K3 --> E3 --> G
    K4 --> E4 --> G
    K5 --> E5 --> G
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef dec fill:#D9D9D9,stroke:#000000,stroke-width:1px,color:#000000
```

**Caption.** Five name-matching and grounding verifications, each with a concrete
IND example: grounding (every one of the 22 grayscale figures names its source
files); figures match context (each is cited where its section discusses it);
GitHub real-time (one commit per file, auto-pushed to one continuously updated
pull request); repository and directory names match (the `trial-ind/` tree mirrors
the document structure); and file names match (for example `sec-06-cmc.tex` maps to
the Chemistry, Manufacturing and Control section). Together they yield a grounded,
monitorable IND.

**Role in the IND.** Renders in §11 (Relevant Information) as the verification
discipline the build applies, and in the Introduction.

**Source files.**
`trial-documents/final-paper/publication/sections/sec-04-results.tex` (Figure 20,
the five verifications, adapted in context with IND examples).
