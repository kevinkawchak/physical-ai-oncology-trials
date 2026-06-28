## Figure 14. Figure grounding: Mermaid reproduced as identical TikZ

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart LR
    SRC["Mermaid source<br/>mermaid/fig-NN.md<br/>1 of 25 catalog files"]:::input
    DEF["Source defines<br/>nodes, edges, palette<br/>quantitative data"]:::proc
    QD["Quantitative spine<br/>n=18, 3+3, USL &ge; 7.0<br/>640 channels, &le; 3 ms E-stop"]:::input
    STY["paperstyle.sty / protostyle.sty<br/>mermaidfig + mm* styles<br/>fboxrule 0.4pt, fboxsep 9pt"]:::proc
    REPRO["Reproduced as TikZ<br/>mermaidfig environment<br/>same nodes, edges, labels"]:::proc
    SAME["Same complexity<br/>Python-to-LaTeX render<br/>no node dropped"]:::accent
    VER{"Verify twice:<br/>no overlaps, arrow looseness<br/>proper box spacing"}:::warn
    DRAFT["draft LaTeX<br/>bracketed pointer<br/>to each figure"]:::proc
    FULL["full LaTeX<br/>TikZ render<br/>of each figure"]:::proc
    FINAL["final LaTeX<br/>identical figure<br/>polished + zip"]:::goal

    SRC -->|opens with| DEF
    DEF -->|carries| QD
    DEF -->|grounded into| STY
    QD -.preserved 1:1.-> REPRO
    STY -->|mm* node styles| REPRO
    REPRO -->|matched against| SAME
    SAME --> VER
    VER -->|fail: re-render| REPRO
    VER -->|pass| FULL
    DRAFT -->|pointer| FULL --> FINAL

    P1["classDef goal<br/>#8B2E3F maroon"]:::input
    P2["classDef proc<br/>#2F5D7C steel blue"]:::input
    P3["classDef accent<br/>#D08770 terracotta"]:::input
    P4["classDef input<br/>#BFD7EA light blue"]:::input
    P5["classDef warn<br/>#D9D9D9 gray"]:::input
    M1["mmgoal<br/>fill protoblue"]:::proc
    M2["mmstep<br/>fill protogray"]:::proc
    M3["mmdark<br/>fill protodark"]:::proc
    M4["mmin<br/>fill protowhite"]:::proc
    M5["mmdec<br/>fill protol3"]:::proc
    P1 -->|maps 1:1| M1
    P2 -->|maps 1:1| M2
    P3 -->|maps 1:1| M3
    P4 -->|maps 1:1| M4
    P5 -->|maps 1:1| M5
    STY -.defines.-> M1

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** Grounding pipeline that turns each of the 25 GitHub-rendered Mermaid sources in mermaid/fig-NN.md into an identical TikZ figure: the source defines nodes, edges, palette, and quantitative data (n=18, 3+3 escalation, USL &ge; 7.0, 640 sensor channels, &le; 3 ms E-stop), which are reproduced in the mermaidfig environment of paperstyle.sty / protostyle.sty using the mm* node styles. A warn gate verifies twice for no text-box overlaps, correct curved-arrow looseness, and proper box spacing, looping back to re-render on failure; the accent node asserts the same-complexity Python-to-LaTeX render drops no node. The five-color block shows the classDef roles mapping one-to-one onto mmgoal, mmstep, mmdark, mmin, and mmdec.

**Role in the paper.** Appears in the Methods as the figure-grounding contract for the visual artifacts, documenting how every Mermaid source becomes a TikZ mermaidfig across the draft, full, and final LaTeX stages.

**Source files.** `mermaid/*` (the 25 fig-NN.md sources, README.md classDef-to-mm* mapping table, and output-mermaid.md narrative); `trial-protocol/final-protocol/publication/protostyle.sty` (the `mermaidfig` environment and the `mmgoal`, `mmstep`, `mmdark`, `mmin`, `mmdec` node styles).
