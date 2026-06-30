## Figure 11. Figure grounding: each Mermaid source reproduced as an identical TikZ figure

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart LR
    SRC["mermaid/fig-NN.md<br/>(GitHub Mermaid)"]:::input
    CON["nodes, edges, grayscale<br/>palette, quantitative data"]:::proc
    TIKZ["TikZ mermaidfig<br/>(mm* node styles)"]:::proc
    OUT["Identical figure in<br/>draft / full / final"]:::goal
    V{"Verify twice:<br/>no overlaps, arrow<br/>looseness, box spacing"}:::dec
    SRC --> CON --> TIKZ --> OUT
    OUT --> V
    V -->|"pass"| OUT
    V -->|"fail"| TIKZ
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef dec fill:#D9D9D9,stroke:#000000,stroke-width:1px,color:#000000
```

**Caption.** Figure grounding. Each Mermaid source in `mermaid/` is reproduced as a
TikZ `mermaidfig` of the same complexity (the same nodes, edges, grayscale palette,
and quantitative data), then verified twice for text-box and arrow overlaps, for
the specified arrow looseness, and for proper box spacing, before it appears
identically in the draft, full, and final stages.

**Role in the IND.** Renders in §11 (Relevant Information) as the
Mermaid-to-LaTeX translation and verification discipline that keeps every figure
identical from Python to LaTeX.

**Source files.**
`trial-documents/final-paper/publication/sections/sec-04-results.tex` (Figure 15,
figure grounding, adapted in context);
`trial-ind/draft-ind/indstyle.sty` (the `mm*` node styles and `mermaidfig`
environment).
