## Figure 19. Five-color palette and Mermaid-to-TikZ fidelity scheme

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    subgraph PAL[Five-step palette plus grayscale gates]
        direction TB
        G["goal #8B2E3F<br/>deep maroon<br/>end goals outcomes decisions"]:::goal
        P["proc #2F5D7C<br/>steel blue<br/>LLM system process"]:::proc
        A["accent #D08770<br/>terracotta<br/>acceleration emphasis"]:::accent
        I["input #BFD7EA<br/>light blue<br/>inputs sources"]:::input
        C["ctx #F4F7F9<br/>near-white<br/>context support"]:::ctx
        W["warn #D9D9D9<br/>gray<br/>gates decisions"]:::warn
    end

    TIKZ["TikZ mm* node style<br/>paperstyle.sty tikzset<br/>mmgoal mmproc mmaccent<br/>mmin mmctx mmdec"]:::proc
    OUT["Identical figure in LaTeX<br/>mermaidfig recolors to<br/>five-step palette"]:::goal
    TXT["Body text stays black<br/>figures carry the color"]:::ctx

    G -- "fill paperred" --> TIKZ
    P -- "fill paperblue" --> TIKZ
    A -- "fill papersand" --> TIKZ
    I -- "fill paperlight" --> TIKZ
    C -- "fill paperbg" --> TIKZ
    W -- "fill paperwarn" --> TIKZ

    TIKZ == "render fidelity 1:1" ==> OUT
    TXT -. "color rule kept" .-> OUT
    OUT -. "same palette next stage" .-> TIKZ

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** The legend defines the six fill classes used across every figure in the paper: goal (#8B2E3F deep maroon) for end goals, outcomes, and decisions; proc (#2F5D7C steel blue) for LLM, system, and process nodes; accent (#D08770 terracotta) for acceleration and emphasis; input (#BFD7EA light blue) for inputs and source files; ctx (#F4F7F9 near-white) for context and support; and warn (#D9D9D9 gray) for gates and decision diamonds. Each class maps by a labeled edge to the matching TikZ mm* node style in paperstyle.sty (mmgoal, mmproc, mmaccent, mmin, mmctx, mmdec), which in turn produces an identical figure in LaTeX through the mermaidfig environment. White text reads on the maroon and steel-blue fills while dark text reads on the light fills, and the looping edge shows the same palette being reused at each subsequent draft-full-final stage.

**Role in the paper.** This figure appears in the Methods section as the rendering and styling key for all other figures, and it becomes a TikZ mermaidfig in the draft, full, and final LaTeX stages so the GitHub-rendered Mermaid and the compiled paper carry the same colors.

**Source files.**
- prompts/prompt-paper.md (the 5 color scheme: #D08770, #8B2E3F, #2F5D7C, #BFD7EA, #F4F7F9 plus grayscale, with black body text kept)
- paperstyle.sty (the mm* TikZ node styles: mmgoal, mmproc, mmaccent, mmin, mmctx, mmdec, and the mermaidfig environment)
