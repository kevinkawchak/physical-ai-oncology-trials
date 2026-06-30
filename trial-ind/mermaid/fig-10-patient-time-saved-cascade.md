## Figure 10. The patient time-saved cascade from a faster IND

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart LR
    A["Faster validated<br/>IND authoring"]:::accent
    B["Earlier IND / IRB<br/>submission + activation"]:::proc
    C["Earlier 30-day clock<br/>elapses; first dose"]:::proc
    D["Faster 3+3 cohort<br/>turnover to RP2D"]:::proc
    E["Shorter white space<br/>between phases"]:::proc
    F["Earlier access to<br/>effective therapy"]:::goal
    G["Extended PFS / OS<br/>(5-year OS < 13%)"]:::goal
    N["Does not change tumor biology<br/>or fixed review clocks"]:::ctx
    A --> B --> C --> D --> E --> F --> G
    N -.-> F
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef accent fill:#6C757D,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef ctx fill:#F5F5F5,stroke:#6C757D,stroke-width:1px,color:#000000
```

**Caption.** The patient time-saved cascade. Faster validated authoring advances
each step from IND submission to first dose, then through faster 3+3 cohort
turnover to the recommended Phase 2 dose, shortening the white space between phases
and, for this low-survival population (five-year overall survival under 13
percent), extending progression-free and overall survival, without changing tumor
biology or the fixed regulatory review clocks.

**Role in the IND.** Renders in §4.1 (Rationale) and §4.6 (Drug Related Risks) as
the patient-level benefit argument supporting the favorable benefit-risk balance.

**Source files.**
`trial-documents/final-paper/publication/sections/sec-05-discussion.tex` (Figure
24, the patient time-saved cascade, adapted in context);
`trial-protocol/final-protocol/publication/sections/sec-02-introduction.tex`
(PDAC survival baseline).
