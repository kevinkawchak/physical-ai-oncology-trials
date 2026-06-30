## Figure 12. The real-world daraxonrasib PDAC document thread

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart LR
    PT["KRAS-mutated PDAC patient<br/>daraxonrasib + robotic Whipple"]:::input
    P1["Initial IND (this build)<br/>Phase 1, 3+3, n=18"]:::goal
    CR["3+3 cohort-review<br/>packages; establish RP2D"]:::proc
    AM["Amendments +<br/>synchronized consent"]:::proc
    TR["Phase 1-to-2 CSR<br/>+ EOP2 briefing"]:::proc
    P2["Phase 2 randomized<br/>protocol (300 mg RP2D)"]:::goal
    M["Each document built by the<br/>single-prompt workflow"]:::accent
    PT --> P1 --> CR --> AM --> TR --> P2
    M -.-> AM
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef accent fill:#6C757D,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
```

**Caption.** The real-world daraxonrasib PDAC document thread, from the
KRAS-mutated patient and the intervention through this initial Phase 1 IND, the
3+3 cohort-review packages that establish the recommended Phase 2 dose, the
amendments with synchronized consent, and the Phase 1-to-2 clinical study report
and end-of-Phase-2 briefing, to the Phase 2 randomized protocol at the 300 mg
recommended Phase 2 dose, each document built by the single-prompt workflow.

**Role in the IND.** Renders in the Introduction (§3.2 Summary of Previous Human
Experience) and §4.4 (Description of First Year Trials), threading the IND into the
real program.

**Source files.**
`trial-documents/final-paper/publication/sections/sec-05-discussion.tex` (Figure
23, the daraxonrasib document thread, adapted in context);
`trial-protocol/final-protocol/publication/sections/sec-04-design.tex` (the 3+3 to
RP2D path).
