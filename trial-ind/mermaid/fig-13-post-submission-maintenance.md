## Figure 13. Post-submission IND maintenance authoring

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    EFF["IND in effect<br/>(30-day clock elapsed)"]:::goal
    SAE["SAE / SUSAR event"]:::input
    PKPD["Emerging PK / PD,<br/>cohort completion"]:::input
    DSMB["DSMB cohort review"]:::input
    N7["IND safety report<br/>7-day (fatal /<br/>life-threatening)"]:::proc
    N15["IND safety report<br/>15-day (other serious,<br/>unexpected)"]:::proc
    DSUR["Annual DSUR;<br/>amendments"]:::proc
    MIN["Dose-escalation minutes;<br/>RP2D memo"]:::proc
    PI["PI: causality and<br/>scientific justification"]:::goal
    SYNC["LLM synchronizes protocol<br/>+ consent + site documents"]:::accent
    EFF --> SAE
    EFF --> PKPD
    EFF --> DSMB
    SAE --> N7
    SAE --> N15
    PKPD --> DSUR
    DSMB --> MIN
    N7 --> PI
    N15 --> PI
    DSUR --> PI
    MIN --> PI
    PI --> SYNC
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef accent fill:#6C757D,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
```

**Caption.** Post-submission authoring once the IND is in effect is driven by
safety data: IND safety reports on the 7-day (fatal or life-threatening suspected
adverse reaction) and 15-day (other serious and unexpected) clocks, the annual
DSUR, amendments, and dose-escalation minutes, with the principal investigator
assessing causality and scientific justification and the LLM synchronizing the
protocol, consent, and site documents so the whole suite stays internally
consistent.

**Role in the IND.** Renders in §9 (Previous Human Experience) and §10 (Additional
Information), showing the maintenance burden the same method carries after
submission.

**Source files.**
`trial-documents/final-paper/publication/sections/sec-03-methods.tex` (Figure 11,
after-trial authoring, adapted in context to post-submission IND maintenance);
`trial-protocol/final-protocol/publication/sections/sec-10-oversight.tex` (the
7 / 15-day clocks and DSMB cadence).
