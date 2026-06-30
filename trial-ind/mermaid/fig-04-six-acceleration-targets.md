## Figure 4. The six greatest-acceleration IND document targets

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart LR
    IN["Faster, validated<br/>IND authoring"]:::input
    T1["1. Initial IND + IRB<br/>package (this build)"]:::goal
    T2["2. Protocol amendments<br/>+ synchronized consent"]:::goal
    T3["3. Cohort-review package<br/>after DLT data mature"]:::proc
    T4["4. Complete clinical-hold<br/>response"]:::goal
    T5["5. EOP2 briefing +<br/>Phase 3 protocol"]:::accent
    T6["6. CSR + NDA/BLA<br/>modules at lock"]:::accent
    OUT["Compressed admin /<br/>prep time<br/>(months to a year)"]:::goal
    LIM["Capped by clinical events<br/>and fixed review clocks"]:::dec
    IN --> T1
    IN --> T2
    IN --> T3
    IN --> T4
    IN --> T5
    IN --> T6
    T1 --> OUT
    T2 --> OUT
    T3 --> OUT
    T4 --> OUT
    T5 --> OUT
    T6 --> OUT
    OUT -.-> LIM
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef accent fill:#6C757D,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef dec fill:#D9D9D9,stroke:#000000,stroke-width:1px,color:#000000
```

**Caption.** The six greatest-acceleration document targets, ranked, with the
**initial IND and IRB package the highest-value target and the subject of this
build**. One input, faster validated authoring, feeds all six; each contributes to
a common compression of administrative and preparation time of months to a year.
The schedule value is highest on the critical path, and the gains are capped by
clinical events and fixed review clocks (they cannot shorten the 30-day FDA review
or manufacture missing data).

**Role in the IND.** Renders in the Introduction (§3.1) and the General
Investigational Plan (§4.1 Rationale), situating the IND as target one and framing
the acceleration claim.

**Source files.**
`trial-documents/final-paper/publication/sections/sec-04-results.tex` (Figure 16,
the six targets, adapted in context to the IND);
`trial-documents/final-paper/publication/sections/sec-05-discussion.tex`
(critical-path and cap argument).
