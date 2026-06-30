## Figure 19. Safety-reporting clocks and the six Physical AI reporting triggers

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart LR
    EV["Adverse event<br/>(clinical or Physical AI)"]:::input
    C7["7 calendar days<br/>fatal / life-threatening<br/>suspected adverse reaction"]:::goal
    C15["15 calendar days<br/>other serious + unexpected<br/>suspected adverse reaction"]:::goal
    subgraph PAI["Six Physical AI triggers (§312.32(g))"]
      direction TB
      G1["1 Serious Physical AI AE (7 / 15d)"]:::proc
      G2["2 System-drug interaction (15d)"]:::proc
      G3["3 Cybersecurity incident (15d; 7d if harm)"]:::proc
      G4["4 Model degradation (>= 3 procedures / 24h)"]:::proc
      G5["5 Sim-to-real divergence > 2x (>4 mm / >1.0 N)"]:::proc
      G6["6 Digital-twin discrepancy"]:::proc
    end
    AUD["Hash-chained audit trail<br/>-24h to +72h (21 CFR part 11)"]:::dark
    EV --> C7
    EV --> C15
    EV --> PAI
    PAI --> AUD
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.2px,color:#FFFFFF
```

**Caption.** The safety-reporting clocks and the six Physical AI reporting
triggers. A fatal or life-threatening suspected adverse reaction is reported within
7 calendar days and any other serious and unexpected suspected adverse reaction
within 15 calendar days (21 CFR §312.32). The six Physical AI triggers of §312.32(g)
are a serious Physical AI adverse event (7 or 15-day), a system-drug interaction
(15-day), a cybersecurity incident (15-day, or 7-day if patient harm is possible),
sustained model degradation (at least 3 consecutive procedures or 24 cumulative
hours), a sim-to-real divergence exceeding twice the validated tolerance (trajectory
over 4 mm or force over 1.0 N), and a digital-twin discrepancy. Each preserves the
hash-chained audit trail from minus 24 to plus 72 hours under 21 CFR part 11.

**Role in the IND.** Renders in §9 (Previous Human Experience) and the safety
narrative of §6.1 (Study Protocol).

**Source files.**
`trial-protocol/final-protocol/publication/sections/sec-08-assessments.tex` (the
7 / 15-day clocks and the six §312.32(g) triggers);
`trial-protocol/final-protocol/publication/sections/sec-10-oversight.tex` (audit
trail).
