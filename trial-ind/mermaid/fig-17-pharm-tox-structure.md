## Figure 17. Pharmacology and toxicology integrated-summary structure

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    PT["Pharmacology and<br/>Toxicology (§8)"]:::goal
    PH["8.1.1 Pharmacology<br/>summary and conclusions"]:::proc
    TX["8.1.2 Toxicology:<br/>integrated summary"]:::proc
    FT["8.1.3 Toxicology:<br/>full data tabulation"]:::proc
    PD["Mechanism, distribution,<br/>RAS(ON) pharmacodynamics"]:::input
    DSIM["Digital-twin and QSP<br/>simulation evidence"]:::input
    SWEEP["32-iteration perioperative<br/>PK sweep (0.45 ng/mL baseline)"]:::input
    SAFEM["Safety margins; DL1 160 mg<br/>conservative first-in-human"]:::input
    PT --> PH --> PD
    PT --> TX --> SAFEM
    PT --> FT --> SWEEP
    PH --> DSIM
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
```

**Caption.** The pharmacology and toxicology integrated-summary structure for §8.
The pharmacology summary covers the RAS(ON) mechanism, drug distribution, and
pharmacodynamics; the integrated toxicology summary states the safety margins
underpinning the conservative DL1 160 mg first-in-human starting dose; and the full
data tabulation tabulates the supporting studies, here grounded in the author's
digital-twin and quantitative-systems-pharmacology simulation evidence and the
32-iteration perioperative pharmacokinetic sweep (0.45 ng/mL baseline serum trough
at the operative timepoint across all iterations).

**Role in the IND.** Renders in §8 (Pharmacology and Toxicology Information),
§8.1.1 to §8.1.3, and §3.4 (Overview of Preclinical Data).

**Source files.**
`trial-protocol/final-protocol/publication/sections/sec-06-intervention.tex`
(perioperative PK sweep);
`trial-ind/inputs/references.bib` (`qspmetpancre`, `fdadigtwinpc`, `pdacdigtwinp`,
`chatgpt100kp`).
