## Figure 22. Previous human experience and the evidence base

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    EVID["Evidence base for the IND<br/>(§3.2, §9)"]:::goal
    subgraph CLIN["Clinical (daraxonrasib)"]
      direction TB
      R302["RASolute 302<br/>Phase 3, pretreated<br/>metastatic PDAC"]:::proc
      R301["RASolve 301<br/>first-line program"]:::proc
      BTD["Breakthrough Therapy<br/>(June 2025)"]:::proc
    end
    subgraph COMP["Computational (author 2025-2026)"]
      direction TB
      C1["60-sec PDAC Whipple +<br/>daraxonrasib simulation"]:::input
      C2["AI digital-twin PDAC sim"]:::input
      C3["QSP metastatic-PDAC sim<br/>(VVUQ)"]:::input
      C4["100,000-patient in silico<br/>Phase 3 (triplicate)"]:::input
      C5["End-to-end PDAC<br/>digital-twin proposals"]:::input
    end
    NEW["First-in-human in the<br/>perioperative, curative-intent,<br/>device-directed context"]:::dark
    R302 --> EVID
    R301 --> EVID
    BTD --> EVID
    C1 --> EVID
    C2 --> EVID
    C3 --> EVID
    C4 --> EVID
    C5 --> EVID
    EVID --> NEW
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.2px,color:#FFFFFF
```

**Caption.** The previous human experience and evidence base. Clinically,
daraxonrasib is supported by the RASolute 302 Phase 3 program in previously treated
metastatic PDAC, the RASolve 301 first-line program, and its June 2025 Breakthrough
Therapy designation; computationally, the author's 2025-2026 works supply the
simulation scaffolding (the 60-second PDAC Whipple and daraxonrasib simulation, the
AI digital-twin PDAC simulation, the quantitative-systems-pharmacology metastatic
PDAC simulation with verification, validation and uncertainty quantification, the
100,000-patient in silico Phase 3 triplicate, and the end-to-end PDAC digital-twin
proposals). Daraxonrasib remains first-in-human in this perioperative,
curative-intent, device-directed surgical context.

**Role in the IND.** Renders in §3.2 (Summary of Previous Human Experience) and §9
(Previous Human Experience), §9.1 to §9.3.

**Source files.**
`trial-protocol/final-protocol/publication/sections/sec-02-introduction.tex` (the
clinical and computational evidence);
`trial-ind/inputs/references.bib` (`pdac060s2030`, `fdadigtwinpc`, `qspmetpancre`,
`chatgpt100kp`, `pdacdigtwinp`).
