## Figure 14. Daraxonrasib mechanism and pharmacological class

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    DRX["Daraxonrasib (RMC-6236)<br/>oral RAS(ON) multi-selective<br/>(pan-RAS) inhibitor"]:::goal
    CYPA["Tri-complex with<br/>cyclophilin A"]:::proc
    RASON["Binds active GTP-bound<br/>mutant RAS (RAS(ON))"]:::proc
    KRAS["KRAS G12 variants<br/>G12D / G12V / G12R"]:::input
    EFF["Suppresses downstream<br/>MAPK / PI3K effector<br/>signaling"]:::accent
    PROL["Reduced tumor-cell<br/>proliferation and survival"]:::goal
    ROUTE["Route: oral, once daily<br/>DL1 160 / DL2 220 / DL3 300 mg"]:::ctx
    BTD["FDA Breakthrough Therapy<br/>(June 2025), pretreated<br/>metastatic KRAS G12 PDAC"]:::dark
    DRX --> CYPA --> RASON
    KRAS --> RASON
    RASON --> EFF --> PROL
    DRX -.-> ROUTE
    DRX -.-> BTD
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef accent fill:#6C757D,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef ctx fill:#F5F5F5,stroke:#6C757D,stroke-width:1px,color:#000000
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.2px,color:#FFFFFF
```

**Caption.** Daraxonrasib (RMC-6236) is an oral RAS(ON) multi-selective (pan-RAS)
small-molecule inhibitor. It forms a tri-complex with cyclophilin A and binds the
active, guanosine-triphosphate-bound state of mutant RAS, including the KRAS G12
variants (G12D, G12V, G12R) that define the trial population, suppressing
downstream MAPK and PI3K effector signaling and reducing tumor-cell proliferation
and survival. It is dosed orally once daily at DL1 160 mg, DL2 220 mg, and DL3 300
mg, and holds FDA Breakthrough Therapy designation (June 2025) for previously
treated metastatic KRAS G12 PDAC.

**Role in the IND.** Renders in §3.1.1 (Name of Drug and Active Ingredients),
§3.1.2 (Pharmacological Class), and §3.1.3 (Structural Formula).

**Source files.**
`trial-protocol/final-protocol/publication/sections/sec-06-intervention.tex`
(mechanism, class, and dosing);
`trial-ind/inputs/references.bib` (`daraxwhipple`, `onpremwhippl`).
