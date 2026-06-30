## Figure 16. Chemistry, Manufacturing and Control information architecture

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    CMC["Chemistry, Manufacturing<br/>and Control (§7)"]:::goal
    DS["7.1.1 Drug substance<br/>daraxonrasib (RMC-6236)"]:::proc
    DP["7.1.2 Drug product<br/>oral tablet, 160 / 220 / 300 mg"]:::proc
    PL["7.1.3 Placebo product<br/>not applicable (open-label)"]:::ctx
    LB["7.1.4 Labeling<br/>'Caution: investigational'<br/>21 CFR 312.6"]:::proc
    EA["7.2 Environmental<br/>assessment / categorical<br/>exclusion 25.31"]:::proc
    DS1["Characterization,<br/>specification, impurities"]:::input
    DP1["Composition, container-<br/>closure, child-resistant"]:::input
    ST["Stability program<br/>(supports shelf life)"]:::input
    CMC --> DS --> DS1
    CMC --> DP --> DP1
    CMC --> PL
    CMC --> LB
    CMC --> EA
    DP --> ST
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef ctx fill:#F5F5F5,stroke:#6C757D,stroke-width:1px,color:#000000
```

**Caption.** The Chemistry, Manufacturing and Control information architecture for
§7. The drug substance (daraxonrasib, RMC-6236) is described by characterization,
specification, and impurity controls; the drug product is the oral tablet at the
160, 220, and 300 mg strengths with its composition and child-resistant,
light-protective container-closure under 21 CFR §312.6, supported by a stability
program; the placebo product is not applicable for this open-label study; labeling
carries the investigational caution statement; and an environmental assessment or
categorical exclusion under 21 CFR §25.31 is provided.

**Role in the IND.** Renders in §7 (Chemistry, Manufacturing and Control
Information), §7.1.1 to §7.2.

**Source files.**
`trial-ind/inputs/ReGARDD_IND_Template.docx` (the §7 CMC content map);
`trial-protocol/final-protocol/publication/sections/sec-06-intervention.tex`
(formulation and packaging).
