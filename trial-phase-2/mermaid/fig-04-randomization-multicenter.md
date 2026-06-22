## Figure 4. Randomization and multicenter design

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    ELIG["Eligible participant<br/>KRAS G12 PDAC"]:::light
    RAND["Central permuted-block<br/>randomization 1:1"]:::mid
    subgraph STRATA["Stratification factors"]
        S1["Resectability<br/>resectable vs borderline"]:::light
        S2["KRAS allele<br/>G12D vs other G12"]:::light
        S3["Neoadjuvant therapy<br/>yes vs no"]:::light
        S4["Site (8 centers)"]:::light
    end
    ARMA["Arm A (n &approx; 110)<br/>daraxonrasib RP2D +<br/>LLM robotic Whipple"]:::goal
    ARMB["Arm B (n &approx; 110)<br/>mFOLFIRINOX +<br/>standard Whipple"]:::mid
    SITES["8 academic HPB sites<br/>harmonized fleet"]:::dark
    BICR["BICR RECIST 1.1 +<br/>central masked pathology"]:::mid
    ADJ["Blinded endpoint<br/>adjudication"]:::goal
    ELIG --> RAND
    STRATA -.-> RAND
    RAND --> ARMA
    RAND --> ARMB
    ARMA --> SITES
    ARMB --> SITES
    SITES --> BICR --> ADJ
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Randomization and multicenter design. Eligible participants undergo central permuted-block randomization 1:1, stratified by resectability, KRAS allele, neoadjuvant therapy, and site, to Arm A (daraxonrasib at the RP2D plus the LLM-directed robotic Whipple) or Arm B (modified FOLFIRINOX plus standard high-volume Whipple). Both arms are delivered across the eight academic HPB sites operating as a harmonized fleet, and all outcomes flow through blinded independent central review and central masked pathology into blinded endpoint adjudication.

**Role in the protocol.** Renders the &sect;4.1 Overall Design and &sect;6.5 measures-to-minimize-bias; the four strata and central mechanism control allocation bias.

**Source files.** `sections/sec-04-design.tex` (stratified randomization, BICR, adjudication); `sections/sec-06-intervention.tex` (central randomization, masked pathology).
