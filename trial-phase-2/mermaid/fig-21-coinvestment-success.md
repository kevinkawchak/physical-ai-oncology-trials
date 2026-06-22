## Figure 21. Co-investment-to-success-likelihood pathway

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart LR
    CAP["Aligned capital and<br/>PACIF tranches"]:::light
    FW{"Capital firewall"}:::warn
    L1["8 academic HPB sites"]:::mid
    L2["Phase 0 compute<br/>(&ge;5000 sims)"]:::mid
    L3["Patient Access and<br/>Equity Fund"]:::mid
    L4["Central review core<br/>(BICR, ctDNA)"]:::mid
    O1["Higher statistical power"]:::goal
    O2["Lower sim-to-real gap<br/>(&lt;1.5 mm, &lt;0.4 N)"]:::goal
    O3["Lower dropout, higher<br/>retention, equitable enrollment"]:::goal
    O4["Definitive, generalizable,<br/>equitable answer"]:::goal
    BLOCK["Endpoints, adjudication,<br/>analysis: no funder path"]:::dark
    CAP --> FW
    FW --> L1
    FW --> L2
    FW --> L3
    FW --> L4
    L1 --> O1
    L2 --> O2
    L3 --> O3
    L4 --> O4
    FW -.->|blocked| BLOCK
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** The co-investment-to-success-likelihood pathway. Aligned capital and the PACIF flow only through the capital firewall into operational levers (eight sites, Phase 0 compute, the Patient Access and Equity Fund, and the central review and biomarker core), which raise statistical power, lower the sim-to-real gap (below 1.5 mm and 0.4 N), lower dropout and raise retention and equitable enrollment, and so raise the probability of a definitive, generalizable, equitable answer. The dashed blocked edge shows that the firewall admits no path from funders to the endpoints, their adjudication, or the analysis: capital raises operational success likelihood while the integrity of the result is structurally protected.

**Role in the protocol.** Renders the &sect;2.3 co-investment logic; shows how capital becomes power, fidelity, retention, equity, and generalizability without buying the answer.

**Source files.** `sections/sec-02-introduction.tex` (co-investment tranches, six operational levers, firewall, success-likelihood outcomes).
