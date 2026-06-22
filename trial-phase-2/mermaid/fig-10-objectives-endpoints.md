## Figure 10. Objectives-to-endpoints hierarchy

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    PRIM["Primary objective<br/>prolong PFS (Arm A vs Arm B)<br/>PFS per RECIST 1.1, BICR sensitivity"]:::goal
    subgraph KEY["Key secondary (fixed-sequence hierarchy, two-sided 0.05)"]
        K1["1. Overall survival (OS)<br/>to 24 months"]:::mid
        K2["2. R0 resection rate<br/>central masked pathology"]:::mid
        K3["3. ISGPS Grade B/C fistula rate"]:::mid
        K4["4. Major pathologic response (MPR)"]:::mid
        K5["5. Week-12 KRAS ctDNA clearance"]:::mid
    end
    SEC["Secondary (estimation)<br/>Clavien-Dindo III+, 90-day mortality,<br/>conversion, operative time, QoL"]:::light
    EXP["Exploratory<br/>advisory-vs-gate concordance, sim-to-real gap,<br/>federated model, ctDNA dynamics, health economics"]:::dark
    PRIM -->|if rejected| K1
    K1 -->|if rejected| K2
    K2 -->|if rejected| K3
    K3 -->|if rejected| K4
    K4 -->|if rejected| K5
    PRIM -.-> SEC
    PRIM -.-> EXP
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Objectives-to-endpoints hierarchy. The primary objective is to prolong progression-free survival in Arm A relative to Arm B (PFS per RECIST 1.1, investigator-assessed with BICR as a sensitivity analysis). The key secondary endpoints are tested only if the primary null is rejected, in a fixed-sequence hierarchy that controls the family-wise error at two-sided 0.05: overall survival, R0 resection rate, ISGPS Grade B/C fistula rate, major pathologic response, and week-12 KRAS ctDNA clearance, with the first non-rejection stopping the confirmatory sequence. Remaining secondary endpoints are estimation targets and exploratory analyses are non-confirmatory.

**Role in the protocol.** Renders the &sect;3 Objectives and Endpoints and the &sect;9.1 hypothesis hierarchy; defines the confirmatory testing order.

**Source files.** `sections/sec-09-statistics.tex` (primary PFS, fixed-sequence hierarchy, secondary and exploratory endpoints); `sections/sec-08-assessments.tex` (endpoint definitions).
