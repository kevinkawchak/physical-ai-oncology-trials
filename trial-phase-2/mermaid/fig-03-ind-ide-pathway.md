## Figure 3. Combined IND / IDE pathway with Subpart J overlay

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    subgraph DRUG["Drug arm (IND, 21 CFR part 312)"]
        PRE["IND in effect<br/>daraxonrasib at RP2D<br/>Phase 2 efficacy"]:::light
        IND["Amended IND &sect;312.21(b)<br/>300 mg PO once daily<br/>no 3+3 escalation"]:::mid
    end
    subgraph DEV["Device arm (IDE, 21 CFR part 812)"]
        SR["Significant-risk IDE<br/>&sect;812.3(m)<br/>randomized controlled"]:::light
        IDE["IDE approved (FDA + IRB)<br/>PancreSpeed II<br/>Class II collaborative"]:::mid
    end
    P0["Phase 0 simulation validation (Subpart J)<br/>&ge;5000 sims, &ge;3 frameworks<br/>trajectory &lt;1.5 mm, tip-force &lt;0.4 N"]:::mid
    USL["USL readiness &ge;8.0<br/>multicenter fleet harmonization<br/>federated audit (21 CFR part 11)"]:::mid
    SIRB["Single IRB (sIRB) across 8 sites<br/>45 CFR 46.114<br/>&le;500 ms e-stop"]:::goal
    ENR["Multicenter randomized enrollment 1:1<br/>full autonomy prohibited &sect;312.21(e)"]:::goal
    PRE --> IND
    SR --> IDE
    IND --> P0
    IDE --> P0
    P0 --> USL --> SIRB --> ENR
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Combined IND / IDE regulatory pathway with the upgraded Physical AI Subpart J overlay for Phase 2. The daraxonrasib drug arm (IND, part 312, at the established RP2D) and the robotic Whipple device arm (IDE, part 812, significant-risk per &sect;812.3(m), randomized controlled) converge through Phase 0 simulation validation across at least three frameworks, USL readiness &ge;8.0 with fleet harmonization, single IRB review across eight sites, and Class II collaborative classification into a single multicenter randomized enrollment pathway.

**Role in the protocol.** Renders the &sect;0 Statement of Compliance pathway; orients the drug, device, and overlay spines that govern the trial.

**Source files.** `sections/sec-00-compliance.tex` (IND/IDE/Subpart J, single IRB, USL &ge;8.0); `sections/sec-04-design.tex` (combined submission, Phase 0 gate).
