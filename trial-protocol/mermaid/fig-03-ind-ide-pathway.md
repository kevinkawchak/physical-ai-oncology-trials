## Figure 3. Combined IND / IDE regulatory pathway with the Physical AI overlay

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    subgraph DRUG["Drug arm - IND, 21 CFR part 312"]
      PRE["Pre-IND meeting<br/>&sect;312.47 + Physical AI review"]:::light
      IND["IND in effect (30-day)<br/>daraxonrasib Phase 1, 3+3"]:::mid
    end
    subgraph DEV["Device arm - IDE, 21 CFR part 812"]
      SR["Significant-risk determination<br/>&sect;812.3(m)"]:::light
      IDE["IDE approved (FDA + IRB)<br/>early feasibility study"]:::mid
    end
    subgraph PAI["Physical AI overlay - Subpart J (&sect;312.400-405)"]
      P0["Phase 0 simulation validation<br/>&ge;2 frameworks, <2mm / <0.5N"]:::mid
      USL["USL readiness &ge; 7.0<br/>pre-procedure safety matrix"]:::mid
      CLASS["Class II collaborative<br/>continuous oversight, &le;500 ms e-stop"]:::goal
    end
    ENROLL["First-in-human enrollment<br/>combined IND + IDE protocol"]:::goal
    PRE --> IND
    SR --> IDE
    IND --> P0
    IDE --> P0
    P0 --> USL --> CLASS --> ENROLL
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
```

**Caption.** The daraxonrasib drug arm (IND, part 312) and the robotic Whipple
device arm (IDE, part 812, significant-risk) converge through the Physical AI
overlay of Subpart J - Phase 0 simulation validation, USL readiness, and Class II
collaborative classification - into one first-in-human protocol.

**Role in the protocol.** Renders in the Statement of Compliance and &sect;4 Study
Design; establishes the dual regulatory spine.

**Source files.** `inputs/21cfr312_adapt/{01_preamble_scope_definitions,02_ind_content_phases,05_clinical_holds_appendices_closing}.tex`
(Subpart J, USL thresholds, Phase 0); `research/ChatGPT-5.5-Thinking-Extended-19Jun26.md`
(significant-risk, early feasibility).
