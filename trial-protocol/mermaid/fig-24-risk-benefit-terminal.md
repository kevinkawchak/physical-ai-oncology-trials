## Figure 24. Risk-benefit framework for advanced and terminally ill PDAC patients

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    BASE["Advanced PDAC baseline<br/>5-year overall survival < 13%<br/>3rd US / 4th EU cancer death"]:::dark
    subgraph RISK["Risks of the combination"]
      R1["Novel device malfunction"]:::warn
      R2["AI advisory error"]:::warn
      R3["Cyber-physical failure"]:::warn
    end
    subgraph BEN["Benefits for this population"]
      B1["Precise in-window R0 resection"]:::mid
      B2["Hard safety limits + audit trail"]:::mid
      B3["Optimized drug-restart timing"]:::mid
    end
    WEIGH{"Benefit-risk for low-survival,<br/>limited-alternative patients"}:::warn
    JUST["Favorable: benefits likely outweigh<br/>risks; expanded-access ethic applies"]:::goal
    BASE --> WEIGH
    R1 & R2 & R3 --> WEIGH
    B1 & B2 & B3 --> WEIGH
    WEIGH --> JUST
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
```

**Caption.** Against a baseline of under-13-percent five-year survival and few
alternatives, the novel risks (device malfunction, advisory error, cyber-physical
failure) are weighed against the benefits (precise in-window R0 resection, layered
hard safety limits with a full audit trail, optimized drug timing). For this
low-survival, limited-alternative population the benefit-risk balance is
favorable, consistent with the expanded-access ethic.

**Role in the protocol.** Structures the &sect;2.3.3 Assessment of Potential Risks
and Benefits.

**Source files.** `inputs/2030-pdac-1min-final-paper/sections/introduction.tex`
(PDAC epidemiology, < 13% 5-year survival); `research/ChatGPT-5.5-Thinking-Extended-19Jun26.md`
(expanded access / compassionate use); `inputs/21cfr312_adapt/02_ind_content_phases.tex` (&sect;312.84 risk-benefit).
