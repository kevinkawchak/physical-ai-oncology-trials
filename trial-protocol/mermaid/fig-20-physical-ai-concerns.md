## Figure 20. Eight Physical AI concerns and their protocol mitigations

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart LR
    subgraph CONCERN["Concern"]
      C1["Physical AI limitations"]:::warn
      C2["Patient safety is paramount"]:::warn
      C3["Loss of human workers"]:::warn
      C4["Single-source software"]:::warn
      C5["Proprietary-model dependency"]:::warn
      C6["Open-source LLMs not domestic"]:::warn
      C7["Overly complex AI workflows"]:::warn
      C8["LLMs are black boxes"]:::warn
    end
    subgraph MIT["Protocol mitigation"]
      M1["Phase 0 sim validation;<br/>narrow locked device, not generative"]:::mid
      M2["10 kHz heartbeat, <=3 ms E-stop,<br/>force caps, vessel gate"]:::mid
      M3["Human-in-the-loop retained;<br/>surgeon + backup operator + ISM"]:::mid
      M4["Cross-framework (Isaac/MuJoCo/<br/>Gazebo/PyBullet), >=2 required"]:::mid
      M5["On-premises repository LLM,<br/>open weights, vendor-isolated"]:::mid
      M6["Domestic on-prem deployment;<br/>data residency preserved"]:::mid
      M7["Deny-by-default, 5 MCP servers,<br/>simplified gate surface"]:::mid
      M8["Hash-chained audit trail;<br/>21 CFR part 11 traceability"]:::goal
    end
    C1 --> M1
    C2 --> M2
    C3 --> M3
    C4 --> M4
    C5 --> M5
    C6 --> M6
    C7 --> M7
    C8 --> M8
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.4px,color:#FFFFFF
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** Each of the eight Physical AI concerns is paired with a concrete
protocol mitigation: validated narrow device behavior over generative
open-endedness, layered hard safety limits, retained human roles, multi-framework
(non-single-source) validation, an on-premises open-weights LLM isolated from the
vendor stack, domestic deployment with preserved data residency, a deny-by-default
simplified workflow, and a 21 CFR part 11 hash-chained audit trail that makes the
black box traceable.

**Role in the protocol.** Forms the &sect;2.2 Background subsection on Physical AI
concerns and recurs in &sect;10 oversight.

**Source files.** `research/{ChatGPT-5.5-Thinking-Extended-18Jun26,Gemini-3.1-Pro-18Jun26}.md`
(LLM black-box, single-source, proprietary, domestic concerns);
`inputs/21cfr312_adapt/01_preamble_scope_definitions.tex` (4 sim frameworks, MCP, audit trail);
`inputs/author_works.bib` (`Kawchak2025CodeGenerationCompetition16` open vs proprietary).
