## Figure 19. Four counterfactual scenarios

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart LR
    A0["Scenario A<br/>Borderline PDAC,<br/>SMV abutment 75 deg"]:::light
    AW["Human-only delay:<br/>unresectable, R0 window<br/>closes, shorter PFS/OS"]:::dark
    AC["LLM + robot:<br/>in-window R0 resection"]:::goal
    B0["Scenario B<br/>Intra-op SMV/PV plane"]:::light
    BW["Manual injury: hemorrhage,<br/>conversion, R1/R2,<br/>fistula-sepsis cascade"]:::dark
    BC["No-fly gate, &le;3 ms e-stop:<br/>injury averted"]:::goal
    C0["Scenario C<br/>Perioperative daraxonrasib"]:::light
    CW["Manual mistiming:<br/>dehiscence or<br/>micrometastatic outgrowth"]:::dark
    CC["Advisory T+7/14/21:<br/>optimally timed restart"]:::goal
    D0["Scenario D<br/>Funding the<br/>definitive answer"]:::light
    DW["Under-funded, single-center,<br/>under-powered: no<br/>definitive equitable answer"]:::dark
    DC["Co-invested behind firewall:<br/>power, retention, equity"]:::goal
    A0 -->|human only| AW
    A0 -->|combination| AC
    B0 -->|human only| BW
    B0 -->|combination| BC
    C0 -->|human only| CW
    C0 -->|combination| CC
    D0 -->|under-funded| DW
    D0 -->|co-invested| DC
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** The four counterfactual scenarios in which withholding the integrated approach worsens the patient or the evidence. Scenario A: a human-only scheduling delay lets a borderline-resectable tumor abutting the superior mesenteric vein progress and closes the R0 window, while the LLM-directed robot achieves an in-window R0 resection. Scenario B: an unrecognized superior-mesenteric or portal-vein injury during manual surgery triggers the fistula-sepsis cascade, while the no-fly gate with a &le;3 ms emergency stop averts it. Scenario C: manual mistiming of the daraxonrasib restart causes dehiscence or micrometastatic outgrowth, while the timed advisory at T+7, T+14, or T+21 prevents it. Scenario D: an under-funded, single-center, under-powered study yields no definitive equitable answer, while patient-aligned co-investment walled off by the capital firewall converts capital into accrual speed, retention, fidelity, and statistical power.

**Role in the protocol.** Renders the &sect;2.1 Study Rationale counterfactuals; motivates the combination and the co-investment model.

**Source files.** `sections/sec-02-introduction.tex` (four counterfactual scenarios A-D, SMV abutment, no-fly gate, advisory timing, capital firewall).
