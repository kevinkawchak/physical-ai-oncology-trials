## Figure 5. Document decision-gate taxonomy

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    Q{{"Is the document on<br/>the critical path?"}}:::warn

    Q -->|"Yes: blocks the<br/>next trial step"| TYPE["Classify the gate type<br/>3 mechanisms"]:::proc
    Q -.->|"No: off-path record<br/>(TMF, annual reports,<br/>abstracts, full CSR)"| OFF["Off-path documents<br/>important but not gating"]:::ctx

    TYPE --> HARD["Hard gate<br/>trial legally or ethically<br/>cannot proceed"]:::goal
    TYPE --> PROTO["Protocol-defined gate<br/>trial's own rules block<br/>next cohort or arm"]:::goal
    TYPE --> DEC["Decision gate<br/>legally possible, sponsor<br/>will not invest first"]:::goal

    HARD --> HDOC["Examples:<br/>Initial IND dossier<br/>IRB approval package<br/>Clinical-hold response<br/>Material amendments"]:::input
    PROTO --> PDOC["Examples:<br/>Cohort-review packages<br/>(DLT, AE, PK/PD tables)<br/>Interim-analysis SAP<br/>DMC/DSMB charter"]:::input
    DEC --> DDOC["Examples:<br/>EOP2 briefing package<br/>Go / no-go after<br/>database lock and SAP"]:::input

    HDOC -->|"30-day FDA review<br/>clock starts earlier"| ACC["Critical-path acceleration<br/>faster authoring moves<br/>start or restart forward"]:::accent
    PDOC -->|"shorten pause after<br/>observation window"| ACC
    DDOC -->|"request meeting / decide<br/>sooner after lock"| ACC

    ACC -.->|"author prospectively<br/>and in parallel"| Q

    NOTE["Note: faster creation speeds the trial<br/>only on the critical path. It cannot<br/>shorten DLT windows, 30-day reviews,<br/>data cleaning or manufacturing."]:::ctx
    ACC --- NOTE

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** This taxonomy classifies trial documents by how they constrain timelines, beginning from the central question of whether a document lies on the critical path. Three gate mechanisms are distinguished: hard gates where the trial legally or ethically cannot proceed (initial IND, IRB approval, clinical-hold response, material amendments), protocol-defined gates where the trial's own rules block the next cohort or arm (cohort-review packages, interim-analysis SAP and DMC/DSMB charter), and decision gates where progress is legally possible but the sponsor will not invest without a decision (EOP2 briefing, go/no-go after database lock). Each gate maps to example documents and to a critical-path acceleration loop, with 30-day FDA review clocks (IND and clinical-hold response) noted as the regulatory durations that earlier authoring can start but not shorten. The context note records that faster creation speeds the trial only on the critical path and cannot shorten DLT windows, statutory reviews, data cleaning or manufacturing.

**Role in the paper.** It appears in the Methods/Results discussion of where accelerated LLM document generation has schedule value, framing the gate taxonomy that organizes the document-type analysis; it becomes a TikZ mermaidfig in the draft, full and final LaTeX stages.

**Source files.** trial-documents/research/document-types/ChatGPT-5-5-Thinking-Extended-DocTypes-2026-06-26.md (hard, protocol-defined and decision gates; example documents; 30-day IND and clinical-hold review clocks; off-path records).
