## Figure 4. The six greatest-acceleration document targets

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
  IN["Faster, validated authoring<br/>prospective drafting, controlled reuse<br/>tables from validated data"]:::input

  T1["1. Initial IND and IRB package<br/>protocol, IB, nonclinical, CMC<br/>Hard gate: starts 30-day clock"]:::goal
  T2["2. Protocol amendments<br/>synchronized consent/site updates<br/>Hard gate: avoids enrollment pauses"]:::goal
  T3["3. Cohort-review packages<br/>after safety data mature<br/>DLT, AE, PK/PD tables"]:::accent
  T4["4. Complete clinical-hold response<br/>addresses every deficiency<br/>Hard gate: 30-day FDA review"]:::goal
  T5["5. Phase 2-to-3 briefing package<br/>EOP2 book and Phase 3 protocol<br/>dose-response, endpoints, estimands"]:::accent
  T6["6. Pivotal CSR and NDA/BLA modules<br/>after database lock<br/>CTD Modules 2 and 5; RTOR staging"]:::accent

  OUT["Compressed administrative/prep time<br/>months to a year cumulatively<br/>lab bench to patient"]:::goal

  NOTE["Schedule value note<br/>highest when document is<br/>on the critical path"]:::ctx
  LIMIT["Smaller benefit when waiting on<br/>clinical events, safety follow-up,<br/>CMC/stability, external review clock"]:::warn

  IN --> T1
  IN --> T2
  IN --> T3
  IN --> T4
  IN --> T5
  IN --> T6

  T1 -- "first-in-human start" --> OUT
  T2 -- "new arm online sooner" --> OUT
  T3 -- "shorter cohort pause" --> OUT
  T4 -- "earlier potential restart" --> OUT
  T5 -- "earlier meeting request" --> OUT
  T6 -- "lock-to-filing shrinks" --> OUT

  NOTE -. "ranks targets 1-6" .-> IN
  OUT -. "gains capped by" .-> LIMIT
  LIMIT -. "poor docs add cycles" .-> T2

  classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
  classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
  classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
  classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
  classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
  classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** This ranked ladder lists the six document targets where faster, validated authoring yields the greatest schedule value, in priority order from the initial IND and IRB package through the pivotal CSR and NDA/BLA modules after database lock. A single input node, faster validated authoring, feeds all six targets, and each target contributes to a common goal of compressed administrative and prep time, which can cumulatively save months to a year between the laboratory bench and the patient. A looping note marks that value is highest when the document sits on the critical path, while a gray limit node captures the constraints (clinical events, safety follow-up, CMC and stability data, external review clocks) that cap the gains and can add cycles when documents are inconsistent.

**Role in the paper.** It appears in the Results/Discussion as the prioritization of acceleration targets and becomes a TikZ mermaidfig in the draft, full, and final LaTeX stages.

**Source files.**
- trial-documents/research/document-types/ChatGPT-5-5-Thinking-Extended-DocTypes-2026-06-26.md (ACCELERATION list: "Where faster document production has the greatest schedule value")
- trial-documents/research/document-types/Gemini-3-1-Pro-DocTypes-2026-06-26.md (Administrative/Prep Time compression framing)
