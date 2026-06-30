## Figure 7. Pre-submission IND authoring workflow

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart LR
    PI["PI clinical rationale<br/>(KRAS PDAC, unmet need)"]:::input
    BS["Biostatistician<br/>3+3 rule, Clopper-Pearson"]:::input
    TM["ReGARDD IND template<br/>+ 21 CFR part 312"]:::input
    LLM["Repository LLM<br/>+ medical-writer review<br/>(eDMS)"]:::proc
    M1["IND modules 1-11<br/>(1571 to Relevant Info)"]:::accent
    M2["Protocol + Investigator's<br/>Brochure"]:::accent
    M3["ICF (6th-8th grade)<br/>+ Physical AI opt-out"]:::accent
    CLK["IRB / FDA submission<br/>30-day FDA clock starts"]:::goal
    PI --> LLM
    BS --> LLM
    TM --> LLM
    LLM --> M1
    LLM --> M2
    LLM --> M3
    M1 --> CLK
    M2 --> CLK
    M3 --> CLK
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef accent fill:#6C757D,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
```

**Caption.** Pre-submission authoring. The principal investigator's clinical
rationale, the biostatistician's 3+3 and exact-confidence-interval rules, and the
ReGARDD IND template under 21 CFR part 312 are compiled by the repository LLM under
medical-writer review into the eleven IND modules, the protocol and Investigator's
Brochure, and the informed consent form with the Physical AI opt-out, which start
the 30-day FDA clock on submission.

**Role in the IND.** Renders in §6.1 (Study Protocol) and the Introduction,
documenting how the submission was authored.

**Source files.**
`trial-documents/final-paper/publication/sections/sec-03-methods.tex` (Figure 9,
pre-trial authoring, adapted in context);
`trial-documents/inputs/llm-adoption/main.tex` (the PI large-document guidance);
`trial-protocol/final-protocol/publication/sections/sec-09-statistics.tex` (the
3+3 and Clopper-Pearson rules).
