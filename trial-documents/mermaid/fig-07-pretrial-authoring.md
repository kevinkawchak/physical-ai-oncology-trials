## Figure 7. Pre-trial document authoring by PI and medical writers

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    PI["PI clinical rationale<br/>background and benefit-risk"]:::input
    BIO["Biostatistician<br/>dose-escalation rules<br/>cohort sizes 3+3 or Bayesian"]:::input
    PHARM["Pharmacologist data<br/>starting dose, PK/PD<br/>nonclinical toxicology"]:::input
    TMPL["TransCelerate Common<br/>Protocol Template<br/>plus ICH M11 structure"]:::input

    EDMS["Medical writers in eDMS<br/>Veeva Vault, version control<br/>controlled collaborative authoring"]:::proc

    PI --> EDMS
    BIO --> EDMS
    PHARM --> EDMS
    TMPL --> EDMS

    IND["IND application<br/>nonclinical, CMC,<br/>clinical rationale dossier"]:::proc
    PROT["Clinical Trial Protocol<br/>escalation, DLT window,<br/>RP2D, sample size"]:::proc
    IB["Investigator's Brochure<br/>cumulative clinical and<br/>nonclinical safety data"]:::proc
    ICF["Informed Consent Form<br/>6th-8th grade<br/>reading level"]:::proc

    EDMS --> IND
    EDMS --> PROT
    EDMS --> IB
    EDMS --> ICF

    PROT -.->|terminology<br/>consistency QC| ICF
    IB -.->|risk language<br/>feeds consent| ICF

    LLM["Repository LLM<br/>compresses authoring time<br/>draft, summarize, reconcile"]:::accent
    LLM -.->|accelerates<br/>drafting| EDMS
    LLM -.->|consistency<br/>checks| IND
    LLM -.->|consistency<br/>checks| PROT
    LLM -.->|consistency<br/>checks| IB
    LLM -.->|plain-language<br/>translation| ICF

    GATE{"IRB/IEC and FDA<br/>review and approval"}:::warn
    IND --> GATE
    PROT --> GATE
    IB --> GATE
    ICF --> GATE

    WAIT["Fixed 30-day FDA<br/>IND waiting period"]:::accent
    GATE -->|IND submitted| WAIT

    DOSE["First-in-human dosing<br/>cohort 1 enrollment"]:::goal
    WAIT -->|no clinical hold| DOSE

    GATE -.->|deficiencies cause<br/>revision loop| EDMS

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** This flowchart traces pre-trial document authoring for a Phase 1 pancreatic cancer study, from controlled inputs (PI clinical rationale, biostatistician dose-escalation rules and cohort sizes, pharmacologist data, and the TransCelerate Common Protocol Template) through medical writers working in a version-controlled eDMS such as Veeva Vault. The writers compile the IND application, Clinical Trial Protocol, Investigator's Brochure, and the Informed Consent Form translated to a 6th-8th grade reading level, with dashed edges showing terminology and risk-language consistency checks across documents. After IRB/IEC and FDA review, the IND triggers the fixed 30-day FDA waiting period that must elapse before first-in-human dosing, and deficiencies feed a revision loop back into the eDMS. A terracotta accent node marks where the repository LLM compresses authoring time by drafting, summarizing, reconciling terminology, and supporting plain-language translation.

**Role in the paper.** This figure appears in the Methods section to establish the regulatory baseline against which the repository LLM efficiency gains are measured, and it becomes a TikZ mermaidfig in the draft, full, and final LaTeX stages.

**Source files.** research/industry-workflow/Gemini-3-1-Pro-Workflow-2026-06-26.md (section B); research/industry-workflow/ChatGPT-5-5-Thinking-Extended-Workflow-2026-06-26.md
