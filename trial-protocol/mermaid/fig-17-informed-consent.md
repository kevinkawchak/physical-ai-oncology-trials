## Figure 17. Informed-consent process with the Physical AI opt-out

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    INTRO["Investigator presents study<br/>disease, alternatives, prognosis"]:::light
    PAI["Explain the Physical AI system<br/>role, oversight, specific risks<br/>(&sect;312.60(f) / &sect;312.23(a)(1)(iv))"]:::mid
    OPT{"Participant choice"}:::warn
    YES["Consents to Physical AI<br/>signed ICF + assent if applicable"]:::goal
    NO["Requests non-Physical AI<br/>administration if clinically feasible"]:::mid
    ENR["Enroll and document<br/>consent in source + CRF"]:::goal
    WD["Right to withdraw at any time<br/>without penalty"]:::light
    INTRO --> PAI --> OPT
    OPT -->|accept| YES --> ENR
    OPT -->|decline AI| NO --> ENR
    ENR -.-> WD
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** Consent explicitly discloses the Physical AI system's role,
continuous human oversight, and system-specific risks, and offers the
participant a documented right to request non-Physical AI administration where
clinically feasible (21 CFR §312.60(f)), with the standard right to withdraw at
any time without penalty.

**Role in the protocol.** Defines &sect;10.1.1 the informed-consent process and the
Physical AI opt-out unique to this trial class.

**Source files.** `inputs/21cfr312_adapt/05_clinical_holds_appendices_closing.tex`
(&sect;312.60(f) Physical AI consent and opt-out);
`nih-protocol/08_supporting_documentation_regulatory_oversight.md` (consent process).
