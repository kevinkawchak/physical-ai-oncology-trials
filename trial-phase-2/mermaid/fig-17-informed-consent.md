## Figure 17. Informed-consent process with Physical AI opt-out

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    INTRO["Investigator presents study<br/>disease, randomization, prognosis,<br/>alternatives"]:::light
    PAI["Explain Physical AI system<br/>role, continuous oversight,<br/>system-specific risks (&sect;312.60(f))"]:::mid
    OPT{"Participant choice"}:::warn
    YES["Consents to Physical AI<br/>signed ICF + assent if applicable"]:::goal
    NO["Requests non-Physical AI<br/>administration if clinically feasible"]:::mid
    ENR["Randomize and document<br/>consent in source + CRF"]:::goal
    WD["Right to withdraw at<br/>any time without penalty"]:::light
    INTRO --> PAI --> OPT
    OPT -->|accept| YES
    OPT -->|decline AI| NO
    YES --> ENR
    NO --> ENR
    ENR -.-> WD
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Informed-consent process with the Physical AI opt-out. Consent discloses the randomized allocation, the open-label nature, the risks of each regimen, and the Physical AI system's role, continuous human oversight, and system-specific risks, and offers a participant randomized to Arm A a documented right to request non-Physical AI administration where clinically feasible under 21 CFR &sect;312.60(f), with the standard right to withdraw at any time without penalty. The election or declination is recorded in the source document and the case report form before randomization.

**Role in the protocol.** Renders the &sect;10.1 Informed Consent Process and the documented Physical AI opt-out.

**Source files.** `sections/sec-10-oversight.tex` (consent process, Physical AI opt-out &sect;312.60(f), randomize and document, withdrawal right).
