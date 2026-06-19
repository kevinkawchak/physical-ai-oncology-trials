## Figure 21. Hash-chained audit trail and deny-by-default data flow

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    AUTHZ["trialmcp-authz<br/>deny-by-default, scoped tokens"]:::goal
    FHIR["trialmcp-fhir<br/>clinical data (de-identified)"]:::mid
    DICOM["trialmcp-dicom<br/>imaging (RECIST)"]:::mid
    LEDGER["trialmcp-ledger<br/>SHA-256 hash chain"]:::goal
    PROV["trialmcp-provenance<br/>commit + seed pinning"]:::mid
    ROBOT["Robot agent workflow<br/>authenticate -> retrieve -> imaging -><br/>execute -> record audit -> record provenance"]:::light
    PART11["21 CFR part 11 record integrity<br/>+ HIPAA Safe Harbor (18 identifiers)"]:::dark
    ROBOT --> AUTHZ
    AUTHZ --> FHIR & DICOM
    FHIR & DICOM --> LEDGER --> PROV
    LEDGER --- PART11
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.4px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
```

**Caption.** The six-step robot-agent workflow accesses clinical and imaging data
only through deny-by-default, scope-limited tokens (trialmcp-authz), with every
action written to a SHA-256 hash-chained ledger and pinned to a commit and seed
(provenance), satisfying 21 CFR part 11 record integrity and HIPAA Safe Harbor
de-identification of all 18 identifiers. This is the auditable answer to the
black-box concern.

**Role in the protocol.** Realizes &sect;10.1.3 confidentiality/privacy and
&sect;10.1.9 data handling, and the &sect;6 MCP integration description.

**Source files.** `inputs/21cfr312_adapt/01_preamble_scope_definitions.tex`
(5 MCP servers, 6-step workflow, hash chain, deny-by-default, HIPAA Safe Harbor);
`research/Gemini-3.1-Pro-19Jun26.md` (tamper-evident audit trail).
