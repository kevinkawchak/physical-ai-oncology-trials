## Figure 11. Three timeline buckets and where compression occurs

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    SRC[Gemini DocTypes<br/>Three timeline buckets]:::input

    SRC --> B1
    SRC --> B2
    SRC --> B3

    subgraph BUCKET1 [Bucket 1: Clinical / Operational Time]
        direction TB
        B1[Patient recruitment<br/>and treatment]:::ctx
        B1a[Wait for data to mature<br/>PFS / OS endpoints]:::ctx
        B1g{{NOT compressible<br/>tumors set the pace}}:::warn
        B1 --> B1a --> B1g
    end

    subgraph BUCKET2 [Bucket 2: Administrative / Prep Time]
        direction TB
        B2[Data cleaning<br/>statistical analysis]:::proc
        B2a[Document writing<br/>protocols, CSRs, dossiers]:::proc
        B2g{{COMPRESSIBLE<br/>by LLM authoring}}:::accent
        B2 --> B2a --> B2g
    end

    subgraph BUCKET3 [Bucket 3: Regulatory Review Time]
        direction TB
        B3[FDA / EMA review<br/>of submitted documents]:::ctx
        B3a[30-day IND wait<br/>30-day clinical-hold response]:::ctx
        B3g{{FIXED<br/>external review clock}}:::warn
        B3 --> B3a --> B3g
    end

    LLM[Repository LLM<br/>document generation]:::proc
    LLM -- targets only Bucket 2 --> B2

    B1g -. no acceleration .-> OUT
    B2g == months saved ==> OUT
    B3g -. no acceleration .-> OUT

    OUT[Months to a year<br/>saved cumulatively]:::goal

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** The oncology trial timeline divides into three buckets, only one of which compresses through faster authoring. Bucket 1 (clinical/operational time: recruitment, treatment, and waiting for progression-free or overall survival data to mature) is not compressible, and Bucket 3 (regulatory review time, including the fixed 30-day IND waiting period and the 30-day clinical-hold response review) is fixed by external review clocks. The Repository LLM points only at Bucket 2 (data cleaning, statistical analysis, and document writing), the administrative/prep time that is compressible, yielding months to a year saved cumulatively across the program.

**Role in the paper.** It appears in the Discussion to scope where LLM-accelerated document generation can and cannot shorten timelines, and it becomes a TikZ mermaidfig in the draft, full, and final LaTeX stages.

**Source files.** research/document-types/Gemini-3-1-Pro-DocTypes-2026-06-26.md (three buckets); research/document-types/ChatGPT-5-5-Thinking-Extended-DocTypes-2026-06-26.md
