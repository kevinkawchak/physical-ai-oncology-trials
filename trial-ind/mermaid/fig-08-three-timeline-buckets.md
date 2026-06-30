## Figure 8. The three timeline buckets, with exact time saved per IND document

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    B1["Clinical / operational<br/>NOT compressible"]:::dec
    B2["Administrative / preparation<br/>COMPRESSIBLE"]:::accent
    B3["Regulatory review<br/>FIXED"]:::dec
    B1D["Recruitment, surgery,<br/>DLT window, PFS / OS<br/>maturation"]:::ctx
    B3D["30-day IND review,<br/>30-day clinical-hold review"]:::ctx
    LLM["Repository LLM acts on<br/>this bucket only"]:::proc
    subgraph SAVE["Time saved per document (traditional to single-prompt)"]
      direction TB
      S1["IND + IRB package: 8-12 weeks to 1-4 days"]:::input
      S2["Protocol amendment + consent: 2-4 weeks to 1-2 days"]:::input
      S3["Cohort-review package: 1-2 weeks to ~1 day"]:::input
      S4["Clinical-hold response: 3-6 weeks to 2-4 days"]:::input
      S5["Annual DSUR: 4-8 weeks to 2-3 days"]:::input
      S6["CSR (ICH E3): 3-6 months to 1-2 weeks"]:::input
    end
    G["Months to a year saved cumulatively"]:::goal
    B1 --> B1D
    B3 --> B3D
    B2 --> LLM --> SAVE --> G
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef accent fill:#6C757D,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef ctx fill:#F5F5F5,stroke:#6C757D,stroke-width:1px,color:#000000
    classDef dec fill:#D9D9D9,stroke:#000000,stroke-width:1px,color:#000000
```

**Caption.** The three timeline buckets. Only the administrative and preparation
bucket compresses; the clinical / operational bucket and the fixed regulatory
review clocks do not. The repository LLM acts on the middle bucket only. The exact
per-document savings are: initial IND and IRB package 8 to 12 weeks reduced to 1 to
4 days; protocol amendment with synchronized consent 2 to 4 weeks to 1 to 2 days;
cohort-review package 1 to 2 weeks to about 1 day; complete clinical-hold response
3 to 6 weeks to 2 to 4 days; annual DSUR 4 to 8 weeks to 2 to 3 days; clinical
study report 3 to 6 months to 1 to 2 weeks, summing to months to a year saved
cumulatively.

**Role in the IND.** Renders in the General Investigational Plan (§4.1 Rationale),
quantifying the schedule value of the method for this submission.

**Source files.**
`trial-documents/final-paper/publication/sections/sec-04-results.tex` (Figure 19,
the three buckets, adapted in context with the per-document figures);
`trial-documents/final-paper/publication/sections/sec-05-discussion.tex` (the
cumulative-savings argument).
