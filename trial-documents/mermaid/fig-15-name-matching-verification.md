## Figure 15. Five name-matching verifications for monitorability

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    subgraph CHECKS["OUTLINE Theme 2: Effective<br/>Verifications Prove Utility"]
        direction TB
        C1{"Check 1<br/>Grounding via<br/>Mermaid figures"}:::warn
        C2{"Check 2<br/>Figures match<br/>paper context"}:::warn
        C3{"Check 3<br/>AI files viewable<br/>on GitHub live"}:::warn
        C4{"Check 4<br/>Repo and directory<br/>names match"}:::warn
        C5{"Check 5<br/>Paper file names<br/>match repo files"}:::warn
    end

    E1["20+ python mermaid figures<br/>== identical LaTeX figures<br/>same complexity preserved"]:::input
    E2["Figure 15 verification text<br/>== sec-02-introduction.tex<br/>OUTLINE 2 claims"]:::input
    E3["Auto-commit / auto-PR<br/>20+ commits streamed live<br/>no held commits"]:::input
    E4["trial-documents/full-paper<br/>== repo directory on main<br/>analogous to trial-protocol"]:::input
    E5["full-paper/sections/<br/>sec-04-results.tex cited<br/>== file on disk"]:::input

    GOAL((("Grounded,<br/>monitorable<br/>paper"))):::goal

    SRC[/"prompts/prompt-paper.md<br/>OUTLINE 2"/]:::ctx

    SRC -. defines theme 2 .-> CHECKS

    E1 -- "evidence" --> C1
    E2 -- "evidence" --> C2
    E3 -- "evidence" --> C3
    E4 -- "evidence" --> C4
    E5 -- "evidence" --> C5

    C1 == "pass" ==> GOAL
    C2 == "pass" ==> GOAL
    C3 == "pass" ==> GOAL
    C4 == "pass" ==> GOAL
    C5 == "pass" ==> GOAL

    GOAL -. "any mismatch<br/>re-verify twice" .-> CHECKS

    ACC["Result: 1-4 day<br/>iteration, real-time trust"]:::accent
    GOAL --> ACC

    classDef goal   fill:#8B2E3F,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc   fill:#2F5D7C,stroke:#000000,stroke-width:1.4px,color:#FFFFFF
    classDef accent fill:#D08770,stroke:#000000,stroke-width:1.3px,color:#111111
    classDef input  fill:#BFD7EA,stroke:#2F5D7C,stroke-width:1.2px,color:#111111
    classDef ctx    fill:#F4F7F9,stroke:#6C757D,stroke-width:1.1px,color:#111111
    classDef warn   fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
```

**Caption.** This checklist shows the five name-matching and grounding verifications that establish a monitorable, trustworthy paper: (1) grounding through Mermaid figures, (2) figures matching paper context, (3) AI-generated files viewable on GitHub in real time, (4) paper repository and directory names matching the actual repository, and (5) paper file names matching repository file names. Each warn-class check (gray diamond) is supplied with a concrete input example, such as "full-paper/sections/sec-04-results.tex cited in text == file on disk", and each passing check feeds the maroon goal node "Grounded, monitorable paper". A looping edge returns control to the checklist whenever any mismatch is detected, enforcing the prompt directive to re-verify twice, while the terminal terracotta node ties successful verification to 1-4 day iteration and real-time reviewer trust.

**Role in the paper.** It appears in the Methods/Results section as the verification framework for OUTLINE theme 2 ("Effective Verifications Prove Utility"), and it becomes a TikZ mermaidfig in the draft, full, and final LaTeX stages.

**Source files.**
- prompts/prompt-paper.md (OUTLINE 2: Effective Verifications Prove Utility)
