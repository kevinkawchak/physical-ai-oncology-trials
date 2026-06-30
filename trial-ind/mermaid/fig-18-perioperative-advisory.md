## Figure 18. Perioperative daraxonrasib pause-and-restart advisory

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'11px','lineColor':'#6C757D'}}}%%
flowchart TB
    PAUSE["Daraxonrasib paused<br/>before surgery"]:::input
    IN1["Input 1: PJ ISGPS<br/>fistula grade A / B / C"]:::input
    IN2["Input 2: serum trough<br/>vs 0.5 ng/mL threshold"]:::input
    LLM["LLM advisory (hash-pinned)<br/>+ human confirmation<br/>(never autonomous)"]:::proc
    T7["Restart T+7d<br/>29 of 32 (Grade A,<br/>sub-threshold trough)"]:::goal
    T14["Restart T+14d<br/>3 of 32 (Grade B or<br/>delayed serum rise)"]:::step
    T21["Restart T+21d or hold<br/>0 of 32 (Grade C)"]:::dark
    PAUSE --> LLM
    IN1 --> LLM
    IN2 --> LLM
    LLM --> T7
    LLM --> T14
    LLM --> T21
    classDef goal fill:#000000,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#3F3F3F,stroke:#000000,stroke-width:1px,color:#FFFFFF
    classDef step fill:#9AA0A6,stroke:#000000,stroke-width:1px,color:#000000
    classDef input fill:#ECECEC,stroke:#3F3F3F,stroke-width:1px,color:#000000
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.2px,color:#FFFFFF
```

**Caption.** The perioperative daraxonrasib pause-and-restart advisory. The drug is
paused before surgery and the postoperative restart day is set by a hash-pinned,
human-confirmed LLM advisory (never autonomous) from two inputs: the
pancreaticojejunostomy ISGPS fistula grade (A, B, or C) and the serum daraxonrasib
trough relative to a 0.5 ng/mL threshold. In the 32-iteration deterministic sweep
the advisory recommended restart at T+7 days in 29 of 32 iterations (Grade A with
sub-threshold trough), T+14 days in 3 of 32 (Grade B or delayed serum rise), and
T+21 days or continued hold in 0 of 32 (reserved for Grade C).

**Role in the IND.** Renders in §3.1.5 (Route of Administration) and §6.1 (Study
Protocol), specifying the drug-administration decision rule.

**Source files.**
`trial-protocol/final-protocol/publication/sections/sec-06-intervention.tex` (the
advisory inputs, outputs, and the 32-iteration sweep distribution).
