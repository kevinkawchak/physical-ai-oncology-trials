## Figure 9. Daraxonrasib perioperative pause-and-restart advisory

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    PRE["Pre-operative hold<br/>daraxonrasib paused before surgery"]:::light
    IN1["Input: PJ ISGPS fistula grade<br/>(A / B / C)"]:::mid
    IN2["Input: serum daraxonrasib<br/>trough vs 0.5 ng/mL"]:::mid
    ADV{"LLM-bound restart advisory<br/>(human confirmed)"}:::warn
    R7["Restart T+7d<br/>Grade A and trough < 0.5<br/>(29 of 32 sim iterations)"]:::goal
    R14["Restart T+14d<br/>Grade B / delayed serum rise<br/>(3 of 32)"]:::mid
    R21["Restart T+21d / hold<br/>Grade C (0 of 32 in sweep)"]:::dark
    PRE --> IN1 --> ADV
    PRE --> IN2 --> ADV
    ADV --> R7
    ADV --> R14
    ADV --> R21
    classDef goal fill:#00417A,stroke:#000000,stroke-width:2px,color:#FFFFFF
    classDef mid fill:#6C757D,stroke:#111111,stroke-width:1.5px,color:#FFFFFF
    classDef light fill:#FFFFFF,stroke:#00417A,stroke-width:1.2px,color:#111111
    classDef warn fill:#D9D9D9,stroke:#000000,stroke-width:1.2px,color:#111111
    classDef dark fill:#222222,stroke:#000000,stroke-width:1.5px,color:#FFFFFF
```

**Caption.** The perioperative advisory pauses daraxonrasib before surgery and
recommends a restart day keyed to the pancreaticojejunostomy ISGPS fistula grade
and the serum trough relative to 0.5 ng/mL: T+7d when the fistula is Grade A and
the trough is sub-threshold (29 of 32 simulation iterations), T+14d for Grade B
with delayed serum rise (3 of 32), and T+21d or continued hold for Grade C (0 of
32 in the sweep). The advisory is bound and human-confirmed, never autonomous.

**Role in the protocol.** Defines the &sect;6.1.2 dosing/restart logic and links
to counterfactual Scenario C (restart mistiming).

**Source files.** `inputs/2030-pdac-1min-final-paper/sections/results.tex`
(restart distribution table, 0.5 ng/mL trough, 29/3/0 of 32); `DARAXONRASIB`
citations in `prompts/prompt-protocol.md`.
