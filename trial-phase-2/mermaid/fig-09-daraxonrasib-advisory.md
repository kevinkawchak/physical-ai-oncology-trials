## Figure 9. Daraxonrasib perioperative pause-and-restart advisory

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6B6B6B'}}}%%
flowchart TB
    PRE["Pre-operative hold<br/>daraxonrasib RP2D paused<br/>before surgery"]:::light
    IN1["Input: PJ ISGPS<br/>fistula grade (A / B / C)"]:::mid
    IN2["Input: serum trough<br/>vs 0.5 ng/mL"]:::mid
    ADV{"LLM-bound restart advisory<br/>(human confirmed,<br/>not autonomous)"}:::warn
    R7["Restart T+7<br/>Grade A,<br/>trough &lt;0.5 ng/mL"]:::goal
    R14["Restart T+14<br/>Grade B /<br/>delayed serum rise"]:::mid
    R21["Restart T+21 / hold<br/>Grade C<br/>(maps to &sect;7 stop rule)"]:::dark
    PRE --> IN1
    PRE --> IN2
    IN1 --> ADV
    IN2 --> ADV
    ADV --> R7
    ADV --> R14
    ADV --> R21
    classDef goal fill:#800020,stroke:#000000,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#2E2E2E,stroke:#000000,stroke-width:1.5px,color:#FFFFFF;
    classDef mid fill:#6B6B6B,stroke:#111111,stroke-width:1.5px,color:#FFFFFF;
    classDef warn fill:#C9C9C9,stroke:#000000,stroke-width:1.2px,color:#111111;
    classDef light fill:#F5F5F5,stroke:#800020,stroke-width:1.2px,color:#111111;
```

**Caption.** Daraxonrasib perioperative pause-and-restart advisory in Arm A at the established RP2D. The restart day is keyed to the pancreaticojejunostomy ISGPS fistula grade and the serum trough relative to 0.5 ng/mL: T+7 for Grade A with a sub-threshold trough, T+14 for Grade B or a delayed serum rise, and T+21 or continued hold for Grade C, which also maps to the &sect;7 drug stop rule. The advisory is LLM-bound and human-checked, never autonomous.

**Role in the protocol.** Renders the &sect;6.1.2 perioperative pause-and-restart advisory; the single drug-device coupling point in Arm A.

**Source files.** `sections/sec-06-intervention.tex` (pause-and-restart advisory, T+7/T+14/T+21, ISGPS grade, 0.5 ng/mL trough).
