# Figure 8 - The go / no-go gates a reviewer applies

**Type.** mermaid-type, flowchart with decisions. **Section.** §3, The Surgical
Set. **Perspective.** *The reviewer's own decision procedure, and which section
of the applications answers each gate.* Every other figure describes the
programme; this one describes the reader.

**Caption (three balanced lines, 64 to 68 characters each).**

```
The five questions a reviewer asks in order, and the section of the
application that answers each. A no at gate 1 or gate 2 ends the
review; a no at gates 3 to 5 changes the ask rather than ending it.
```

## Mermaid source

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart LR
    G1{"1. Is the person<br/>credible?"}:::dec
    G2{"2. Is the science<br/>plausible?"}:::dec
    G3{"3. Is the site<br/>reachable?"}:::dec
    G4{"4. Is the budget<br/>proportionate?"}:::dec
    G5{"5. Is the claim<br/>bounded?"}:::dec
    S1["Sec 1<br/>14 deposited works"]:::soft
    S2["Sec 3<br/>QSP, twin, RASolute 302"]:::soft
    S3["Sec 5<br/>Moores feasibility route"]:::soft
    S4["Sec 5<br/>$700K per year, gated"]:::soft
    S5["Back matter<br/>scope of claims"]:::soft
    END1["Review ends"]:::gray
    END2["Ask is revised"]:::gray
    FUND["Proceed"]:::goal
    G1 -->|yes| G2
    G2 -->|yes| G3
    G3 -->|yes| G4
    G4 -->|yes| G5
    G5 -->|yes| FUND
    G1 -->|no| END1
    G2 -->|no| END1
    G3 -->|no| END2
    G4 -->|no| END2
    G5 -->|no| END2
    S1 -.-> G1
    S2 -.-> G2
    S3 -.-> G3
    S4 -.-> G4
    S5 -.-> G5
    classDef dec fill:#CED4DA,stroke:#000000,stroke-width:0.8px,color:#000000
    classDef soft fill:#DCE8F1,stroke:#3C7DB2,stroke-width:1px,color:#00417A
    classDef gray fill:#E9ECEF,stroke:#6C757D,stroke-width:1px,color:#000000
    classDef goal fill:#00417A,stroke:#00417A,stroke-width:1.5px,color:#FFFFFF
```

## TikZ construction notes

| Element | Style token | Placement |
|:--|:--|:--|
| Five gates | `mmdec`, aspect 2.1 | One rank, y = 0, pitch 2.9 |
| Five answering sections | `mmsoft` | Rank y = 1.9, directly above the gate each answers |
| Two outcomes | `mmgray` | Rank y = -2.1, at x = 1.45 and x = 8.7 |
| Proceed | `mmgoal` | End of the rank, x = 12.8 |
| Answer edges | `mmedged` vertical | 1.9 of clear space, so a dashed answer edge never crosses a gate edge |

Placing each answering section directly above its gate means the pairing is read
from position, and the dashed edges only confirm it.

## Repository sources

- `funding/pdac-funding-applications/applications/app-01-nih-pioneer-award/` - §1, §3, §5 and back matter, the canonical section numbering
- `funding/potential-partners/UC-San-Diego/README.md` - gate 3, the feasibility route
- `funding/RFA-RM-27-001-v2/LaTeX Source Files.zip` - gate 4, the budget frame
