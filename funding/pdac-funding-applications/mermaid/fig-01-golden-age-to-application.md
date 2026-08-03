# Figure 1 - From one paragraph of federal policy to ten addressed applications

**Type.** mermaid-type, flowchart. **Section.** §2, The Ten Applications.
**Perspective.** *How a single policy sentence becomes ten specific asks.* No
other figure in the paper traces the policy-to-recipient mapping; Figure 3
tabulates the ten applications but does not show where they come from.

**Caption (three balanced lines, 62 to 66 characters each).**

```
One sentence of federal policy, three mechanism families, and the
ten recipients each family reaches. The split is by mechanism, not
by agency, which is why one recipient is not a federal funder.
```

## Mermaid source

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    POL["Science: A New Golden Age<br/>prioritize the individual scientist<br/>over legacy institutions"]:::goal
    B200["approximately $200B<br/>annual federal R&D portfolio<br/>no systematic framework"]:::mid
    PER["Person-based,<br/>long horizon"]:::soft
    ORG["Organization-type,<br/>novel performer"]:::soft
    PART["Partnership and<br/>cost share"]:::soft
    A01["01 NIH Pioneer Award"]:::leaf
    A07["07 HHMI Investigator"]:::leaf
    A02["02 ARPA-H"]:::leaf
    A03["03 NSF TIP X-Labs"]:::leaf
    A09["09 Convergent FRO"]:::leaf
    A05["05 NIH SEED SBIR"]:::leaf
    A04["04 DOE Genesis Mission"]:::leaf
    A06["06 FNIH AMP"]:::leaf
    A08["08 NCI CTEP"]:::leaf
    A10["10 UC San Diego Moores"]:::gray
    POL --> B200
    B200 --> PER
    B200 --> ORG
    B200 --> PART
    PER --> A01
    PER --> A07
    ORG --> A02
    ORG --> A03
    ORG --> A09
    ORG --> A05
    PART --> A04
    PART --> A06
    PART --> A08
    PART --> A10
    classDef goal fill:#00417A,stroke:#00417A,stroke-width:1.5px,color:#FFFFFF
    classDef mid fill:#3C7DB2,stroke:#00417A,stroke-width:1px,color:#FFFFFF
    classDef soft fill:#DCE8F1,stroke:#3C7DB2,stroke-width:1px,color:#00417A
    classDef leaf fill:#FFFFFF,stroke:#00417A,stroke-width:0.8px,color:#000000
    classDef gray fill:#E9ECEF,stroke:#6C757D,stroke-width:0.8px,color:#000000
```

## TikZ construction notes

| Element | Style token | Placement |
|:--|:--|:--|
| Policy sentence | `mmgoal` | Rank 0, centred at x = 5.5 |
| $200B portfolio | `mmmid` | Rank 1, y = -1.9 |
| Three mechanism families | `mmsoft` | Rank 2, y = -3.8, x = 0.6 / 5.5 / 10.4 |
| Ten recipients | `mmin`, and `mmgray` for recipient 10 | Rank 3, y = -6.0, on a 2.35 pitch inside each family |
| Edges | `mmedgeb` down the spine, `mmedge` to leaves | No edge crosses a node: families are separated by 4.9 |

Recipient 10 is set in `mmgray` because it is the only recipient that is not a
funder. The distinction has to be visible without reading the labels.

## Repository sources

- `funding/science-golden-age/chunk-01-front-matter-and-summary.md` - the policy sentence
- `funding/science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md` - the $200B finding
- `funding/pdac-funding-applications/applications/README.md` - the ten recipients and their anchors
