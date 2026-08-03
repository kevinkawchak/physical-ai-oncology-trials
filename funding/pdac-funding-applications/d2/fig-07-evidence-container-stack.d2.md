# Figure 7 - Four evidence tiers, nested by strength of claim

**Type.** d2-type, nested containers. **Section.** §5, Trial Evidence.
**Perspective.** *Containment, not sequence.* Application 01's Figure 2 sets the
same four tiers in a flat grid; here they are nested, because the paper's claim
is that each tier is contained by the next and none may be read as the one
outside it.

**Caption (three balanced lines, 62 to 66 characters each).**

```
Four evidence tiers drawn as containment rather than as a ladder.
Nothing inside a container may be cited as though it were the
container, which is the whole of the paper's evidentiary discipline.
```

## D2 source

```d2
proposed: "Tier 4: proposed clinical research" {
  style.stroke: "#6C757D"
  style.stroke-dash: 3
  registrational: "Tier 3: peer-reviewed clinical result" {
    style.fill: "#DCE8F1"
    verified: "Tier 2: verified computation" {
      style.fill: "#FFFFFF"
      insilico: "Tier 1: in silico exploration" {
        style.fill: "#E9ECEF"
        qsp: "QSP, 10 arms, 250 ODEs, mOS 12.8 vs 5.4"
        empirical: "Empirical triplicate, 100,000 patients"
      }
      twin: "Digital twin, credibility 81.9, 55 tests"
      vv40: "ASME V and V 40, ICH M15 alignment"
    }
    rasolute: "RASolute 302, mOS 13.2 vs 6.6, RAS G12"
  }
  phase1: "This Phase 1, up to 18 participants, no result yet"
}
```

## TikZ construction notes

| Element | Style token | Placement |
|:--|:--|:--|
| Outer container, tier 4 | `d2ghost` with dashed `pagraym` stroke | Fit to everything plus 8pt; dashed because it contains no evidence yet |
| Tier 3 | `d2cont` filled `pablue2` | Fit to tiers 1 and 2 plus the RASolute leaf |
| Tier 2 | `d2cont` filled white | Fit to tier 1 plus two leaves |
| Tier 1 | `d2cont2` filled `pagrayl` | Innermost, two leaves |
| Leaves | `d2soft` inside tiers 1 and 2, `d2key` for RASolute, `d2gray` for the Phase 1 | Leaf boxes 3.9cm wide, 0.5 apart vertically |
| Titles | `d2title` at each container's top-left, outside the child field | 1.2mm above the container edge |

Container titles sit outside the child field at the top-left corner, never
overlapping a leaf. The nesting is drawn on the background layer so no container
edge crosses a leaf box.

## Repository sources

- `funding/pdac-funding-applications/applications/app-01-nih-pioneer-award/sections/sec-03-evidence.tex` - the flat four-tier grid this figure re-reads as containment
- `funding/supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip` - every tier 1 and tier 2 number
- `funding/daraxonrasib-llm-story.md` - the tier 3 result and the three stated differences
