# Figure 11 - Where every dollar lands, layer by layer

**Type.** d2-type, layers. **Section.** §8, Budget and Leverage.
**Perspective.** *Cash against contributed value.* Application 02's Figure 3
layers a single 36-month budget; this layers the five-year ask and separates the
non-federal contributed share, which no PART I figure does.

**Caption (three balanced lines, 63 to 67 characters each).**

```
Five years of cash on the left, contributed non-federal value on the
right, and the two layers that appear on both sides. The contributed
column is what the annex asks agencies to prioritise.
```

## D2 source

```d2
direction: right
cash: "Federal or foundation cash, $3,500,000 over 5 years" {
  style.fill: "#FFFFFF"
  clinical: "Clinical conduct, site and pharmacy: $1,600,000"
  regulatory: "IND maintenance and safety reporting: $720,000"
  engineering: "Interlock rig, logging, audit replay: $780,000"
  release: "Verification package and archive: $400,000"
}
contributed: "Non-federal contributed share, not requested in cash" {
  style.fill: "#E9ECEF"
  drug: "Investigational drug supply"
  crossref: "Regulatory cross-reference"
  theatre: "Operating theatre and robotic platform time"
  path: "Pathology and specimen handling"
  bio: "Bioanalytical and assay support"
}
cash.clinical -> contributed.theatre: "same activity, split funding"
cash.regulatory -> contributed.crossref: "same activity, split funding"
```

## TikZ construction notes

| Element | Style token | Placement |
|:--|:--|:--|
| Cash layers | `d2key`, `d2mid`, `d2soft`, `d2gray`, deepening downward | Left container, four layers 6.2cm wide, 1.15 apart |
| Cash amounts | Separate `d2cell` strips | 2.0cm wide, 3mm to the right of each layer, so the amount never crowds the label |
| Contributed layers | `d2cell` on `pagrayl` | Right container, five layers, same pitch |
| Split-funding edges | `d2edgeb` | Two edges only, from the two layers that appear on both sides |
| Containers | `d2cont` left, `d2cont2` right | Titles above, at the top-left corner |

The layers are ordered by when the money is committed rather than by when it is
spent, because a reviewer's question is what is obligated at signature.

## Repository sources

- `funding/RFA-RM-27-001-v2/LaTeX Source Files.zip` - the $700,000 per year and $3,500,000 total frame
- `funding/pdac-funding-applications/applications/app-02-arpa-h/sections/sec-05-budget-site.tex` - the 36-month layer split this figure extends to five years
- `funding/pdac-funding-applications/applications/app-06-fnih-amp/` - the contributed-share column
- `funding/science-golden-age/chunk-08-annex-...md` - non-federal cost share as an annex directive
