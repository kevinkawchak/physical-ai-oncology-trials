# Figure 20 - Which custodian holds which artifact if the programme stops

**Type.** diagrams (python)-type, custody clusters. **Section.** §10, Build
Method and Reproducibility. **Perspective.** *What survives a termination at any
milestone, who is holding it, and how much of the work a third party could
reproduce from the public custodian alone.* No other figure in this paper looks
past the programme's end.

**Caption (3 balanced lines, 66 to 70 characters, numbered as printed).**

```
Figure 20. Four custodians and what each still holds if the programme
stops. Everything a third party needs to reproduce the computation
sits with the first custodian, which needs no one to stay in business.
```

## Specification, as a diagrams-library graph

No `.py` file is emitted.

```python
# illustrative only, not executed, no .py file is written to the repository
with Diagram("Artifact custody after a stop", direction="LR"):
    with Cluster("Public custodian, Zenodo and GitHub, permanent"):
        pub = [
            Node("Protocols, Phase 1 and 2"),
            Node("IND package, as deposited"),
            Node("QSP stack and VVUQ suite"),
            Node("Twelve milestone artifacts"),
            Node("Replay bundles and hashes"),
        ]
    with Cluster("Sponsor file, ChemicalQDevice, two years"):
        spn = [
            Node("Monitoring reports"),
            Node("3454 and 3455 filings"),
            Node("FDA correspondence"),
        ]
    with Cluster("Site file, UC San Diego, per policy"):
        sit = [Node("Source documents"), Node("Consent forms"), Node("Medical records")]
    with Cluster("Agency file, FDA IND"):
        fda = [Node("IND and amendments"), Node("Expedited safety reports")]
    pub >> Edge(label="reproducible without any custodian") >> Node("Third party")
    spn >> Edge(label="on request only", style="dashed") >> fda
    sit >> Edge(label="never leaves the site", style="dashed") >> fda
```

## The four custodians

| Custodian | Holds | Retention | Reachable by a third party |
|:--|:--|:--|:--|
| Public, Zenodo and GitHub | Protocols, IND package as deposited, QSP stack, VVUQ suite, twelve milestone artifacts, replay bundles and hashes | Permanent, DOI-addressed | Yes, without permission |
| Sponsor file, ChemicalQDevice | Monitoring reports, Forms 3454 and 3455, FDA correspondence | Two years past the last approval or discontinuation, 21 CFR §312.57 | On request only |
| Site file, UC San Diego Moores | Source documents, consent forms, medical records | Per site policy and 21 CFR §312.62 | No |
| Agency file, FDA | IND and amendments, expedited safety reports | Agency schedule | Through a request under the applicable disclosure rules |

## What a third party can reproduce from the public custodian alone

| Work product | Reproducible | What is needed |
|:--|:--|:--|
| The QSP simulation, 10 arms, 250 ODEs | Yes | The deposited stack and a machine |
| The 1000-patient digital twin | Yes | Same |
| The 55-test VVUQ suite and the 81.9 credibility score | Yes | Same |
| The interlock bench verification, 200 runs | Partly | The report and manifest are public; the rig is not |
| The replay of any advisory decision | Yes | The replay bundle and its hash |
| Any participant-level clinical result | No | Source documents, which never leave the site |
| This paper and all twenty figures | Yes | The LaTeX source in the repository |

Five of the seven are fully reproducible from the public custodian. That is the
strongest argument in the plan for funding a company with no institution behind
it: if the company dissolves at any milestone, the computational work does not
become unavailable, because it was never in the company's custody in the first
place.

## Glyph assignment

| Tile | Glyph macro | Tile style | Cluster |
|:--|:--|:--|:--|
| Protocols, Phase 1 and 2 | `\glyphdoc` | `dgtiled` | Public |
| IND package, as deposited | `\glyphflask` | `dgtile` | Public |
| QSP stack and VVUQ suite | `\glyphcpu` | `dgtilem` | Public |
| Twelve milestone artifacts | `\glyphchart` | `dgtile` | Public |
| Replay bundles and hashes | `\glyphlock` | `dgtiled` | Public |
| Monitoring reports | `\glyphmon` | `dgtileg` | Sponsor |
| Forms 3454 and 3455 | `\glyphdoc` | `dgtileg` | Sponsor |
| FDA correspondence | `\glyphlink` | `dgtileg` | Sponsor |
| Source documents | `\glyphdoc` | `dgtilek` | Site |
| Consent forms | `\glyphhand` | `dgtilek` | Site |
| Medical records | `\glyphdb` | `dgtilek` | Site |
| IND and amendments | `\glyphbank` | `dgtileg` | Agency |
| Expedited safety reports | `\glyphsignal` | `dgtileg` | Agency |

## TikZ construction notes

Canvas 14.6 by 9.0 cm. The public cluster is drawn larger than the other three
combined, because it holds more and because that proportion is the argument.

| Element | Style token | Placement |
|:--|:--|:--|
| Public tiles | `\dgnodew`, `\dgnode` | x = 0, 2.70, 5.40; y = 0 and -2.20 (five tiles) |
| Public cluster | `dgcluster`, solid `pablue1` border, `fit` five tiles and labels | `inner sep=7pt` |
| Sponsor tiles | `\dgnodeg` | x = 8.70, 11.40; y = 0 and -2.20 (three tiles) |
| Sponsor cluster | `dgcluster2`, `fit` three tiles and labels | `inner sep=7pt` |
| Site tiles | `\dgnodeg` with `dgtilek` | x = 0, 2.70, 5.40; y = -5.00 |
| Site cluster | `dgcluster2`, `fit` three tiles and labels | `inner sep=7pt` |
| Agency tiles | `\dgnodeg` | x = 8.70, 11.40; y = -5.00 |
| Agency cluster | `dgcluster2`, `fit` two tiles and labels | `inner sep=7pt` |
| Third party | `dgtiled` with `\glyphteam`, label beneath | x = 13.90, y = -2.60 |
| Reproducible edge | `dgedgeb`, 1 pt | Public cluster east to third party, `bend left=16` |
| On-request edges | `dgedged` | Sponsor to agency and site to agency, both vertical |
| Blocked edge | `dgedged` with `\pxmark` at midpoint | Site to third party, marked as never crossing |
| Retention badges | `d2cellg`, `minimum width=22mm` | Anchored south east on each cluster, carrying the retention term |
| Reproducibility count | `d2cellk`, `text width=26mm` | x = 13.90, y = -5.00, carrying five of seven |
| In-figure note | `pnote`, `text width=132mm` | x = 0, y = -8.20 |

Tile pitch is 27 mm horizontally and 22 mm vertically. The public cluster's
border is solid `pablue1` while the other three are dashed `pagrayd`, which is
the only place in the paper where a `dgcluster` border style carries meaning:
solid is permanent, dashed is conditional.

Edge discipline: four edges. The one long edge, public to third party, takes
`bend left=16` and is routed above the sponsor cluster's north edge with 7 mm
of clearance. The blocked edge is drawn to one third of its length and
terminated with `\pxmark`, so it reads as a path that does not complete rather
than as an edge someone forgot to finish.

## Repository sources

- `funding/supplementary/Physical AI Oncology Trial Founding Documents.md` - the deposited works, all DOI-addressed
- 21 CFR §312.57 and §312.62, the sponsor and investigator retention terms
- `funding/capitalization-plan/mermaid/fig-13-twelve-milestone-calendar.md` - the twelve artifacts in the public cluster
- `funding/pdac-funding-applications/final-apply/sections/sec-06-physical-ai-governance.tex` - the replay bundle and hash method
- `funding/pdac-funding-applications/final-apply/sections/sec-10-risks-and-limits.tex` - the reproduce-the-81.9 verification record
- `LICENSE` at the repository root - the MIT terms that make the public cluster reachable without permission
