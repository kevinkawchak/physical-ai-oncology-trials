# Figure 25 - Single-vendor exposure and the watermark chain

**Type.** diagrams (python)-type, clustered infrastructure. **Section.** §8,
Limitations and Future Work. **Perspective.** *The two structural limitations of
this system drawn with the mitigation attached to each exposure, so the
limitations section carries an engineering answer rather than an apology.* No
other figure in this paper is about the system's weaknesses; every other figure
describes what it produces or how.

**Caption (2 balanced lines, 70 and 72 characters, numbered as printed).**

```
Figure 25. Two structural exposures, single-vendor production and text
watermarking, each with the mitigation that reduces it and its residual.
```

## Python diagrams source

```python
from diagrams import Diagram, Cluster, Edge
from diagrams.generic.blank import Blank

with Diagram("Limitations and mitigations", show=False, direction="LR"):
    with Cluster("Exposure 1, single-vendor production"):
        e1 = [Blank("One model family writes all source"),
              Blank("One context window bounds each stage"),
              Blank("Vendor availability is a project risk")]

    with Cluster("Mitigation 1"):
        m1 = [Blank("Two independent reviewer manufacturers"),
              Blank("Machine-readable specifications, portable"),
              Blank("Every artifact deposited under a DOI")]

    with Cluster("Exposure 2, text watermarking"):
        e2 = [Blank("Generated text may carry a watermark"),
              Blank("Detection policy is not yet settled"),
              Blank("Journals differ on disclosure")]

    with Cluster("Mitigation 2"):
        m2 = [Blank("Provenance disclosed on the cover page"),
              Blank("Commit history is the primary record"),
              Blank("Preprint and repository, not journal gate")]

    residual = Blank("Residual: funding decision risk")

    for a, b in zip(e1, m1):
        a >> Edge(label="reduced by") >> b
    for a, b in zip(e2, m2):
        a >> Edge(label="reduced by") >> b
    m1[2] >> Edge(style="dashed") >> residual
    m2[2] >> Edge(style="dashed") >> residual
```

## TikZ construction table

Absolute coordinates. Canvas 14.8 by 9.0 cm. Two exposure clusters on the left,
two mitigation clusters on the right, one residual node at the foot.

| Element | Style token | Placement |
|:--|:--|:--|
| Exposure 1 tiles | `dgnodeg` with `dgtilek`, glyphs `\glyphai`, `\glyphdoc`, `\glyphcloud` | x = 1.05, y = 1.55, -0.35, -2.25; pitch 1.90 cm |
| Mitigation 1 tiles | `dgnode` with `dgtilem`, glyphs `\glyphteam`, `\glyphlink`, `\glyphlock` | x = 8.35, same three y values |
| Exposure 2 tiles | `dgnodeg` with `dgtilek`, glyphs `\glyphsignal`, `\glyphmon`, `\glyphbank` | x = 1.05, y = -4.55, -6.45, -8.35 |
| Mitigation 2 tiles | `dgnode` with `dgtilem`, glyphs `\glyphdoc`, `\glyphdb`, `\glyphserver` | x = 8.35, same three y values |
| Four cluster frames | `dgcluster2` for exposures, `dgcluster` for mitigations | `fit` over each block, `inner sep=7pt` |
| Cluster titles | `dgctitle2` for exposures, `dgctitle` for mitigations | Anchored north west, 1.5 mm inset |
| Residual node | `dgnodew` with `dgtiled`, glyph `\glyphchart`, `line width=1pt` | x = 12.95, y = -4.45, outside every cluster |
| Residual halo | Burgundy ring, 0.6 pt, radius 7.5 mm | Centered on the residual tile |
| Pairing edges | `dgedge` | Six horizontal edges at the six shared y values, each 4.90 cm long |
| Residual edges | `dgedged` | Two dashed edges from mitigation tiles at y = -2.25 and y = -8.35 to the residual tile |
| Band divider | Slate Gray hairline, 0.4 pt, dashed | Horizontal at y = -3.40, full canvas width, separating exposure 1 from exposure 2 |
| In-figure note | `pnote` | x = 0.35, y = -9.55, `text width=140mm` |

Exposures and mitigations share six y values exactly, so every pairing edge is
horizontal and no edge is oblique. The band divider at y = -3.40 sits midway
between the two exposure clusters and touches neither frame.

## Exposure, mitigation, and residual

| Exposure | Mitigation | Residual after mitigation |
|:--|:--|:--|
| One model family writes all source, so a defect in its idiom appears in every artifact | Two independent reviewer manufacturers read every artifact at defined milestones | A defect all three manufacturers share would survive |
| One context window bounds each stage, so a stage larger than the window truncates | Specifications are machine-readable and portable, so a stage can be re-executed piecewise | Re-execution costs a stage, not a project |
| Vendor availability is a project risk | Every artifact is deposited under a DOI as it is produced | Work already deposited is unaffected by later unavailability |
| Generated text may carry a watermark | Provenance is disclosed on the cover page of every deposited work | Disclosure does not remove the watermark |
| Detection policy is not yet settled | Commit history, not a claim of authorship, is the primary record | Policy may change after deposit |
| Journals differ on disclosure requirements | Deposit is to a preprint and a repository rather than through a journal gate | A funder that requires journal peer review is not reached |

The single residual node is deliberate: after all six mitigations, exactly one
consequence remains, and it is a funding decision risk rather than a scientific
or a safety one. That is the sentence the Limitations section is written to
support.

## Glyph table

| Tile | Pictogram | Why this glyph |
|:--|:--|:--|
| One model family writes all source | `\glyphai` | The model itself is the exposure |
| One context window bounds each stage | `\glyphdoc` | A document that does not fit |
| Vendor availability | `\glyphcloud` | A service outside the building |
| Two reviewer manufacturers | `\glyphteam` | More than one party |
| Portable specifications | `\glyphlink` | A form that transfers |
| DOI deposit | `\glyphlock` | An immutable record |
| Watermark present | `\glyphsignal` | An embedded signal |
| Detection policy unsettled | `\glyphmon` | A watching function not yet defined |
| Journal disclosure variance | `\glyphbank` | An institutional rule |
| Cover page disclosure | `\glyphdoc` | A statement in a document |
| Commit history | `\glyphdb` | A store of record |
| Preprint and repository | `\glyphserver` | A host that is not a journal |
| Residual funding decision risk | `\glyphchart` | A decision measured in money |

## Edge routing

Eight edges. The six pairing edges are strictly horizontal at six distinct y
values 1.90 cm apart and run through the clear corridor between x = 2.55 and
x = 6.85, where no tile sits, so none can cross another or touch a label box.
The two dashed residual edges leave mitigation tiles at y = -2.25 and y = -8.35
and converge on the residual tile at y = -4.45 from above and below, entering at
its north west and south west anchors respectively; their runs pass to the right
of the mitigation cluster frames, at x above 10.05, where the canvas is empty.

## Repository sources

- `new-trial-system/prompts/prompt-new-trial.md` - the single-model instruction that creates exposure 1
- `new-trial-system/inputs/AI_Peer_Review_Acceleration_of_LLM_Generated_Glioblastoma_Clinical_Trial_Patient_Matching_ML__FDA_ICH_ISO__and_FastAPI.zip` - the multi-manufacturer review that mitigates it, and the study's own recorded limitations
- `funding/RFA-RM-27-001-v2/LaTeX Source Files.zip` - the disclosure language stating that the models are drafting and peer-review aids and not applicants, investigators, sponsors, regulators, clinicians, or decision-makers
- `funding/capitalization-plan/final-capital/publication/LaTeX Source Files.zip` - the risks and limits section this figure's structure is adapted from
