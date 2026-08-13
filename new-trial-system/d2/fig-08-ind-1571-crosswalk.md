# Figure 8 - Form FDA 1571 content items to generated files

**Type.** d2-type, sql tables. **Section.** §3, IND.
**Perspective.** *The crosswalk a reviewer actually needs: each codified 21 CFR
312.23 content item, the generated section file that satisfies it, and the
repository path where that file is deposited, drawn as three joined record
tables rather than as prose.* No other figure in this paper joins a regulatory
requirement to a file on disk; Figure 6 draws the IND as an assembled structure,
Figure 7 draws when it was assembled, and Figure 9 draws what would have to
fail for it to be refused.

**Caption (2 balanced lines, 73 and 71 characters, numbered as printed).**

```
Figure 8. Twelve codified IND content items joined to the section file that
satisfies each, and to the repository path where that file is deposited.
```

## D2 source

```d2
direction: right

cfr: "21 CFR 312.23 content item" {
  shape: sql_table
  style: { fill: "#FFFFFF"; stroke: "#800020" }
  item_id: "int" { constraint: primary_key }
  citation: "312.23(a)(1) forms"
  intro: "312.23(a)(3)(i) introductory statement"
  plan: "312.23(a)(3)(iv) general plan"
  brochure: "312.23(a)(5) investigator brochure"
  protocol: "312.23(a)(6) protocols"
  cmc: "312.23(a)(7) chemistry manufacturing"
  pharmtox: "312.23(a)(8) pharmacology toxicology"
  prior: "312.23(a)(9) previous human experience"
  additional: "312.23(a)(10) additional information"
  relevant: "312.23(a)(11) relevant information"
}

sec: "Generated section file" {
  shape: sql_table
  style: { fill: "#FFFFFF"; stroke: "#800020" }
  file_id: "int" { constraint: primary_key }
  item_id: "int" { constraint: foreign_key }
  name: "sec-NN-name.tex"
  chars: "character count at deposit"
  figures: "figures carried"
}

repo: "Repository deposit" {
  shape: sql_table
  style: { fill: "#FFFFFF"; stroke: "#800020" }
  path_id: "int" { constraint: primary_key }
  file_id: "int" { constraint: foreign_key }
  path: "trial-ind/final-ind/publication"
  doi: "10.5281/zenodo.21097442"
  version: "IND v1.0, repository v4.3.0"
}

cfr.item_id -> sec.item_id: "one to one"
sec.file_id -> repo.file_id: "one to one"
```

## TikZ construction table

Absolute coordinates. Canvas 15.0 by 8.8 cm, three record boxes left to right,
because a join reads horizontally.

| Element | Style token | Placement |
|:--|:--|:--|
| CFR record header | `d2cellh`, width 46 mm, height 0.52 cm | x = 0, y = 0 |
| CFR record rows, 10 | `d2celll`, width 46 mm, height 0.46 cm | x = 0, y = -0.52 down to -4.66, pitch 0.46 cm |
| Section record header | `d2cellh`, width 44 mm | x = 5.20, y = 0 |
| Section record rows, 5 | `d2celll`, width 44 mm | x = 5.20, y = -0.52 down to -2.36 |
| Repository record header | `d2cellh`, width 44 mm | x = 10.20, y = 0 |
| Repository record rows, 5 | `d2celll`, width 44 mm | x = 10.20, y = -0.52 down to -2.36 |
| Key marks | `\tiny` burgundy `PK` and `FK` glyphs | Right-aligned inside the relevant row, 2 mm inset |
| Join edge 1 | `d2edgeb`, 0.8 pt | From CFR row 2 east anchor to Section row 3 west anchor |
| Join edge 2 | `d2edgeb`, 0.8 pt | From Section row 2 east anchor to Repository row 3 west anchor |
| Join cardinality labels | `d2edge` label, white fill | Midpoint of each join edge |
| Deposit summary strip | `d2mid`, `text width=52mm` | x = 10.20, y = -3.85 |
| Count strip | `d2soft`, `text width=44mm` | x = 5.20, y = -3.85 |
| In-figure note | `pnote` | x = 0, y = -5.55, `text width=142mm` |

The three record boxes share one row height, 0.46 cm, and one header height,
0.52 cm, so the join edges leave and enter at the vertical center of a row
rather than at an arbitrary point on a box edge.

## Crosswalk table

| 21 CFR 312.23 item | Generated file | Deposit |
|:--|:--|:--|
| Cover letter, transmittal | `sec-00-cover-letter.tex` | `trial-ind/final-ind/publication` |
| (a)(1) Form FDA 1571 and 1572 | `sec-01-fda-forms.tex` | same |
| (a)(3)(i) Introductory statement | `sec-02-introduction.tex` | same |
| (a)(3)(iv) General investigational plan | `sec-03-general-investigational-plan.tex` | same |
| (a)(5) Investigator's brochure | `sec-04-investigator-brochure.tex` | same |
| (a)(6) Protocols | `sec-05-proposed-clinical-research.tex` | same |
| (a)(7) Chemistry, manufacturing, and control | `sec-06-cmc.tex` | same |
| (a)(8) Pharmacology and toxicology | `sec-07-pharmacology-toxicology.tex` | same |
| (a)(9) Previous human experience | `sec-08-previous-human-experience.tex` | same |
| (a)(10) Additional information | `sec-09-additional-information.tex` | same |
| (a)(11) Relevant information | `sec-10-relevant-information.tex` | same |
| References and back matter | `sec-11-references-backmatter.tex` | same |

The IND carries 22 grayscale figures in a single document sequence and section
scoped table numbering, and it is deposited at
`doi:10.5281/zenodo.21097442` at IND v1.0 and repository v4.3.0.

## Edge routing

Only two edges exist, both left to right, both between record boxes that share
a row height, so neither can cross a row. The first join leaves the CFR record
at y = -1.21 and enters the Section record at y = -1.44, a 2.3 mm drop over a
1.20 cm run, which is a straight line with no bend and clears both boxes. The
second join leaves the Section record at y = -1.44 and enters the Repository
record at the same height, a horizontal line. The cardinality labels sit at the
midpoint of each run with a white fill, punching a hole in the line beneath.
The two summary strips sit 1.49 cm below the last record row and are separated
from each other by 0.56 cm, so neither touches the other or a record box.

## Repository sources

- `trial-ind/final-ind/publication/LaTeX Source Files.zip` - the twelve section files, their names, and the codified item each satisfies
- `trial-ind/final-ind/README.md` - the 22-figure catalog, the section-scoped table numbering, and the deposit DOI
- `regulatory` and `national-platform/21cfr312_adapt` - the adapted 21 CFR 312 text the crosswalk is checked against
