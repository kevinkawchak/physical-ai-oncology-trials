# Stage 1, sub-prompt 4 - `references.bib` and the back matter

## Goal

Build the bibliography once, at stage 1, and carry it unchanged through stages 2
and 3. Every entry that has a digital object identifier carries both a `doi`
field and a `url` field pointing at `https://doi.org/<doi>`, so the printed
reference shows the DOI as text with a clickable target and the link cannot run
off the right side of the page.

## Sources

| Source | Entries taken |
|:--|:--|
| `funding/move-in/inputs/READMES/README-LLM-Pancreatic-Oncology-Clinical-Trial-System-Large-Documents-Funding-and-AI-Peer-Review.md` | The twenty deposited papers with dates and DOIs |
| `funding/move-in/inputs/ChemicalQDevice_Accomplishments.docx` | The seventeen numbered references, including the three 2024 ChemRxiv and bioRxiv works |
| `funding/pdac-funding-applications/final-apply/references.bib` | Policy entries: the White House report, the FY 2028 budget priorities annex, Executive Order 14363, Executive Order 14303 |
| `funding/move-in/inputs/READMES/README-Physical-AI-Oncology-Clinical-Trial-Site-Complete-Documentation-Package.md` | The San Francisco package itself, DOI 10.5281/zenodo.19176370 |
| California and federal codified law | Health and Safety Code, Title 22, 21 CFR parts 11, 50, 54, 312 and 812, ICH E6(R3), and the California Environmental Quality Act |

## Rules

1. `unsrturl` bibliography style, not `unsrt`. Plain `unsrt.bst` reads neither
   the `doi` nor the `url` field, so a reference list built on DOIs prints none
   of them and offers nothing to click. This defect was found and fixed in
   `funding/capitalization-plan` and must not be reintroduced.
2. Every entry must be cited from the body. An uncited entry is a defect,
   because `unsrt` orders by first appearance and an uncited entry never
   appears.
3. No DOI is invented. The paper's own DOI stays `10.5281/zenodo.xxxxxxxx`.
4. A title containing a section symbol is brace-protected, because the `unsrt`
   family lowercases title fields.

## Back matter

`sec-16-backmatter.tex` carries, in this order: an abbreviations table set in
two column pairs at the body measure; a data and code availability paragraph
naming the repository and this subtree; author contributions and conflicts,
which must state the 21 CFR part 54 position, that the chief executive is not a
clinical investigator; a citation line ending in the placeholder DOI; and the
reference list.

## Acceptance

- `bibtex main` returns 0 errors and 0 warnings about missing fields.
- The compiled reference list shows a clickable DOI on every entry that has one.
- No reference line overflows the measure. The `\UrlBreaks` re-assertion after
  `url` and `hyperref` is what makes this true; it must be present.

## Commit

Two commits: `move-in/draft: references.bib` and
`move-in/draft: sec-16 backmatter`.
