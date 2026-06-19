## output-draft-bill

# Stage 5 narrative: the H. R. 9510 Bill v5.0 draft scaffold

*Independent research draft. Not an enacted law and not legal advice. All
dollar figures are illustrative unless tied to a cited statute or notice.*

Stage 5 wrote the **draft scaffold** of H. R. 9510 Bill v5.0, the Financial
Data Amendment, into `auto-bill-02/draft-bill`. The scaffold compiles in
Overleaf as committed (pdfLaTeX; pdflatex, bibtex, pdflatex, pdflatex) and
lays the complete v5.0 architecture so the full bill only has to execute the
bracketed [DRAFTING INSTRUCTIONS].

## What the scaffold fixes in place

1. **The amendment apparatus.** Caption, condensed referral block, A BILL,
   enacting clause, and the cover page closing with the ASCII overview of the
   v5.0 financial build (Cover Figure, from Stage 3), dated June 9, 2026.
2. **SECTION 1 and the clickable contents.** Five operative sections and five
   appendices, every entry linking to its part, with the page-filling `\vfill`
   distribution.
3. **The operative skeleton.** SEC. 2 opens the evidence-to-law-to-cost
   findings; SEC. 3 states the insertion of new section 515D after section
   515C (21 U.S.C. § 360e-4) and scaffolds the new financial-data record
   (515D(k)); SEC. 4 states the comparative-print convention and scaffolds
   the twelve reproduced sections (the eleven carried plus § 379j); SEC. 5
   states the financial-data transparency duty operatively and scaffolds the
   user-fee treatment, the authorization of appropriations, and the
   budgetary-effects clause.
4. **The five financial appendices**, each with instructions naming its exact
   upstream sources and its figure and table slots (A cost estimate; B
   verification economics; C the financial-data transparency standard; D the
   financial research-influence matrix; E development transparency).
5. **The merged style.** `usctitle.sty` v5.0.0 merges the v3.0 ASCII
   primitives and table columns with the v4.0 gray-scale Mermaid TikZ
   primitives and `\billbox`, and adds the right-aligned `R{<width>}` numeric
   column for dollar tables (Stage 2 decision).
6. **The full bibliography.** `references.bib` carries the complete Bill v3.0
   set unchanged plus the v5.0 financial layer (Statutory PAYGO, House Rule
   XXI, the Congressional Budget Act § 402, the GAO glossary and Green Book,
   OMB A-11, UMRA, §§ 379i-379j, MDUFA V and the rate notices, 21 CFR part
   54, Open Payments, the v4.0 DOI, and this repository) - 111 entries, all
   printed by `\nocite{*}` in the research-aid back matter.
7. **Back matter.** The Authorship, Notices, and Provenance block (with the
   v5.0 placeholder DOI per Rule 15 and the v4.0/v3.0/repository DOIs) and
   the Provenance and Research Sources bibliography.

## The slot ledger the full bill must fill

| Slot | Medium | Location |
|:--|:--|:--|
| Cover Figure | ASCII (rendered in this scaffold) | caption page |
| Fig. 1, 3, 6, 9, 10, 11 | gray-scale Mermaid as TikZ | SEC. 2, 3, 5; App. B, C, E |
| Fig. 2, 4, 5, 7, 8 | ASCII | SEC. 2, 3, 4, 5; App. A |
| Tbl. 1-6 | full-width tables (R columns for dollars) | SEC. 2, 3, 4, 5 |
| Tbl. 7-15 | full-width tables | App. A, B, C, D, E |

## Commit record (milestone M6)

One commit per file, pushed in real time: `main.tex`, `usctitle.sty`,
`references.bib`, the nine `sections/*.tex` scaffolds, this narrative, the
re-filed `prompt-draft-bill.md`, the directory `README.md`, and the
error-fix-and-bundle commit carrying `draft-bill-LaTeX.zip`.
