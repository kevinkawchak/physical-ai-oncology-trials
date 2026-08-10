# draft-capital - Stage 6, the skeleton

[![Stage](https://img.shields.io/badge/Stage-6%20of%208-00417A.svg)](../sub-prompts/stage-6-draft-capital)
[![Sections](https://img.shields.io/badge/Sections-12-3C7DB2.svg)](sections)
[![Figure slots](https://img.shields.io/badge/Figure%20slots-20-6C757D.svg)](.)
[![Tables](https://img.shields.io/badge/Tables-3%20written%2C%2021%20indexed-6C757D.svg)](.)
[![Compile](https://img.shields.io/badge/pdfLaTeX-0%20errors-6C757D.svg)](.)
[![Overfull](https://img.shields.io/badge/Overfull%20boxes-0-9AA1A8.svg)](.)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

The first of three paper stages. This document already compiles, already
carries its final figure and table numbering, and already names the exact
repository file every later stage must read. It does not yet carry the
argument.

## Files

| File | Contents |
|:--|:--|
| [`main.tex`](main.tex) | Cover, badges, clickable contents, twelve `\input` lines |
| [`capstyle.sty`](capstyle.sty) | The shared style, adapted from the parent work |
| [`references.bib`](references.bib) | The parent bibliography plus nine codified entries |
| [`sections/`](sections) | One `.tex` per section, `sec-00` to `sec-11` |
| `draft-capital-LaTeX.zip` | Overleaf bundle of everything above |

## What `capstyle.sty` changes, and what it keeps

It is `funding/pdac-funding-applications/final-apply/applystyle.sty` with three
changes and no others.

| Change | From | To | Why |
|:--|:--|:--|:--|
| Figure spacing invariant | `\vskip 26pt` then `\vspace{-0.7cm}` | `\vskip 24.5pt` then `\vspace{-0.65cm}` | The master prompt fixes the caption gap at `-0.65cm`. The rigid skip is retuned so the frame-to-caption distance stays constant at 6.06 pt for every figure and every table |
| Cover macros | `\paymast`, `\paystrip`, `\paycell` | `\capmast`, `\capledger`, `\capcell` | A ruled panel with a top accent band over a three-cell money ledger, because this paper has one recipient and three numbers, not ten recipients and one |
| Diagram nudge | not present | `\DiagramXShift` plus a `changepage` carrier | The author's own correction 11 in `final-apply/publication/useredits.md` |

Everything else is kept unchanged and deliberately so: the five TikZ diagram
vocabularies (`mm*`, `uml*`, `d2*`, `dg*`, `gv*`), all 24 `\glyph*` pictograms,
the quantitative primitives, the eight-token palette with no black fill, the
`apptable` and `\tabcap` table system, the URL-breaking rules, and the
senior-author typographic penalties. A figure written for the parent work
compiles here without edit.

## The spacing invariant, stated once

Every figure and every table in all three stages is written as:

```latex
\end{appfig}
\vspace{-0.65cm}
\figcaption{...}
```

`appfig` closes with a rigid `\vskip 24.5pt` and `\figcaption` opens with
`\nointerlineskip`, so the distance from the frame rule to the first caption
line is exactly 24.5 pt minus 0.65 cm, that is **6.06 pt**, for every figure in
the paper, floating or inline, whatever precedes or follows it on the page.

## The twelve sections

| File | § | Title | Figure slots |
|:--|:--|:--|:--|
| `sec-00-front.tex` | 0 | Abstract, Executive Summary, Reader's Guide | Tables 2, 3 |
| `sec-01-novel-performer-case.tex` | 1 | The Novel-Performer Case | 1, 2, 3 |
| `sec-02-entity-and-asset.tex` | 2 | The Entity and the Asset | 4, 5, 6 |
| `sec-03-gate-and-programme.tex` | 3 | The $1.6M Gate and the $3.5M Programme | 7, 8, 9 |
| `sec-04-capital-bridge.tex` | 4 | Non-Dilutive to Dilutive Bridge | 10, 11, 12 |
| `sec-05-twelve-milestones.tex` | 5 | Twelve Milestones a Program Officer Can Audit | 13, 14, 15 |
| `sec-06-clinical-evidence.tex` | 6 | The Clinical Evidence a Funder Is Buying | 16, 17 |
| `sec-07-operating-plan.tex` | 7 | Small-Business Operating Plan | 18 |
| `sec-08-san-diego-traction.tex` | 8 | San Diego and the August 2026 Record | 19 |
| `sec-09-risks-and-limits.tex` | 9 | Risks, Stop Conditions, and What This Is Not | none |
| `sec-10-build-method.tex` | 10 | Build Method and Reproducibility | 20 |
| `sec-11-references-backmatter.tex` | 11 | Back Matter | none |

## Rule 5 source map

Every `\draftinstr` in this stage names a path with `\appfile`, so stage 7 has
nothing to search for. The complete permitted source set:

| Path | What stage 7 takes from it |
|:--|:--|
| `funding/science-golden-age/chunk-01-front-matter-and-summary.md` | The SBIR and STTR focus recommendation, and the $200 billion finding |
| `funding/science-golden-age/chunk-03-...-revitalizing-...md` | The NOVEL PERFORMERS section, its scale statement, and its Table 1 |
| `funding/science-golden-age/chunk-04-...-securing-dominance-...md` | Institution-agnostic grants and SBIR or STTR deployed strategically |
| `funding/science-golden-age/chunk-05-...-better-lives-...md` | Programs like SBIR open doors for technician-founded ventures |
| `funding/science-golden-age/chunk-08-annex-...md` | The 3 to 1 leverage target and the non-federal cost-share instruction |
| `funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex` | The four-layer $3,500,000 frame, reused verbatim |
| `funding/pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex` | The six clinical quantities and their stated limitations |
| `funding/pdac-funding-applications/final-apply/sections/sec-06-physical-ai-governance.tex` | The 3 ms and 500 ms stop figures and the trust-boundary argument |
| `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed/` | The $306,000 and $1,300,000 split and its term |
| `funding/pdac-funding-applications/applications/emailed-source/` | The nine applications emailed 2026-08-04 to 2026-08-08 |
| `funding/pdac-funding-applications/final-apply/publication/useredits.md` | The correction list applied in stage 8 |
| `funding/supplementary/source-files/Physical-AI-Oncology-Trial-Competition-Proposal.zip` | The January 13, 2026 baseline the asset register dates from |
| `funding/supplementary/Physical AI Oncology Trial Founding Documents.md` | The thirteen owned assets and their DOIs |
| `funding/RFA-RM-27-001-v2/` | The cover theme this paper varies from, and the budget statement |
| `funding/potential-partners/UC-San-Diego/` | The Moores record and the first-in-human positioning correction |
| `trial-protocol/`, `trial-ind/`, `trial-phase-2/`, `trial-documents/` | The protocol, IND, Phase 2 and guidance assets |
| `funding/capitalization-plan/{mermaid,plantuml,d2,diagrams-python,graphviz}/` | The twenty figure specifications the slots are sized for |

## Compile record

| Check | Result |
|:--|:--|
| `pdflatex` exit code | 0 |
| Overfull boxes | 0 |
| Underfull boxes above `\hbadness=4000` | 0 |
| Undefined citations | 0 |
| Undefined references | 0 |
| Bibliography entries rendered | 41 |
| Pages | 32 |
| Raster images | none |

`\nocite{*}` is issued in `sec-11` at this stage only. The draft body carries
no real `\cite`, because every citation is still named inside a drafting
instruction, so without it the reference list would be empty and the author
could not check at this stage that all 41 entries resolve, that every DOI prints
as text with a clickable target, and that no reference line runs past the right
margin. Stage 7 replaces every named key with a real `\cite` and removes the
`\nocite`, so the final list is ordered by first appearance under `unsrt`.

Two defects were found and fixed inside this stage rather than carried forward.
The three-cell cover ledger was 14.34 pt past the measure, because three cells
at `0.288\textwidth` plus two gaps at `0.056\textwidth` plus six `inner sep` of
4 pt exceeds the body width; every cell is now
`\dimexpr0.28\textwidth-9pt\relax` so its outer width is exactly
`0.28\textwidth`, and the gaps are `0.055\textwidth`. The path
`diagrams-python/` set in `\texttt` overflowed a 3.3 cm table column by
11.23 pt in three rows, because Courier has no break point in a path; all five
directory paths in Table 1 are now set with `\appfile`, which inserts a break
opportunity after every character.
