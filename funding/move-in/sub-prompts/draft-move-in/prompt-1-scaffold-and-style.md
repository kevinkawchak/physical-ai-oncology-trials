# Stage 1, sub-prompt 1 - scaffold and `movestyle.sty`

## Goal

Create `funding/move-in/draft-move-in/` and the single style file that all three
stages share. The style is written once, at this stage, and copied unchanged
into `full-move-in/` and `final-move-in/`, so a formatting fix made at stage 3
is a fix to one file rather than three.

## Inputs

| Path | What is taken from it |
|:--|:--|
| `funding/pdac-funding-applications/final-apply/applystyle.sty` | Typography block, `L`/`C`/`R`/`Y` column types, `\apptable`, `\tabcap`, `\bmhead`, `\keywords`, `\orcidicon`, the `\UrlBreaks` re-assertion after `url` and `hyperref`, and the compressed `\l@section` table of contents entry |
| `funding/capitalization-plan/final-capital/capstyle.sty` | The `unsrturl` bibliography style and the accent-colored printed DOI |
| `funding/move-in/inputs/Physical-AI-Oncology-Clinical-Trial-Site-Complete-Documentation-Package.zip` | `physical_ai_legislation.sty`: the redefined `\part` that adds a document to the contents without a blank page, and `\theHsection` / `\theHsubsection` keyed to a document counter so hyperref anchors stay unique across fifteen documents that each restart at `\section{1}` |

## What to delete from the parent style

Rule 3 forbids diagrams. Remove all five TikZ diagram vocabularies (`mm*`,
`uml*`, `d2*`, `dg*`, `gv*`), the `\appfig` frame, the `\appfloat` carrier, the
`\figcaption` macro, the `\figslot` placeholder, and the twenty-four vector
glyph macros. Keep TikZ loaded only for the three pieces of cover furniture that
survive: `\mvbadge`, `\mvbadgel` and `\orcidicon`. A palette audit is a grep for
`appfig`, which must return nothing.

## What to add

1. `\docpart{title}` - the per-document part heading. It steps a `docpart`
   counter, resets `section` to zero, writes a contents line, and sets the title
   as a ruled block. `main.tex` issues `\clearpage` before every one, so each
   document starts on its own page (clause J).
2. `\mvcover{...}` - the centered cover block adapted from the template theme.
3. `\mvstrip` - the fifteen-cell document strip, three rows of five.
4. `\mvrule{width}` - the accent rule between cover blocks.
5. `\draftnote{...}` - the bracketed drafting instruction carrier, stage 1 only.
   Every one names an exact repository path through `\mvfile`.
6. `\mvfile{path}` - a monospace repository path with a break opportunity after
   every character, so a path can never overflow the measure.
7. `\dlink{doi}` - printed DOI text with a clickable `https://doi.org/` target.

## Acceptance

- `pdflatex` on a two-line test document that loads the style returns 0 errors.
- `grep -c 'raggedright' movestyle.sty` is greater than zero and `grep -c
  'appfig\|figcaption\|mmbar\|d2cont' movestyle.sty` is zero.
- The geometry block sets a body measure that every table will match exactly.

## Commit

One commit, message `move-in/draft: movestyle.sty, the shared style for all
three stages`.
